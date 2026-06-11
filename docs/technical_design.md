# 四足机器狗 Agent 系统技术设计文档

## 一、架构总览

本系统按"分层解耦"原则构建，共四层：

```
语音交互层  ──►  Agent 决策层  ──►  技能层  ──►  本体控制层
(Voice)         (LangGraph)       (Skills)      (ROS 2 / SDK)
```

各层职责单一，跨层通过定义清晰的接口交互，Agent 层不侵入实时控制回路。

---

## 二、考核项一：双向语音交互系统

### 2.1 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 唤醒词 | OpenWakeWord | 开源、ONNX推理、支持自定义模型 |
| VAD | Silero VAD | 轻量、准确率高、支持流式推理 |
| ASR | FasterWhisper (medium) | 比 Whisper 快 4x，支持中文、本地部署 |
| TTS | Kokoro（本地）/ edge-tts（备用） | 本地无网延迟低；edge-tts 免费云端备用 |
| 回声消除 | sounddevice + 软件 AEC | 硬件麦克风阵列接入时可替换为硬件 AEC |

### 2.2 打断（Barge-in）实现

```
TTS 播放线程（50ms 块播放）
         ↑
音频回调检测 VAD 概率 ≥ threshold
         ↓
interrupt_flag.set() → 下一个 50ms 块停止播放
```
目标：用户开口 → 停止播报 ≤ 1.5s（实测约 50-200ms）

### 2.3 端到端时延分解

```
用户说完 → VAD 端点检测 (~200ms)
         → ASR 识别 (~500-800ms, FasterWhisper medium)
         → LLM 推理 (~800-1200ms, GPT-4o-mini)
         → TTS 合成 (~200-400ms, Kokoro 本地)
         → 播放开始
总计目标: ≤ 2500ms
```

---

## 三、考核项二：本体集成与技能层

### 3.1 ROS 2 通信架构

| 话题/动作 | 类型 | 方向 | 用途 |
|-----------|------|------|------|
| /robot_dog/cmd_vel | geometry_msgs/Twist | Publish | 速度指令 |
| /robot_dog/odom | nav_msgs/Odometry | Subscribe | 里程计 |
| /robot_dog/imu/data | sensor_msgs/Imu | Subscribe | 姿态 |
| /robot_dog/battery_state | sensor_msgs/BatteryState | Subscribe | 电量 |
| /robot_dog/joint_states | sensor_msgs/JointState | Subscribe | 关节 |

### 3.2 技能层设计原则

1. **参数校验**：Pydantic 模型约束入参范围
2. **安全边界**：SafetyGuard 前置检查，clip_velocity 裁剪
3. **数据陈旧防护**：传感器数据超过 2s 视为陈旧，返回 None
4. **异常处理**：所有 publish 包裹 try/except，返回 SkillResult
5. **接口标准化**：统一返回 SkillResult(success, message, data)

---

## 四、考核项三：Agent 架构

### 4.1 LangGraph ReAct 工作流

```
用户输入
    │
    ▼
[agent_node]  LLM 推理，决定下一步
    │
    ├─ has tool_calls ──► [tools_node]  执行技能
    │                          │
    │                          └──────► [agent_node] （循环）
    │
    └─ no tool_calls ──► END  返回最终回复
```

### 4.2 任务示例：巡检后返回

```
用户：去前方 3 米处巡检一下，看看有没有异常，然后回来

Agent 规划：
  1. get_status()         → 确认当前状态正常
  2. move_to_position(3,0) → 前往目标点
  3. get_imu()             → 检查姿态（模拟传感巡检）
  4. move_to_position(0,0) → 返回原点
  5. 汇报结果
```

### 4.3 安全设计

- `max_iterations=10`：防止 Agent 死循环
- 系统提示词中硬编码"停止"关键词立即触发 stop 工具
- SafetyGuard 在技能层独立运行，不受 LLM 指令绕过
- 急停回调直接发布 /cmd_vel 零速，不经过 Agent 层

---

## 五、部署与监控

### 5.1 本地部署

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 OPENAI_API_KEY
python scripts/launch.py --mode full --mock-robot  # 无硬件调试
python scripts/launch.py --mode full               # 实机
```

### 5.2 容器化部署

```bash
cd docker
docker compose up -d
```

### 5.3 Prometheus 指标

访问 `http://localhost:8000/metrics` 查看：
- `voice_e2e_latency_seconds` - 端到端时延分布
- `barge_in_latency_seconds` - 打断响应时延
- `wake_word_detections_total` - 唤醒统计
- `robot_battery_percent` - 实时电量
- `agent_tool_calls_total` - 工具调用统计

---

## 六、已知局限与后续优化

1. **导航**：`move_to_position` 当前为简单比例控制，生产环境建议接入 Nav2
2. **感知**：暂无摄像头/激光雷达集成，后续可接入用于避障
3. **ASR 噪声**：在强噪声环境（> 70dB）建议切换到云端 ASR 或配合麦克风阵列
4. **本地 TTS**：Kokoro 模型首次加载约需 2-3s，建议在启动时预热
