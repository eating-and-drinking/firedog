# 机器狗 Agent 系统

本项目由[eating-and-drinking](https://github.com/eating-and-drinking)在火狗智能公司开发，是四足机器狗智能体开发框架的第一版，覆盖双向语音交互、本体集成与自主任务执行。

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                    语音交互层 (Voice Layer)                │
│  Wake Word → VAD → ASR → LLM → TTS                      │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│                    Agent 决策层 (Agent Layer)              │
│  LangGraph Workflow / Tool Dispatcher / Memory           │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│                    技能层 (Skills Layer)                   │
│  Motion / Sensor / Navigation / Safety Guard             │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│               本体控制层 (Robot SDK / ROS 2)               │
│  ROS2 Nodes / SDK Wrapper / Hardware Interface           │
└──────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp config/config.example.yaml config/config.yaml

# 3. 启动系统
python scripts/launch.py --mode full

# 4. 仅语音模式（阶段一）
python scripts/launch.py --mode voice_only
```

## 目录结构

```
robot_dog_agent/
├── config/               # 配置文件
├── src/
│   ├── voice/            # 考核项一：双向语音交互
│   │   ├── wake_word.py        # 唤醒词检测
│   │   ├── vad.py              # 语音活动检测
│   │   ├── asr.py              # 语音识别
│   │   ├── tts.py              # 语音合成
│   │   └── voice_pipeline.py  # 端到端语音管道
│   ├── skills/           # 考核项二：技能层
│   │   ├── motion_skills.py    # 运动控制技能
│   │   ├── sensor_skills.py    # 传感器技能
│   │   └── skill_registry.py  # 技能注册表
│   ├── agent/            # 考核项三：Agent 架构
│   │   ├── graph.py            # LangGraph 工作流
│   │   ├── tools.py            # Tool 定义
│   │   └── memory.py           # 记忆模块
│   ├── integration/      # 本体集成
│   │   ├── ros2_bridge.py      # ROS 2 通信桥
│   │   └── sdk_wrapper.py      # SDK 封装
│   └── utils/            # 工具类
│       ├── logger.py
│       ├── metrics.py
│       └── safety.py
├── tests/                # 测试套件
├── docs/                 # 文档
├── docker/               # 容器化部署
└── launch/               # ROS 2 Launch 文件
```

## 验收标准

| 考核项 | 指标 | 目标值 |
|--------|------|--------|
| 唤醒成功率 | Wake Word | ≥ 95% |
| 误唤醒 | 安静环境 | ≤ 1次/小时 |
| ASR 字准率 | 常用指令 | ≥ 90% |
| 端到端时延 | 说完→播报 | ≤ 2.5s |
| 打断响应 | 开口→停播 | ≤ 1.5s |

## 技术选型

- **ASR**: FasterWhisper (本地)
- **VAD**: Silero VAD
- **唤醒词**: OpenWakeWord
- **TTS**: Kokoro TTS (本地) / ElevenLabs (云端)
- **Agent**: LangGraph
- **LLM**: OpenAI GPT-4o / 本地 Qwen
- **ROS**: ROS 2 Humble/Jazzy
