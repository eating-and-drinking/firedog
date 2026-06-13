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

## 技术选型（当前实际栈）

- **ASR**: SenseVoice-Small (FunASR, 本地 `./SenseVoiceSmall`)
- **VAD**: Silero VAD（流式，每用途独立实例）
- **唤醒词**: OpenWakeWord（无可用中文模型时自动降级为 VAD+ASR 关键词兜底）
- **TTS**: CosyVoice3 (本地 `./Fun-CosyVoice3-0.5B-2512`)；备用 Kokoro / ElevenLabs / edge-tts
- **LLM**: Qwen2.5-3B-Instruct（本地 transformers 4.51.x，流式生成按句送 TTS）
- **声纹**: resemblyzer d-vector（打断者身份验证）
- **Agent**: LangGraph（规划中）
- **ROS**: ROS 2 Humble/Jazzy（规划中）

## 音频前端（实际部署必读）

麦克风信号链：`raw → mic_gain → RNNoise 降噪 → 有界队列 → VAD/ASR/声纹`

- **降噪**: RNNoise（`voice.denoise`，CPU 实时 2.3ms/chunk，~24ms 均匀延迟，噪底 -17dB），
  与 Pipecat 等生产语音框架同款方案。**默认关闭**：实测 Silero VAD 与 SenseVoice
  自身已抗噪（0dB 噪声下 VAD 零误触发、ASR 全对），RNNoise 伪影反而让 ASR 出错字；
  仅在强噪声环境（户外风噪/人群）导致待机误触发时开启。
- **回声消除**: PulseAudio WebRTC AEC，运行 `scripts/setup_aec.sh` 启用（见下）。
- **看门狗**: 麦克风静默超过 `voice.audio.watchdog_timeout_s` 自动重连（USB 拔插恢复）。
- **唤醒**: 主路为 sherpa-onnx KWS 流式关键词检测（中文，CPU 实时，~80ms 级延迟，
  待机不占 GPU；关键词在 config 改拼音即生效，无需训练）。模型下载：
  ```bash
  curl -sL https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2 | tar xj
  ```
  模型目录不存在时自动降级为 ASR 兜底（带拼音模糊匹配 `wake_word.fuzzy_pinyin`，
  容忍 SenseVoice 近音错字"机器狗"→"一气狗"）。

## 低延迟与打断

- LLM 流式生成 + 按句切分 + TTS 合成/播放双线程流水线：首音延迟 ≈ 首句生成 + 首句合成，
  而非全文生成 + 全文合成。
- 打断（barge-in）成功后，触发打断的语音直接续接为下一句的开头，无需重说。
- 回答完后保持聆听（连续对话），静音超时才回待机。
- 回声消除：运行 `scripts/setup_aec.sh` 启用 PulseAudio WebRTC AEC+AGC+NS 后，
  `mic_gain` 必须改为 1.0（增益由 AGC 接管，两级增益叠加会放大环境音），
  `voice.barge_in.threshold` 降到 0.5、`post_tts_cooldown_ms` 降到 200，
  打断不再需要提高嗓门。脚本注释里有两次现场事故的完整记录，调参前先读。
