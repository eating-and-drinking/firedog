"""
scripts/launch.py
系统主入口：组装各模块并启动完整运行时

支持三种启动模式：
  --mode full         完整系统（语音 + Agent + 本体）
  --mode voice_only   仅语音管道（阶段一验收用）
  --mode agent_cli    命令行文字输入模式（调试用）
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import click
import yaml

# 项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.safety import SafetyGuard
from src.utils.metrics import start_metrics_server

log = get_logger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 简单环境变量展开（$VAR 或 ${VAR}）
    import re
    def _replace(m):
        var = m.group(1) or m.group(2)
        return os.environ.get(var, m.group(0))
    raw = re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, raw)
    return yaml.safe_load(raw)


@click.command()
@click.option("--config", default="config/config.yaml", help="配置文件路径")
@click.option(
    "--mode",
    type=click.Choice(["full", "voice_only", "agent_cli"]),
    default="full",
    help="启动模式",
)
@click.option("--mock-robot", is_flag=True, default=False, help="使用 mock 机器狗（无硬件时）")
def main(config: str, mode: str, mock_robot: bool) -> None:
    cfg = load_config(config)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("file"))

    log.info("system_starting", mode=mode, mock_robot=mock_robot)

    # Prometheus 监控
    if cfg.get("metrics", {}).get("enabled", True):
        start_metrics_server(cfg["metrics"].get("export_port", 8000))

    # 安全守卫
    robot_cfg = cfg.get("robot", {})
    safety_cfg = robot_cfg.get("safety", {})
    safety = SafetyGuard(
        max_linear_vel=safety_cfg.get("max_linear_velocity", 0.8),
        max_angular_vel=safety_cfg.get("max_angular_velocity", 1.5),
        battery_low=safety_cfg.get("battery_low_threshold", 15.0),
        battery_critical=safety_cfg.get("battery_critical_threshold", 5.0),
        joint_temp_max=safety_cfg.get("joint_temp_max", 80.0),
    )

    # ROS 2 桥 / SDK
    use_mock = mock_robot or robot_cfg.get("backend", "ros2") == "mock"
    from src.integration.ros2_bridge import ROS2Bridge
    bridge = ROS2Bridge(
        namespace=robot_cfg.get("ros2", {}).get("namespace", "/robot_dog"),
        safety=safety,
        mock=use_mock,
    )
    safety._e_stop_cb = bridge.publish_stop

    # 技能层
    from src.skills.motion_skills import MotionSkills
    from src.skills.sensor_skills import SensorSkills
    from src.skills.skill_registry import SkillRegistry
    motion = MotionSkills(bridge, safety)
    sensor = SensorSkills(bridge)
    registry = SkillRegistry(motion, sensor)

    # Agent
    from src.agent.tools import build_tools
    from src.agent.graph import RobotDogAgent
    tools = build_tools(registry)
    llm_cfg = cfg.get("llm", {})
    agent = RobotDogAgent(
        tools=tools,
        llm_model=llm_cfg.get("model", "gpt-4o-mini"),
        llm_api_key=os.environ.get("OPENAI_API_KEY", llm_cfg.get("api_key", "")),
        llm_base_url=llm_cfg.get("base_url", ""),
        max_iterations=cfg.get("agent", {}).get("max_iterations", 10),
        memory_window=cfg.get("agent", {}).get("memory_window", 20),
    )

    if mode == "agent_cli":
        _run_cli(agent)
        return

    if mode == "voice_only":
        _run_voice(cfg, agent, voice_only=True)
    else:
        _run_voice(cfg, agent, voice_only=False)


def _run_cli(agent) -> None:
    """命令行文字交互模式（调试 Agent 逻辑）。"""
    from rich.console import Console
    console = Console()
    console.print("[bold green]机器狗 Agent CLI 启动[/bold green]（输入 'quit' 退出）")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in ("quit", "exit", "退出"):
            break
        if not user_input:
            continue
        reply = agent.handle(user_input)
        console.print(f"[cyan]Robot >[/cyan] {reply}\n")

    console.print("[yellow]已退出[/yellow]")


def _run_voice(cfg: dict, agent, voice_only: bool = False) -> None:
    """启动完整语音管道。"""
    from src.voice.voice_pipeline import VoicePipeline, VoicePipelineConfig
    from src.voice.asr import ASRConfig
    from src.voice.tts import TTSConfig
    from src.voice.vad import VADConfig
    from src.voice.wake_word import WakeWordConfig

    v = cfg.get("voice", {})
    pipeline_cfg = VoicePipelineConfig(
        wake_word=WakeWordConfig(
            model_name=v.get("wake_word", {}).get("model", "hey_robot"),
            threshold=v.get("wake_word", {}).get("threshold", 0.7),
        ),
        vad=VADConfig(
            threshold=v.get("vad", {}).get("threshold", 0.5),
            speech_pad_ms=v.get("vad", {}).get("speech_pad_ms", 400),
        ),
        asr=ASRConfig(
            model_id=v.get("asr", {}).get("model_id", "iic/SenseVoice-Small"),
            language=v.get("asr", {}).get("language", "zh"),
            device=v.get("asr", {}).get("device", "cpu"),
            use_itn=v.get("asr", {}).get("use_itn", True),
        ),
        tts=TTSConfig(
            backend=v.get("tts", {}).get("backend", "cosyvoice"),
            model_dir=v.get("tts", {}).get("model_dir", "./Fun-CosyVoice3-0.5B-2512"),
            prompt_speech=v.get("tts", {}).get("prompt_speech", ""),
            instruct=v.get("tts", {}).get("instruct", ""),
            speed=v.get("tts", {}).get("speed", 1.0),
            sample_rate=v.get("tts", {}).get("sample_rate", 24000),
            kokoro_voice=v.get("tts", {}).get("kokoro_voice", "zh_female_1"),
        ),
        sample_rate=v.get("audio", {}).get("sample_rate", 16000),
        input_device=v.get("audio", {}).get("input_device", None),
        output_device=v.get("audio", {}).get("output_device", None),
        silence_timeout_s=v.get("timeouts", {}).get("silence_timeout_s", 5.0),
    )

    pipeline = VoicePipeline(
        config=pipeline_cfg,
        llm_handler=agent.handle,
    )

    def _shutdown(sig, frame):
        log.info("shutdown_signal_received")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.start()
    log.info("system_ready", mode="voice", mock=("voice_only" if voice_only else "full"))

    print("\n✅ 系统已就绪，等待唤醒词…（Ctrl+C 退出）\n")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
