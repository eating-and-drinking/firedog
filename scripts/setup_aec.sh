#!/usr/bin/env bash
# scripts/setup_aec.sh
# 启用 PulseAudio WebRTC 回声消除（AEC），供 barge-in 打断使用。
#
# 原理：module-echo-cancel 以扬声器输出为参考信号，从麦克风信号中
# 减去回声，得到干净的人声。启用后：
#   1. config.yaml 中 voice.audio.input_device 改为 null（走系统默认源，
#      即下面创建的 echocancel 源；auto:ListenGo 会绕过 AEC，不要再用）
#   2. voice.barge_in.threshold 可从 0.7 降到 0.4 左右
#   3. voice.timeouts.post_tts_cooldown_ms 可从 800 降到 200
#
# 注意：仅对当前 PulseAudio 会话生效，重启后失效。
# 持久化：把 load-module 行（去掉 pactl 前缀）追加到 /etc/pulse/default.pa。
set -euo pipefail

if ! command -v pactl >/dev/null; then
    echo "未找到 pactl，请先安装 pulseaudio-utils" >&2
    exit 1
fi

# 重复执行时先卸载旧实例
pactl unload-module module-echo-cancel 2>/dev/null || true

# aec_args（两次现场事故换来的结论，改之前先看完）:
#   digital_gain_control=1 必须开：ListenGo 电平极低（RMS ~0.0002），
#     接近零电平的信号过 WebRTC AEC/NS 后辅音被啃掉，ASR 全是错字
#     （"你好小狗"→"老师好"），唤醒和声纹一起崩。AGC 让 WebRTC 在
#     正常电平上工作，是治本。
#   但 config 的 mic_gain 必须同时 = 1.0！AGC 静音期会把增益拉高
#     ~160 倍，再叠加 mic_gain=50 就是"环境音全被识别"事故。
#     两级增益只能留一级。
#   noise_suppression=1 保留 WebRTC 降噪。
#   analog_gain_control=0（USB 阵列无模拟增益可调）。
pactl load-module module-echo-cancel \
    aec_method=webrtc \
    source_name=echocancel_source \
    sink_name=echocancel_sink \
    aec_args='"analog_gain_control=0 digital_gain_control=1 noise_suppression=1"'

pactl set-default-source echocancel_source
pactl set-default-sink echocancel_sink

echo "AEC 已启用："
pactl list short sources | grep echocancel || true
pactl list short sinks  | grep echocancel || true
echo
echo "请确认 config.yaml："
echo "  voice.audio.input_device: null     # 走默认源（带 AEC）"
echo "  voice.audio.output_device: null    # 走默认汇（作为 AEC 参考）"
echo "  voice.audio.mic_gain: 1.0          # 必须！增益已由 AGC 接管"
echo "  voice.barge_in.threshold: 0.5"
echo "  voice.timeouts.post_tts_cooldown_ms: 200"
