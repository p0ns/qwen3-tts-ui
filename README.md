# qwen3-tts-ui

A desktop UI for Qwen3-TTS text-to-speech models running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)

## Features

- **CustomVoice** — Pick from built-in speaker voices with emotion/style instruct prompts
- **VoiceDesign** — Describe the voice you want (pitch, tone, age, gender) and let the model create it
- **VoiceClone** — Record a reference audio clip and clone the voice
- Preset emotion buttons (Happy, Sad, Angry, Excited, Calm, Whisper)
- Output device selection
- Cmd+Enter to generate

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/p0ns/qwen3-tts-ui.git
cd qwen3-tts-ui
uv sync
```

Models are downloaded automatically from Hugging Face on first use.

## Usage

```bash
uv run python main.py
```

## Models

| Mode | Model |
|------|-------|
| CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` |
| VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` |
| VoiceClone | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |
