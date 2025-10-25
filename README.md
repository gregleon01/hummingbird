# Hummingbird

Whisper-based macOS overlay for voice-driven notes with on-device summarisation.

Hummingbird is a minimalist macOS overlay for capturing short voice notes and transcribing
them with Whisper. A frameless PyQt6 window toggles on a global `Control+Option+M` shortcut,
animates a live volume bloom, records through PortAudio, then pushes the transcript straight
to your clipboard. You can also transform the transcript into three tailored summaries —
**Coding**, **Personal**, and **Meetings** — using a small, local `transformers` model so no
cloud access or paid APIs are required.

## Features
- Always-on-top floating orb with smooth blue→coral glow tied to microphone input.
- One-click record/stop with automatic Whisper transcription (any Whisper model).
- Global `Control+Option+M` hotkey registered through Quartz for instant show/hide.
- Dedicated **Copy Transcript** and **Copy Summary** actions for fast clipboard access.
- Sleek focus selector that generates Coding, Personal, or Meetings summaries on demand.
- Drag the window anywhere; Esc or the hotkey hides it again.

## Requirements
- macOS Monterey (12) or later.
- Python 3.12 (Homebrew’s `python@3.12` recommended).
- An active virtual environment with the dependencies below.
- FFmpeg (`brew install ffmpeg`) for the underlying audio stack.
- Microphone access and Accessibility permission for your terminal (System Settings ▸ Privacy & Security).
- The first time you generate a summary the app will download a compact text-to-text
  model (default `google/flan-t5-small`).

## Installation
```bash
# Clone the repository
 git clone git@github.com:gregleon01/hummingbird.git
 cd hummingbird

# Create a virtual environment
 /opt/homebrew/bin/python3.12 -m venv venv
 source venv/bin/activate
 pip install --upgrade pip

# Install runtime dependencies
 pip install -r requirements.txt

# (Optional) Install pyinstaller if you plan to bundle an app
 pip install pyinstaller
```

## Usage
```bash
# Activate your environment and launch the overlay
source venv/bin/activate
./scripts/hummingbird

# …or run the entry script directly
python hummingbird.py --model base
```

The first run will download the chosen Whisper model (default `base`). When the orb is visible:

1. Click the orb to begin recording. The glow deepens as the input level increases.
2. Click again to stop and begin transcription.
3. Choose a focus (Coding, Personal, Meetings) from the dropdown to generate an AI summary.
4. Press **Copy Transcript** or **Copy Summary** to send the selected text to your clipboard.
5. Use `Control+Option+M` at any time to hide or reshow the overlay.

## Configuration
- `--model` — change the Whisper model (`tiny`, `base`, `small`, `medium`, etc.).
- `--samplerate` — override the default 16 kHz recording sample rate.
- Set the environment variable `VENV` when calling `scripts/hummingbird` to point to a specific venv.
- Set the environment variable `HUMMINGBIRD_SUMMARY_MODEL` to override the default
  summariser (any Hugging Face text-to-text model compatible with `transformers`).

### Suggested free summary models
- `google/flan-t5-small` *(default)* — fast on Apple Silicon and good for short notes.
- `google/flan-t5-base` — higher quality rewriting if you have extra RAM (~1.3 GB).
- `philschmid/flan-t5-base-samsum` — tuned for meeting notes; produces rich agendas while
  remaining completely free to use.

## Development
- The main entry point lives in `hummingbird.py`.
- UI tweaks sit inside `VoiceOverlay._configure_window`, `GlowCanvas.paintEvent`, and the `resolve_font_family` helper.
- Run `python -m py_compile hummingbird.py` after edits to ensure everything still compiles.

## TODO
- Optional: fade animations when toggling visibility.
- Optional: live partial transcription while recording.
- Optional: packaged `.app` via PyInstaller for Dock pinning.

## License
MIT License (see `LICENSE`).
