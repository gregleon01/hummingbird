#!/usr/bin/env python3
"""
Frameless macOS overlay that toggles with Control+Option+M, visualises microphone input,
records speech, and transcribes with Whisper. The transcript stays hidden until
the user presses the copy button.
"""
import argparse
import math
import os
import queue
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QPointF, QRectF
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QPainterPath,
    QRadialGradient,
    QRegion,
    QBrush,
)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QGraphicsDropShadowEffect,
    QComboBox,
    QPlainTextEdit,
)

try:
    import Quartz  # noqa: F401
    import Quartz.CoreGraphics as CG
    import CoreFoundation as CF
except ImportError as exc:  # pragma: no cover - pyobjc missing
    raise SystemExit(
        "PyObjC is required for the global hotkey. Install with `pip install pyobjc`."
    ) from exc


OPTION_MASK = CG.kCGEventFlagMaskAlternate
CONTROL_MASK = CG.kCGEventFlagMaskControl
KEYCODE_M = 46  # macOS virtual keycode for the 'M' key


class SummaryGenerator:
    """Lazy text summariser powered by a small transformers model."""

    PROMPTS: Dict[str, str] = {
        "coding": (
            "You rewrite transcripts into concise programming direction. "
            "Return bullet points with requirements, APIs, assumptions, and next actions. "
            "Transcript:\n{transcript}\n\nCoding summary:"
        ),
        "personal": (
            "You are an empathetic editor. Clean up grammar and punctuation while keeping the voice first-person. "
            "Transcript:\n{transcript}\n\nPolished version:"
        ),
        "meetings": (
            "You are a chief of staff capturing a meeting. Use headings for Key Decisions, Action Items, Deadlines, "
            "Notable Quotes, and Other Notes. If a section is empty, write 'None'. Prioritise urgent items first. "
            "Transcript:\n{transcript}\n\nStructured meeting report:"
        ),
    }

    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("HUMMINGBIRD_SUMMARY_MODEL", "google/flan-t5-small")
        self.model_name = model_name
        self._pipeline = None
        self._lock = threading.Lock()
        self._using_fallback = False

    def _load_pipeline(self):
        with self._lock:
            if self._pipeline is None:
                try:
                    from transformers import pipeline
                except ModuleNotFoundError:
                    self._pipeline = False
                    self._using_fallback = True
                    return None
                self._pipeline = pipeline("text2text-generation", model=self.model_name)
        if self._pipeline is False:
            return None
        return self._pipeline

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def generate(self, transcript: str, focus: str) -> str:
        if not transcript.strip():
            return ""
        prompt = self.PROMPTS.get(focus)
        if prompt is None:
            raise ValueError(f"Unknown summary focus: {focus}")
        generator = self._load_pipeline()
        if generator is None:
            self._using_fallback = True
            return self._fallback_summary(transcript, focus)
        self._using_fallback = False
        result = generator(
            prompt.format(transcript=transcript.strip()),
            max_length=512,
            num_beams=4,
        )
        if not result:
            return ""
        return result[0]["generated_text"].strip()

    def _fallback_summary(self, transcript: str, focus: str) -> str:
        sentences = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+", transcript.strip())
            if chunk.strip()
        ]
        if focus == "coding":
            bullets = []
            for sentence in sentences:
                cleaned = " ".join(sentence.split())
                if not cleaned:
                    continue
                if len(cleaned) > 140:
                    cleaned = cleaned[:137].rstrip() + "…"
                bullets.append(f"• {cleaned}")
                if len(bullets) >= 6:
                    break
            if not bullets:
                bullets = ["• Clarify the goal and restate the main steps."]
            return "\n".join(bullets)

        if focus == "personal":
            text = " ".join(transcript.split())
            if not text.endswith(('.', '!', '?')):
                text += "."
            return text

        if focus == "meetings":
            sections = {
                "Key Decisions": [],
                "Action Items": [],
                "Deadlines": [],
                "Notable Quotes": [],
                "Other Notes": [],
            }
            for sentence in sentences:
                lower = sentence.lower()
                target = "Other Notes"
                if any(keyword in lower for keyword in ("decide", "agreement", "approved")):
                    target = "Key Decisions"
                elif any(keyword in lower for keyword in ("todo", "action", "follow up", "follow-up")):
                    target = "Action Items"
                elif any(keyword in lower for keyword in ("due", "deadline", "by ")):
                    target = "Deadlines"
                elif any(quote in lower for quote in ("said", "stated", "noted")):
                    target = "Notable Quotes"
                sections[target].append(sentence.strip())
            lines = []
            for heading, items in sections.items():
                lines.append(f"{heading}:")
                if items:
                    for item in items[:5]:
                        lines.append(f"- {item}")
                else:
                    lines.append("- None")
                lines.append("")
            return "\n".join(lines).strip()

        return transcript.strip()


def is_accessibility_granted() -> bool:
    options = {"AXTrustedCheckOptionPrompt": True}
    if hasattr(CG, "AXIsProcessTrustedWithOptions"):
        return CG.AXIsProcessTrustedWithOptions(options)
    if hasattr(CG, "AXIsProcessTrusted"):
        return CG.AXIsProcessTrusted()
    return True


class GlobalHotkey:
    """Registers an Option+M event tap using Quartz."""

    def __init__(self, callback):
        if not is_accessibility_granted():
            print(
                "Accessibility access missing: enable Terminal/iTerm in System Settings ▸ Privacy & Security ▸ Accessibility, then relaunch.",
                file=sys.stderr,
            )
        self.callback = callback
        self._tap = None
        self._loop = None
        self._source = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        event_mask = CG.CGEventMaskBit(CG.kCGEventKeyDown)
        self._tap = CG.CGEventTapCreate(
            CG.kCGSessionEventTap,
            CG.kCGHeadInsertEventTap,
            CG.kCGEventTapOptionDefault,
            event_mask,
            self._event_callback,
            None,
        )
        if not self._tap:
            print(
                "Failed to create keyboard event tap. Check Accessibility permissions.",
                file=sys.stderr,
            )
            return
        else:
            print("Global hotkey Control+Option+M ready.")

        self._source = CF.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._loop = CF.CFRunLoopGetCurrent()
        CF.CFRunLoopAddSource(self._loop, self._source, CF.kCFRunLoopCommonModes)
        CG.CGEventTapEnable(self._tap, True)
        CF.CFRunLoopRun()

    def _event_callback(self, proxy, event_type, event, _refcon):
        if event_type == CG.kCGEventKeyDown:
            keycode = CG.CGEventGetIntegerValueField(event, CG.kCGKeyboardEventKeycode)
            flags = CG.CGEventGetFlags(event)
            use_combo = (flags & OPTION_MASK) and (flags & CONTROL_MASK)
            if keycode == KEYCODE_M and use_combo:
                self.callback()
                return event
        return event

    def stop(self):
        if self._loop:
            CF.CFRunLoopStop(self._loop)
        if self._tap:
            CF.CFMachPortInvalidate(self._tap)
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)


def mix_colors(cold: QColor, hot: QColor, amount: float) -> QColor:
    """Return an interpolated color between cold and hot."""
    amount = max(0.0, min(1.0, amount))
    r = int(cold.red() + (hot.red() - cold.red()) * amount)
    g = int(cold.green() + (hot.green() - cold.green()) * amount)
    b = int(cold.blue() + (hot.blue() - cold.blue()) * amount)
    return QColor(r, g, b)


def resolve_font_family(
    primary: str = "SF Pro Rounded",
    fallbacks=("SF Pro Display", "Avenir Next", "Inter", "Helvetica Neue", "Rubik", "Menlo", "SFMono-Regular"),
):
    families = set(QFontDatabase.families())
    if primary in families:
        return primary
    for name in fallbacks:
        if name in families:
            return name
    system = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    return system.family()


class GlowCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self._recording = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(QSize(320, 320))

    @property
    def level(self):
        return self._level

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.update()

    def set_recording(self, recording: bool):
        self._recording = recording
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        rect = self.rect().adjusted(18, 18, -18, -18)
        center = rect.center()
        base_radius = min(rect.width(), rect.height()) * 0.3

        cold = QColor(31, 64, 104)
        hot = QColor(255, 99, 71)
        accent = mix_colors(cold, hot, self._level ** 0.82)

        glow_radius = min(rect.width(), rect.height()) * (0.6 + 0.28 * self._level)
        gradient = QRadialGradient(center, glow_radius)
        gradient.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 205))
        gradient.setColorAt(0.16, QColor(accent.red(), accent.green(), accent.blue(), 140))
        gradient.setColorAt(0.38, QColor(accent.red(), accent.green(), accent.blue(), 80))
        gradient.setColorAt(0.7, QColor(32, 40, 66, 28))
        gradient.setColorAt(0.92, QColor(12, 16, 26, 10))
        gradient.setColorAt(1.0, QColor(12, 16, 26, 0))

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        glow_rect = QRectF(
            center.x() - glow_radius,
            center.y() - glow_radius,
            glow_radius * 2,
            glow_radius * 2,
        )
        painter.drawEllipse(glow_rect)
        painter.restore()

        core_color = accent if self._recording else QColor(22, 28, 43)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(core_color)
        painter.drawEllipse(QPointF(center), base_radius, base_radius)


class VoiceOverlay(QWidget):
    def __init__(self, model_name: str, samplerate: int):
        super().__init__()
        self.model_name = model_name
        self.samplerate = samplerate
        self.model = None
        self.stream = None
        self._audio_writer = None
        self._recording_path = None
        self.level_queue = queue.Queue(maxsize=5)
        self.recording = False
        self.transcribing = False
        self.last_transcript = ""
        self.summary_generator: Optional[SummaryGenerator] = None
        self.summary_results: Dict[str, str] = {}
        self.summarizing_focus: Optional[str] = None
        self.summarizing = False
        self._drag_offset = QPointF(0, 0)
        self._drag_active = False

        self._font_family: str = resolve_font_family(
            primary="SF Pro Rounded",
            fallbacks=(
                "SF Pro Display",
                "Avenir Next",
                "Inter",
                "Helvetica Neue",
                "Rubik",
                "Menlo",
                "SFMono-Regular",
            ),
        )

        self._build_ui()
        self._configure_window()
        self._setup_timers()
        self._install_hotkey()

    def _configure_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(
            f"""
            QWidget#container {{
                background: rgba(12, 16, 26, 0.88);
                border-radius: 28px;
                border: 1px solid rgba(94, 104, 128, 0.22);
            }}
            QLabel {{
                color: rgba(224, 231, 255, 0.9);
                font-family: "{self._font_family}";
                letter-spacing: 0.2px;
            }}
            QComboBox {{
                background: rgba(17, 24, 39, 0.94);
                color: #f8fafc;
                border: 1px solid rgba(94, 104, 128, 0.32);
                border-radius: 16px;
                padding: 9px 18px;
                font-family: "{self._font_family}";
                font-size: 12px;
            }}
            QComboBox QAbstractItemView {{
                background: rgba(15, 23, 42, 0.98);
                border-radius: 14px;
                selection-background-color: rgba(59, 130, 246, 0.28);
                selection-color: #f8fafc;
                font-family: "{self._font_family}";
            }}
            QPlainTextEdit {{
                background: rgba(8, 11, 18, 0.8);
                border: 1px solid rgba(94, 104, 128, 0.22);
                border-radius: 22px;
                color: rgba(226, 232, 240, 0.94);
                padding: 18px;
                font-family: "{self._font_family}";
                font-size: 12px;
                line-height: 1.4em;
            }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(30, 44, 71, 0.98),
                                            stop:1 rgba(18, 27, 45, 0.98));
                color: #f8fafc;
                border-radius: 20px;
                padding: 13px 38px;
                font-family: "{self._font_family}";
                font-weight: 600;
                letter-spacing: 0.8px;
            }}
            QPushButton:hover {{
                background: rgba(43, 58, 90, 0.98);
            }}
            QPushButton:pressed {{
                background: rgba(20, 26, 42, 1.0);
            }}
            QPushButton:disabled {{
                background: rgba(30, 41, 59, 0.6);
                color: rgba(248, 250, 252, 0.45);
            }}
        """
        )
        self._update_mask()

    def _update_mask(self):
        if self.rect().isNull():
            return
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 28, 28)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget(objectName="container")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(36, 36, 36, 30)
        container_layout.setSpacing(26)

        self.container = container
        container.installEventFilter(self)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 32)
        shadow.setColor(QColor(8, 11, 18, 160))
        container.setGraphicsEffect(shadow)

        self.glow = GlowCanvas()
        self.glow.setCursor(Qt.CursorShape.PointingHandCursor)
        self.glow.mousePressEvent = self._handle_click  # type: ignore[method-assign]

        font_family = self._font_family

        self.status = QLabel("Control+Option+M to summon • Tap the orb to capture a thought")
        self.status.setWordWrap(True)
        status_font = QFont(font_family, 10)
        self.status.setFont(status_font)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)

        self.summary_selector = QComboBox()
        self.summary_selector.addItem("Select summary focus…", userData=None)
        self.summary_selector.addItem("Coding", userData="coding")
        self.summary_selector.addItem("Personal", userData="personal")
        self.summary_selector.addItem("Meetings", userData="meetings")
        self.summary_selector.setEnabled(False)
        self.summary_selector.currentIndexChanged.connect(self._update_summary_display)  # type: ignore[arg-type]

        self.summary_output = QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlainText("Summaries appear here once generated.")

        self.copy_button = QPushButton("Copy Transcript")
        self.copy_button.setEnabled(False)
        copy_font = QFont(font_family, 12, QFont.Weight.Medium)
        self.copy_button.setFont(copy_font)
        self.copy_button.clicked.connect(self._copy_transcript)  # type: ignore[arg-type]

        self.summary_copy_button = QPushButton("Copy Summary")
        self.summary_copy_button.setEnabled(False)
        summary_copy_font = QFont(font_family, 12, QFont.Weight.Medium)
        self.summary_copy_button.setFont(summary_copy_font)
        self.summary_copy_button.clicked.connect(self._copy_summary)  # type: ignore[arg-type]

        control_row.addStretch()
        control_row.addWidget(self.copy_button)
        control_row.addWidget(self.summary_copy_button)
        control_row.addStretch()

        container_layout.addWidget(self.glow, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.status)
        container_layout.addWidget(self.summary_selector)
        container_layout.addWidget(self.summary_output)
        container_layout.addLayout(control_row)

        root_layout.addWidget(container)

    def _setup_timers(self):
        self.visual_timer = QTimer(self)
        self.visual_timer.setInterval(40)  # ~25 FPS
        self.visual_timer.timeout.connect(self._update_level_from_queue)  # type: ignore[arg-type]
        self.visual_timer.start()

    def _install_hotkey(self):
        def toggle():
            QApplication.instance().postEvent(self, _ToggleEvent(self._toggle_visibility))

        self.hotkey = GlobalHotkey(callback=toggle)

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self._center_on_primary()
            self.show()
            self.raise_()
            self.activateWindow()

    def _center_on_primary(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        screen_geometry = screen.availableGeometry()
        x = screen_geometry.center().x() - (self.width() // 2)
        y = screen_geometry.center().y() - (self.height() // 2)
        self.move(int(x), int(y))

    def _handle_click(self, _event):
        if self.transcribing:
            return
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        while not self.level_queue.empty():
            try:
                self.level_queue.get_nowait()
            except queue.Empty:
                break

        try:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self._recording_path = Path(path)
            self._audio_writer = sf.SoundFile(
                self._recording_path,
                mode="w",
                samplerate=self.samplerate,
                channels=1,
            )
        except Exception as exc:
            self._set_status(f"File error: {exc}")
            self._cleanup_recording_file(remove=True)
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self._set_status(f"Mic error: {exc}")
            self._cleanup_recording_file(remove=True)
            return

        self.recording = True
        self.copy_button.setEnabled(False)
        self.summary_copy_button.setEnabled(False)
        self.summary_selector.setCurrentIndex(0)
        self.summary_selector.setEnabled(False)
        self.summary_results.clear()
        self.summarizing = False
        self.summarizing_focus = None
        self.summary_output.setPlainText("Summaries appear here once generated.")
        self.glow.set_recording(True)
        self._set_status("Listening… tap again to finish")

    def _stop_recording(self):
        if self.stream is None:
            return
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None

        recording_path = self._recording_path
        self._recording_path = None
        self._close_writer()

        self.glow.set_recording(False)
        self.glow.set_level(0.0)
        self.transcribing = True
        self._set_status("Transcribing…")

        threading.Thread(
            target=self._transcribe_file,
            args=(recording_path,),
            daemon=True,
        ).start()

    def _audio_callback(self, indata, _frames, _time, _status):
        if not self.recording:
            return
        if self._audio_writer is not None:
            try:
                self._audio_writer.write(indata)
            except Exception as exc:
                print(f"File write error: {exc}", file=sys.stderr)
        rms = float(np.sqrt(np.mean(np.square(indata))))
        level = min(1.0, rms * 12.0)
        try:
            self.level_queue.put_nowait(level)
        except queue.Full:
            pass

    def _update_level_from_queue(self):
        if not self.recording:
            self.glow.set_level(0.0)
            return
        level = 0.0
        while not self.level_queue.empty():
            level = self.level_queue.get()
        smoothed = 0.7 * self.glow.level + 0.3 * level
        self.glow.set_level(smoothed)

    def _transcribe_file(self, path):
        try:
            wav_path = Path(path) if path is not None else None
            if wav_path is None or not wav_path.exists() or wav_path.stat().st_size == 0:
                self._finish_transcription("", "No audio detected")
                return

            try:
                if self.model is None:
                    self._set_status("Loading Whisper model…")
                    self.model = whisper.load_model(self.model_name)
                result = self.model.transcribe(str(wav_path), language="en")
                transcript = result.get("text", "").strip()
                if not transcript:
                    self._finish_transcription("", "No speech recognised")
                else:
                    self._finish_transcription(transcript, "Transcript ready — press Copy")
            except Exception as exc:
                self._finish_transcription("", f"Whisper error: {exc}")
        finally:
            self._cleanup_recording_file(path=path, remove=True)

    def _close_writer(self):
        if self._audio_writer is not None:
            try:
                self._audio_writer.flush()
            except Exception:
                pass
            try:
                self._audio_writer.close()
            except Exception:
                pass
            self._audio_writer = None

    def _cleanup_recording_file(self, path=None, remove=False):
        if path is None:
            path = self._recording_path
        self._close_writer()
        if remove and path is not None:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"Failed to delete temp file {path}: {exc}", file=sys.stderr)
        if path is self._recording_path:
            self._recording_path = None

    def _finish_transcription(self, text: str, status: str):
        def update():
            self.transcribing = False
            self.last_transcript = text
            self.summarizing = False
            self.summarizing_focus = None
            self.summary_results.clear()
            self.copy_button.setEnabled(bool(text))
            self.summary_selector.setEnabled(bool(text))
            if bool(text):
                self.summary_selector.setCurrentIndex(0)
                self.summary_output.setPlainText("Pick a focus from the menu to generate a summary.")
                self.summary_copy_button.setEnabled(False)
            if not text:
                self.summary_output.setPlainText("Summaries appear here once generated.")
                self.summary_copy_button.setEnabled(False)
            self._set_status(status)
        QApplication.instance().postEvent(self, _ToggleEvent(update))

    def _copy_transcript(self):
        if not self.last_transcript:
            return
        QApplication.clipboard().setText(self.last_transcript)
        self.copy_button.setEnabled(False)
        self._set_status("Copied to clipboard")

    def _copy_summary(self):
        focus = self.summary_selector.currentData()
        if not focus:
            return
        summary = self.summary_results.get(focus)
        if not summary:
            return
        QApplication.clipboard().setText(summary)
        self.summary_copy_button.setEnabled(False)
        self._set_status("Summary copied to clipboard")

    def _start_summary_generation(self, focus: str):
        if not self.last_transcript:
            return
        if self.summarizing and self.summarizing_focus == focus:
            return
        self.summarizing = True
        self.summarizing_focus = focus
        self.summary_copy_button.setEnabled(False)
        self._set_status(f"Generating {focus} summary…")

        def worker():
            if self.summary_generator is None:
                self.summary_generator = SummaryGenerator()
            try:
                summary_text = self.summary_generator.generate(self.last_transcript, focus)
                if self.summary_generator.using_fallback:
                    message = (
                        f"{focus.title()} summary ready (basic mode — install 'transformers' for advanced results)"
                    )
                else:
                    message = f"{focus.title()} summary ready"
            except Exception as exc:
                summary_text = ""
                message = f"Summary error: {exc}"

            def apply():
                self.summarizing = False
                if summary_text:
                    self.summary_results[focus] = summary_text
                    if self.summary_selector.currentData() == focus:
                        self.summary_output.setPlainText(summary_text)
                        self.summary_copy_button.setEnabled(True)
                else:
                    if self.summary_selector.currentData() == focus:
                        self.summary_output.setPlainText("Failed to generate summary.")
                        self.summary_copy_button.setEnabled(False)
                self._set_status(message)

            QApplication.instance().postEvent(self, _ToggleEvent(apply))

        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, message: str):
        self.status.setText(message)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is getattr(self, "container", None):
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton and not self._hit_interactive_child(event.position().toPoint()):
                    self._drag_active = True
                    self._drag_offset = event.globalPosition() - QPointF(self.x(), self.y())
                    event.accept()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if self._drag_active and (event.buttons() & Qt.MouseButton.LeftButton):
                    target = event.globalPosition() - self._drag_offset
                    self.move(int(target.x()), int(target.y()))
                    event.accept()
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._drag_active:
                    self._drag_active = False
                    event.accept()
                    return True
        return QWidget.eventFilter(self, watched, event)

    def _hit_interactive_child(self, pos):
        child = self.container.childAt(pos)
        return child in {
            self.glow,
            self.copy_button,
            self.summary_copy_button,
            self.summary_selector,
            self.summary_output,
            self.summary_output.viewport(),
        }

    def closeEvent(self, event):  # noqa: N802
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except sd.PortAudioError:
                pass
        self._cleanup_recording_file(remove=True)
        if hasattr(self, "hotkey"):
            self.hotkey.stop()
        super().closeEvent(event)

    def event(self, event):  # noqa: N802
        if isinstance(event, _ToggleEvent):
            event.invoke()
            return True
        return super().event(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_mask()

    def _update_summary_display(self, _index: int = 0):
        focus = self.summary_selector.currentData()
        if not focus:
            self.summary_output.setPlainText("Summaries appear here once generated.")
            self.summary_copy_button.setEnabled(False)
            return
        if not self.last_transcript:
            self.summary_output.setPlainText("Record something to generate a summary.")
            self.summary_copy_button.setEnabled(False)
            return
        summary = self.summary_results.get(focus)
        if summary:
            self.summary_output.setPlainText(summary)
            self.summary_copy_button.setEnabled(True)
            self._set_status(f"{focus.title()} summary ready")
        else:
            self.summary_output.setPlainText("Generating summary…")
            self._start_summary_generation(focus)


class _ToggleEvent(QEvent):
    """Custom event to marshal callbacks onto the Qt thread."""

    EVENT_TYPE = QEvent.registerEventType()

    def __init__(self, callback):
        super().__init__(self.EVENT_TYPE)
        self.callback = callback

    def invoke(self):
        self.callback()


def main():
    parser = argparse.ArgumentParser(description="Mac voice overlay using Whisper.")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base).")
    parser.add_argument("--samplerate", type=int, default=16000, help="Recording sample rate.")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    overlay = VoiceOverlay(model_name=args.model, samplerate=args.samplerate)
    overlay.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
