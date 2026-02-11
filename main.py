import os
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QKeySequence, QShortcut, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
)

from mlx_audio.tts.utils import load_model

MODELS = {
    "CustomVoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "VoiceDesign": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "VoiceClone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}

SAMPLES_DIR = Path(__file__).parent / "samples"

SAMPLE_RATE = 24000

STYLESHEET = """
QWidget#main {
    background-color: #1a1a2e;
}
QLabel {
    color: #a0a0c0;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QLabel#title {
    color: #e0e0ff;
    font-size: 18px;
    font-weight: 700;
}
QLabel#status {
    color: #6060a0;
    font-size: 12px;
    font-weight: 400;
}
QComboBox {
    background-color: #16213e;
    color: #e0e0ff;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}
QComboBox:hover {
    border-color: #4a4a8a;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0ff;
    border: 1px solid #2a2a4a;
    selection-background-color: #2a2a5a;
}
QLineEdit {
    background-color: #16213e;
    color: #e0e0ff;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: #4a4aaa;
    selection-color: #ffffff;
}
QLineEdit:focus {
    border-color: #5a5aaa;
}
QPlainTextEdit {
    background-color: #16213e;
    color: #e0e0ff;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
    selection-background-color: #4a4aaa;
    selection-color: #ffffff;
}
QPlainTextEdit:focus {
    border-color: #5a5aaa;
}
QPushButton#say {
    background-color: #4a4aaa;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 28px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton#say:hover {
    background-color: #5a5abb;
}
QPushButton#say:pressed {
    background-color: #3a3a99;
}
QPushButton#say:disabled {
    background-color: #2a2a4a;
    color: #5a5a7a;
}
QPushButton#record {
    background-color: #6e2a2a;
    color: #ff9090;
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#record:hover {
    background-color: #8a3a3a;
}
QPushButton#record:checked {
    background-color: #aa3a3a;
    color: #ffffff;
}
QFrame#separator {
    background-color: #2a2a4a;
    max-height: 1px;
}
QPushButton.preset {
    background-color: #22224a;
    color: #9090cc;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
}
QPushButton.preset:hover {
    background-color: #2a2a5a;
    border-color: #4a4a8a;
    color: #c0c0ee;
}
QPushButton.preset:checked {
    background-color: #3a3a7a;
    border-color: #5a5aaa;
    color: #e0e0ff;
}
"""

PRESETS = {
    "Happy": "Happy and cheerful.",
    "Sad": "Sad and melancholic.",
    "Angry": "Angry and intense.",
    "Excited": "Very excited and energetic.",
    "Calm": "Calm and soothing.",
    "Whisper": "Soft whispering voice.",
}


def get_output_devices():
    devices = sd.query_devices()
    out = []
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            out.append((i, d["name"]))
    return out


def list_samples():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(f.name for f in SAMPLES_DIR.glob("*.wav"))


class Signals(QObject):
    model_ready = Signal(str, list)
    status = Signal(str)
    done = Signal()
    recording_saved = Signal()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("main")
        self.setWindowTitle("Qwen TTS")
        self.setFixedWidth(460)
        self.setStyleSheet(STYLESHEET)

        self.signals = Signals()
        self.signals.model_ready.connect(self._on_model_ready)
        self.signals.status.connect(self._set_status)
        self.signals.done.connect(self._on_done)
        self.signals.recording_saved.connect(self._refresh_samples)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(24, 20, 24, 20)

        # Title
        title = QLabel("Qwen TTS")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addSpacing(4)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)
        layout.addSpacing(8)

        # Output device
        layout.addWidget(QLabel("OUTPUT DEVICE"))
        self.device_combo = QComboBox()
        self.devices = get_output_devices()
        default_dev = sd.default.device[1]
        default_idx = 0
        for i, (dev_id, name) in enumerate(self.devices):
            self.device_combo.addItem(name)
            if dev_id == default_dev:
                default_idx = i
        self.device_combo.setCurrentIndex(default_idx)
        layout.addWidget(self.device_combo)
        layout.addSpacing(4)

        # Mode
        layout.addWidget(QLabel("MODE"))
        self.mode_combo = QComboBox()
        for name in MODELS:
            self.mode_combo.addItem(name)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)
        layout.addSpacing(4)

        # Voice (CustomVoice only)
        self.voice_label = QLabel("VOICE")
        layout.addWidget(self.voice_label)
        self.voice_combo = QComboBox()
        self.voice_combo.setEnabled(False)
        layout.addWidget(self.voice_combo)
        layout.addSpacing(4)

        # --- Voice Clone section ---
        self.clone_label = QLabel("REFERENCE AUDIO")
        layout.addWidget(self.clone_label)

        clone_row = QHBoxLayout()
        clone_row.setSpacing(6)
        self.sample_combo = QComboBox()
        self.sample_combo.setSizePolicy(
            self.sample_combo.sizePolicy().horizontalPolicy(),
            self.sample_combo.sizePolicy().verticalPolicy(),
        )
        self._populate_samples()
        self.sample_combo.currentTextChanged.connect(self._on_sample_changed)
        clone_row.addWidget(self.sample_combo, 1)

        self.record_btn = QPushButton("Record")
        self.record_btn.setObjectName("record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_record_toggle)
        clone_row.addWidget(self.record_btn)
        self.clone_row_layout = clone_row
        layout.addLayout(clone_row)
        layout.addSpacing(4)

        self.ref_text_label = QLabel("REFERENCE TEXT")
        layout.addWidget(self.ref_text_label)
        self.ref_text_edit = QLineEdit()
        self.ref_text_edit.setPlaceholderText("Transcript of the reference audio")
        self.ref_text_edit.editingFinished.connect(self._save_ref_text)
        layout.addWidget(self.ref_text_edit)
        layout.addSpacing(4)

        # Instruct
        self.instruct_label = QLabel("INSTRUCT")
        layout.addWidget(self.instruct_label)

        # Preset buttons
        self.presets_row = QHBoxLayout()
        self.presets_row.setSpacing(6)
        self.preset_buttons = []
        for name, prompt in PRESETS.items():
            btn = QPushButton(name)
            btn.setProperty("class", "preset")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name, p=prompt: self._on_preset(n, p))
            self.presets_row.addWidget(btn)
            self.preset_buttons.append((name, btn))
        self.presets_row.addStretch()
        self.presets_widget = QWidget()
        self.presets_widget.setLayout(self.presets_row)
        layout.addWidget(self.presets_widget)

        self.instruct_edit = QPlainTextEdit()
        self.instruct_edit.setFixedHeight(44)
        self.instruct_edit.setPlaceholderText("e.g. Very happy and excited.")
        self.instruct_edit.textChanged.connect(self._on_instruct_edited)
        layout.addWidget(self.instruct_edit)
        layout.addSpacing(4)

        # Text
        layout.addWidget(QLabel("TEXT"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(100)
        layout.addWidget(self.text_edit)
        layout.addSpacing(8)

        # Bottom row: status + say button
        bottom = QHBoxLayout()
        self.status_label = QLabel("Loading model...")
        self.status_label.setObjectName("status")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        self.say_btn = QPushButton("Say")
        self.say_btn.setObjectName("say")
        self.say_btn.setEnabled(False)
        self.say_btn.clicked.connect(self._on_say)
        bottom.addWidget(self.say_btn)
        layout.addLayout(bottom)

        # Cmd+Enter shortcut
        shortcut = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Return), self)
        shortcut.activated.connect(self._on_say)

        # Recording state
        self.recording = False
        self.recorded_frames = []

        self.models = {}
        self.current_mode = "CustomVoice"
        self._update_ui_for_mode()
        threading.Thread(target=self._load_model, args=("CustomVoice",), daemon=True).start()

    def _populate_samples(self):
        self.sample_combo.clear()
        samples = list_samples()
        if samples:
            for s in samples:
                self.sample_combo.addItem(s)
        else:
            self.sample_combo.addItem("(no samples — record one)")

    def _refresh_samples(self):
        prev = self.sample_combo.currentText()
        self._populate_samples()
        # select the most recently added (last item)
        self.sample_combo.setCurrentIndex(self.sample_combo.count() - 1)

    def _save_ref_text(self):
        sample_name = self.sample_combo.currentText()
        txt_path = SAMPLES_DIR / Path(sample_name).with_suffix(".txt")
        ref_text = self.ref_text_edit.text().strip()
        if ref_text and SAMPLES_DIR.exists():
            txt_path.write_text(ref_text)

    def _on_sample_changed(self, name):
        txt_path = SAMPLES_DIR / Path(name).with_suffix(".txt")
        if txt_path.exists():
            self.ref_text_edit.setText(txt_path.read_text().strip())
        else:
            self.ref_text_edit.clear()

    def _on_record_toggle(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording = True
        self.recorded_frames = []
        self.record_btn.setText("Stop")
        self.status_label.setText("Recording...")

        def callback(indata, frames, time, status):
            self.recorded_frames.append(indata.copy())

        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self.input_stream.start()

    def _stop_recording(self):
        self.recording = False
        self.record_btn.setText("Record")
        self.record_btn.setChecked(False)
        self.input_stream.stop()
        self.input_stream.close()

        if not self.recorded_frames:
            self.status_label.setText("Ready")
            return

        audio = np.concatenate(self.recorded_frames).flatten()

        SAMPLES_DIR.mkdir(exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        filepath = SAMPLES_DIR / filename

        # Write WAV using stdlib
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(str(filepath), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        # Save reference text alongside the wav
        ref_text = self.ref_text_edit.text().strip()
        if ref_text:
            filepath.with_suffix(".txt").write_text(ref_text)

        self.status_label.setText(f"Saved {filename}")
        self.signals.recording_saved.emit()

    def _load_model(self, mode):
        model = load_model(MODELS[mode])
        self.models[mode] = model
        voices = model.get_supported_speakers() if mode == "CustomVoice" else []
        self.signals.model_ready.emit(mode, voices)

    def _on_model_ready(self, mode, voices):
        if mode == "CustomVoice":
            self.voice_combo.clear()
            for v in voices:
                self.voice_combo.addItem(v)
            self.voice_combo.setEnabled(True)
        self._update_ui_for_mode()
        self.say_btn.setEnabled(True)
        self.status_label.setText("Ready")

    def _on_mode_changed(self, mode):
        self.current_mode = mode
        if mode in self.models:
            self._update_ui_for_mode()
            return
        self.say_btn.setEnabled(False)
        self.status_label.setText(f"Loading {mode} model...")
        threading.Thread(target=self._load_model, args=(mode,), daemon=True).start()

    def _update_ui_for_mode(self):
        mode = self.current_mode
        is_custom = mode == "CustomVoice"
        is_clone = mode == "VoiceClone"

        # Voice dropdown: CustomVoice only
        self.voice_label.setVisible(is_custom)
        self.voice_combo.setVisible(is_custom)

        # Clone section: VoiceClone only
        self.clone_label.setVisible(is_clone)
        self.sample_combo.setVisible(is_clone)
        self.record_btn.setVisible(is_clone)
        self.ref_text_label.setVisible(is_clone)
        self.ref_text_edit.setVisible(is_clone)

        # Instruct: CustomVoice and VoiceDesign only
        show_instruct = not is_clone
        self.instruct_label.setVisible(show_instruct)
        self.instruct_edit.setVisible(show_instruct)
        self.presets_widget.setVisible(show_instruct)

        if is_custom:
            self.instruct_edit.setPlaceholderText("e.g. Very happy and excited.")
        elif mode == "VoiceDesign":
            self.instruct_edit.setPlaceholderText(
                "e.g. A cheerful young female voice with high pitch and energetic tone."
            )

    def _on_preset(self, name, prompt):
        for btn_name, btn in self.preset_buttons:
            if btn_name != name:
                btn.setChecked(False)
        active = next(btn for n, btn in self.preset_buttons if n == name)
        if active.isChecked():
            self.instruct_edit.setPlainText(prompt)
        else:
            self.instruct_edit.clear()

    def _on_instruct_edited(self):
        text = self.instruct_edit.toPlainText()
        for _, btn in self.preset_buttons:
            if btn.isChecked():
                matching = any(text == prompt for prompt in PRESETS.values())
                if not matching:
                    btn.setChecked(False)

    def _set_status(self, text):
        self.status_label.setText(text)

    def _on_say(self):
        text = self.text_edit.toPlainText().strip()
        if not text or not self.say_btn.isEnabled():
            return

        mode = self.current_mode
        voice = self.voice_combo.currentText()
        instruct = self.instruct_edit.toPlainText().strip() or None
        dev_idx = self.devices[self.device_combo.currentIndex()][0]

        ref_audio = None
        ref_text = None
        if mode == "VoiceClone":
            sample_name = self.sample_combo.currentText()
            sample_path = SAMPLES_DIR / sample_name
            if not sample_path.exists():
                self.status_label.setText("Record a sample first")
                return
            ref_audio = str(sample_path)
            ref_text = self.ref_text_edit.text().strip()
            if not ref_text:
                self.status_label.setText("Enter reference text")
                return

        self.say_btn.setEnabled(False)
        self.text_edit.selectAll()
        self.text_edit.setFocus()
        self.status_label.setText("Generating...")
        threading.Thread(
            target=self._generate,
            args=(mode, text, voice, instruct, ref_audio, ref_text, dev_idx),
            daemon=True,
        ).start()

    def _generate(self, mode, text, voice, instruct, ref_audio, ref_text, device_idx):
        try:
            model = self.models[mode]
            chunks = []
            if mode == "CustomVoice":
                results = model.generate_custom_voice(
                    text=text,
                    speaker=voice,
                    language="auto",
                    instruct=instruct,
                )
            elif mode == "VoiceDesign":
                results = model.generate_voice_design(
                    text=text,
                    language="auto",
                    instruct=instruct or "",
                )
            else:
                results = model.generate(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            for result in results:
                chunks.append(np.array(result.audio, dtype=np.float32))
            audio = np.concatenate(chunks)
            sr = result.sample_rate

            self.signals.status.emit("Playing...")
            sd.play(audio, samplerate=sr, device=device_idx)
            sd.wait()
        except Exception as e:
            self.signals.status.emit(f"Error: {e}")
            self.signals.done.emit()
            return
        self.signals.done.emit()

    def _on_done(self):
        self.say_btn.setEnabled(True)
        self.status_label.setText("Ready")


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
