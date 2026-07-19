"""Voice layer — STT (faster-whisper) + TTS (edge-tts) + energy VAD."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np

from . import config

_sd = None
try:
    import sounddevice as _sd
except Exception:
    pass


class Speaker:
    """Text-to-speech bang edge-tts, phat qua sounddevice."""

    def __init__(self, voice: str | None = None, rate: str | None = None):
        self.voice = voice or config.TTS_VOICE
        self.rate = rate or config.TTS_RATE

    def say(self, text: str):
        if not text.strip():
            return
        if _sd is None:
            print(f"[voice] (no sounddevice) {text}")
            return
        try:
            import edge_tts
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            asyncio.run(communicate.save(tmp_path))

            data, sr = sf.read(tmp_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            _sd.play(data, samplerate=sr)
            _sd.wait()

            Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"[voice] TTS loi: {e}")


class Listener:
    """Microphone -> energy VAD -> STT."""

    def __init__(self):
        self._model = None
        self._sample_rate = config.SAMPLE_RATE
        self._frame_duration = 0.03
        self._frame_size = int(self._sample_rate * self._frame_duration)
        self._energy_threshold = 0.01

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print("[voice] Dang tai Whisper model...")
            self._model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
            print(f"[voice] Whisper '{config.WHISPER_MODEL}' san sang.")

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def _record_until_silence(self) -> np.ndarray | None:
        if _sd is None:
            return None
        frames: list[np.ndarray] = []
        silent_chunks = 0
        max_silent = int(config.SILENCE_THRESHOLD_MS / (self._frame_duration * 1000))
        speaking = False

        with _sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16",
            blocksize=self._frame_size,
        ) as stream:
            while True:
                data, _ = stream.read(self._frame_size)
                frame = data[:, 0]
                is_speech = self._rms(frame) > self._energy_threshold

                if is_speech:
                    speaking = True
                    silent_chunks = 0
                    frames.append(frame.copy())
                elif speaking:
                    silent_chunks += 1
                    frames.append(frame.copy())
                    if silent_chunks >= max_silent:
                        break

        if not frames:
            return None
        return np.concatenate(frames).astype(np.float32) / 32768.0

    def listen(self, prompt: str = "") -> str:
        if _sd is None:
            if prompt:
                print(f"[voice] {prompt}")
            return input("text only> ").strip()

        self._ensure_model()
        if prompt:
            print(f"[voice] {prompt}")

        audio = self._record_until_silence()
        if audio is None:
            return ""

        segments, _ = self._model.transcribe(
            audio, language=config.WHISPER_LANGUAGE, beam_size=3,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            print(f"[voice] Heard: {text}")
        return text
