"""Voice I/O: text-to-speech alerts/answers and speech-to-text questions.

TTS uses the built-in Windows speech engine (System.Speech via PowerShell) - offline, no
extra packages. On other OSes it falls back to pyttsx3 if installed.
STT uses the SpeechRecognition package (free Google Web Speech API, needs internet).
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
import threading

_IS_WINDOWS = sys.platform == "win32"
_speaking: subprocess.Popen | None = None
_lock = threading.Lock()

_PS_SPEAK = ("Add-Type -AssemblyName System.Speech; "
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             "$s.Rate = 1; $s.Speak($env:FMD_TTS_TEXT)")
_PS_WAV = ("Add-Type -AssemblyName System.Speech; "
           "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "$s.Rate = 1; $s.SetOutputToWaveFile($env:FMD_TTS_OUT); "
           "$s.Speak($env:FMD_TTS_TEXT); $s.Dispose()")


def _powershell(script: str, **env: str) -> subprocess.Popen:
    # Text is passed through environment variables, never spliced into the command.
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
    )


def is_speaking() -> bool:
    return _speaking is not None and _speaking.poll() is None


def speak(text: str, interrupt: bool = False) -> None:
    """Speak text out loud on this computer without blocking. Skips if already speaking."""
    global _speaking
    text = clean_for_speech(text)
    if not text:
        return
    with _lock:
        if is_speaking():
            if not interrupt:
                return
            _speaking.kill()
        if _IS_WINDOWS:
            _speaking = _powershell(_PS_SPEAK, FMD_TTS_TEXT=text)
        else:
            threading.Thread(target=_pyttsx3_say, args=(text,), daemon=True).start()


def _pyttsx3_say(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print(f"[voice] {text}")


def synthesize_wav(text: str) -> bytes | None:
    """Render speech to WAV bytes (for playing in the browser). None if unavailable."""
    text = clean_for_speech(text)
    if not text:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        if _IS_WINDOWS:
            _powershell(_PS_WAV, FMD_TTS_TEXT=text, FMD_TTS_OUT=path).wait(timeout=120)
        else:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, path)
            engine.runAndWait()
        with open(path, "rb") as f:
            data = f.read()
        return data or None
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def transcribe(wav_bytes: bytes, language: str = "en-IN") -> str:
    """Speech-to-text for a recorded WAV clip. Raises RuntimeError with a readable message.

    Uses Groq Whisper when GROQ_API_KEY is set (more accurate), else Google Web Speech.
    """
    if os.getenv("GROQ_API_KEY"):
        import groq
        from config import GROQ_WHISPER_MODEL
        try:
            result = groq.Groq().audio.transcriptions.create(
                file=("question.wav", wav_bytes), model=GROQ_WHISPER_MODEL)
            text = (result.text or "").strip()
            if text:
                return text
        except groq.APIError:
            pass  # fall back to Google below
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        raise RuntimeError("Sorry, I couldn't understand the audio - please try again.")
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech recognition service unavailable: {exc}")


def clean_for_speech(text: str, max_chars: int = 700) -> str:
    """Strip Markdown so the voice doesn't read out symbols; keep it short."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)        # links -> label
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)      # tables
    text = re.sub(r"[#*_`>|~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind(".") + 1] or cut
    return text
