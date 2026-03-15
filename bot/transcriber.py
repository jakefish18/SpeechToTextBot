import os
import subprocess
import tempfile
import warnings

import torch
import torchaudio
import whisper

from bot.config import WHISPER_MODEL
from vad.utils_vad import init_jit_model, read_audio, get_speech_timestamps

torch.set_num_threads(1)

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".webm", ".mpga", ".mpeg", ".flac"}

_vad_model = init_jit_model("vad/silero_vad.jit")
_whisper_model = whisper.load_model(WHISPER_MODEL)
_decode_options = whisper.DecodingOptions(language="russian")


def _to_wav(file_path: str) -> str:
    """Convert any audio file to 16kHz mono WAV using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", tmp.name],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp.name


def transcribe(file_path: str) -> str:
    wav_path = _to_wav(file_path)
    try:
        wav = read_audio(wav_path)
        speech_timestamps = get_speech_timestamps(wav, _vad_model)

        if not speech_timestamps:
            return ""

        segments = [(t["start"] / 16000, t["end"] / 16000) for t in speech_timestamps]
        waveform, sample_rate = torchaudio.load(wav_path, backend="soundfile")

        results = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name

        try:
            for start, end in segments:
                trimmed = waveform[:, int(sample_rate * start): int(sample_rate * end)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torchaudio.save(seg_path, trimmed, sample_rate)

                audio = whisper.load_audio(seg_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio, n_mels=_whisper_model.dims.n_mels).to(_whisper_model.device)
                result = whisper.decode(_whisper_model, mel, _decode_options)
                if result.text.strip():
                    results.append(result.text.strip())
        finally:
            os.remove(seg_path)

        return " ".join(results)
    finally:
        os.remove(wav_path)
