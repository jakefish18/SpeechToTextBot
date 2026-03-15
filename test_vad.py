import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from vad.utils_vad import OnnxWrapper, read_audio, get_speech_timestamps, init_jit_model
from subprocess import CalledProcessError, run

torch.set_num_threads(1)
model = init_jit_model('vad/silero_vad.jit')
MIN_PAUSE_LEN = 0.0


def to_minutes(ts):
    ts_sec = ts / 16000
    mins = ts_sec // 60
    secs = ts_sec % 60
    return f"{mins} minutes, {secs} seconds"


def get_speech_ts(WAV):
    # WAV = "sevast1.wav"
    wav = read_audio(WAV)

    # adaptive way
    speech_timestamps = get_speech_timestamps(wav, model)
    res = [(t["start"] / 16000, t["end"] / 16000) for t in speech_timestamps]
    return res


def extract_speech_pauses(inp_file):
    # audio = read_audio(inp_file)
    speech_ts = get_speech_ts(inp_file)

    pauses = []
    prevend = None
    for (start, end) in speech_ts:
        if not prevend is None:
            diff = start - prevend
            if diff > MIN_PAUSE_LEN:
                pauses.append((prevend, start))
        prevend = end
    print('Pauses: {}', pauses)
    # return pauses
    return speech_ts


if __name__ == '__main__':
    input_file = "test_data/age_physo.mp3"
    import whisper

    wisper_model = whisper.load_model("turbo")

    # Pauses: {} [(2.174, 2.626), (3.518, 3.906), (5.214, 6.146), (8.382, 9.282)]
    sentences = extract_speech_pauses(input_file)
    print(sentences)
    # print(torchaudio.info("test_data/speech.wav"))
    results = []
    # Load audio file
    waveform, sample_rate = torchaudio.load(input_file)

    print(waveform.shape, sample_rate)
    # sentences = [(2.174, 2.626), (3.518, 3.906), (5.214, 6.146), (8.382, 9.282)]
    for i, sen in enumerate(sentences):
        # Trim the tensor
        file = f'{i}.wav'
        start, end = sen
        print(start, end)
        trimmed_waveform = waveform[:, int(sample_rate * start): int(sample_rate * end)]
        torchaudio.save(f'temp.wav', trimmed_waveform, sample_rate)

        audio = whisper.load_audio("temp.wav")
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=wisper_model.dims.n_mels).to(wisper_model.device)

        # detect the spoken language
        # _, probs = wisper_model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        # options = whisper.DecodingOptions()
        """
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish"
        """
        options = whisper.DecodingOptions(language="russian")
        result = whisper.decode(wisper_model, mel, options)

        # print the recognized text
        print(result.text)

        # results.append(f'Frame Timestamp: ({start} - {end})\nSpeech: {result.text}')
        results.append(f'{result.text}')

    with open(input_file.replace('.mp3', '.txt'), 'w', encoding='utf8') as result_writer:
        result_writer.write('\n'.join(results))
