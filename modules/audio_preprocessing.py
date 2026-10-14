import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


def preprocess_audio(audio_path):
    """
    Preprocess the selected carrier audio.

    Steps:
        1. Load WAV audio
        2. Convert to mono
        3. Resample to 16 kHz
        4. Normalize amplitude
        5. Extract MFCC features

    Returns:
        audio          : preprocessed audio signal
        sample_rate   : sample rate
        mfcc           : MFCC features
    """

    audio_path = Path(audio_path)

    print("\n")
    print("=" * 60)
    print("             AUDIO PREPROCESSING")
    print("=" * 60)

    print(f"\nInput audio:")
    print(audio_path)

    # ---------------------------------------------------------
    # 1. Load audio
    # ---------------------------------------------------------
    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True
    )

    print(f"\nOriginal/processed sample rate : {sample_rate} Hz")
    print(f"Audio samples                  : {len(audio)}")
    print(
        f"Audio duration                : "
        f"{len(audio) / sample_rate:.2f} seconds"
    )

    # ---------------------------------------------------------
    # 2. Normalize audio
    # ---------------------------------------------------------
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    print("Normalization                  : Completed")

    # ---------------------------------------------------------
    # 3. MFCC extraction
    # ---------------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    print(f"MFCC shape                     : {mfcc.shape}")

    print("\nAudio preprocessing completed.")
    print("=" * 60)

    return audio, sample_rate, mfcc


def save_preprocessed_audio(audio, sample_rate, output_path):
    """
    Save preprocessed audio as WAV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(output_path),
        audio,
        sample_rate
    )

    print(f"\nPreprocessed audio saved to:")
    print(output_path)


if __name__ == "__main__":

    # Test with carrier audio selector
    from modules.carrier_audio_selector import select_carrier_audio

    # ---------------------------------------------------------
    # 1. Select carrier audio
    # ---------------------------------------------------------
    carrier_audio = select_carrier_audio()

    # ---------------------------------------------------------
    # 2. Preprocess selected carrier
    # ---------------------------------------------------------
    audio, sample_rate, mfcc = preprocess_audio(
        carrier_audio
    )

    # ---------------------------------------------------------
    # 3. Save preprocessed audio
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent

    output_path = (
        project_root
        / "Data"
        / "audio"
        / "preprocessed"
        / "preprocessed_carrier.wav"
    )

    save_preprocessed_audio(
        audio,
        sample_rate,
        output_path
    )

    # ---------------------------------------------------------
    # 4. Display final result
    # ---------------------------------------------------------
    print("\n")
    print("=" * 60)
    print("                 FINAL RESULT")
    print("=" * 60)

    print(f"Carrier audio : {carrier_audio.name}")
    print(f"MFCC shape    : {mfcc.shape}")
    print(f"Saved audio   : {output_path}")

    print("=" * 60)