import librosa
import numpy as np
from pathlib import Path


# ==========================================
# AUDIO PREPROCESSING
# ==========================================

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Path of the audio file
AUDIO_PATH = BASE_DIR / "data" / "audio" / "sample.wav"


def load_audio(audio_path):
    """
    Load the audio file.
    """

    # Check whether the file exists
    if not audio_path.exists():
        print("\nERROR: Audio file not found!")
        print("Expected location:")
        print(audio_path)

        print("\nPlease make sure your project structure is:")
        print("""
Secure_Face_Authentication
│
├── data
│   └── audio
│       └── sample.wav
│
├── preprocessing
│   └── audio_preprocessing.py
│
└── venv
        """)

        return None, None

    try:
        # Load audio
        audio, sample_rate = librosa.load(
            str(audio_path),
            sr=None,
            mono=True
        )

        print("\nAudio loaded successfully!")

        return audio, sample_rate

    except Exception as e:
        print("\nError while loading audio:")
        print(e)

        return None, None


def normalize_audio(audio):
    """
    Normalize the audio values between -1 and 1.
    """

    if audio is None:
        return None

    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio

    normalized_audio = audio / max_value

    return normalized_audio


def extract_mfcc(audio, sample_rate):
    """
    Extract MFCC features from the audio.
    """

    if audio is None:
        return None

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    return mfcc


def preprocess_audio():
    """
    Complete audio preprocessing pipeline.
    """

    print("=" * 50)
    print("AUDIO PREPROCESSING STARTED")
    print("=" * 50)

    print("\nLooking for audio file:")
    print(AUDIO_PATH)

    # Step 1: Load audio
    audio, sample_rate = load_audio(AUDIO_PATH)

    if audio is None:
        return

    # Step 2: Display original information
    print("\nOriginal Audio Information")
    print("-" * 30)
    print("Sample Rate:", sample_rate)
    print("Number of Samples:", len(audio))
    print("Duration:", round(len(audio) / sample_rate, 2), "seconds")

    # Step 3: Normalize audio
    normalized_audio = normalize_audio(audio)

    print("\nAudio normalized successfully.")

    # Step 4: Extract MFCC features
    mfcc_features = extract_mfcc(
        normalized_audio,
        sample_rate
    )

    print("\nMFCC Features Extracted Successfully!")
    print("MFCC Shape:", mfcc_features.shape)

    print("\n" + "=" * 50)
    print("AUDIO PREPROCESSING COMPLETED")
    print("=" * 50)

    return normalized_audio, sample_rate, mfcc_features


# ==========================================
# MAIN FUNCTION
# ==========================================

if __name__ == "__main__":

    processed_audio, sample_rate, mfcc_features = preprocess_audio()