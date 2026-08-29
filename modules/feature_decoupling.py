import os
import numpy as np
import librosa


def preprocess_audio(audio_path):
    """
    Load the audio file, normalize it,
    and extract 13 MFCC features.
    """

    print("=" * 50)
    print("AUDIO PREPROCESSING STARTED")
    print("=" * 50)

    print("\nLooking for audio file:")
    print(audio_path)

    # Check whether the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"\nAudio file not found:\n{audio_path}"
        )

    # Load audio
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None
    )

    print("\nAudio loaded successfully!")

    print("\nOriginal Audio Information")
    print("-" * 30)
    print("Sample Rate:", sample_rate)
    print("Number of Samples:", len(audio))
    print(
        "Duration:",
        round(len(audio) / sample_rate, 2),
        "seconds"
    )

    # Normalize audio
    max_value = np.max(np.abs(audio))

    if max_value != 0:
        audio = audio / max_value

    print("\nAudio normalized successfully.")

    # Extract 13 MFCC features
    mfcc_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    print("\nMFCC Features Extracted Successfully!")
    print("MFCC Shape:", mfcc_features.shape)

    print("\n" + "=" * 50)
    print("AUDIO PREPROCESSING COMPLETED")
    print("=" * 50)

    return mfcc_features


def decouple_audio_features(audio_path=None):
    """
    Decouple the MFCC features into:

    Content Features:
        First 8 MFCC coefficients

    Style Features:
        Remaining 5 MFCC coefficients

    The style features are saved as:
        data/audio/style_features.npy
    """

    print("\n" + "=" * 50)
    print("AUDIO FEATURE DECOUPLING STARTED")
    print("=" * 50)

    # Get project root directory
    # D:\Secure_Face_Authentication
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Use default audio file if no path is provided
    if audio_path is None:

        audio_path = os.path.join(
            project_root,
            "data",
            "audio",
            "sample.wav"
        )

    # --------------------------------------------------
    # STEP 1: PREPROCESS AUDIO
    # --------------------------------------------------

    mfcc_features = preprocess_audio(audio_path)

    print("\nInput MFCC Shape:")
    print(mfcc_features.shape)

    # --------------------------------------------------
    # STEP 2: EXTRACT CONTENT FEATURES
    # --------------------------------------------------

    # First 8 MFCC coefficients
    content_features = mfcc_features[:8, :]

    print("\nContent Features Extracted Successfully!")
    print(
        "Content Feature Shape:",
        content_features.shape
    )

    # --------------------------------------------------
    # STEP 3: EXTRACT STYLE FEATURES
    # --------------------------------------------------

    # Remaining 5 MFCC coefficients
    style_features = mfcc_features[8:, :]

    print("\nStyle Features Extracted Successfully!")
    print(
        "Style Feature Shape:",
        style_features.shape
    )

    # --------------------------------------------------
    # STEP 4: SAVE STYLE FEATURES
    # --------------------------------------------------

    audio_data_folder = os.path.join(
        project_root,
        "data",
        "audio"
    )

    # Create the folder if it does not exist
    os.makedirs(
        audio_data_folder,
        exist_ok=True
    )

    style_output_path = os.path.join(
        audio_data_folder,
        "style_features.npy"
    )

    # Save style features
    np.save(
        style_output_path,
        style_features
    )

    print("\nStyle Features Saved Successfully!")
    print("Saved at:")
    print(style_output_path)

    print("\n" + "=" * 50)
    print("AUDIO FEATURE DECOUPLING COMPLETED")
    print("=" * 50)

    # Return both features
    return content_features, style_features


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    content_features, style_features = decouple_audio_features()