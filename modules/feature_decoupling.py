import os
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):
    """
    Load and preprocess the selected VCC2020 carrier audio.

    Processing:
        1. Check audio file
        2. Load WAV using SoundFile
        3. Convert stereo to mono
        4. Convert audio to float32
        5. Normalize amplitude
        6. Extract 13 MFCC coefficients

    Returns:
        mfcc_features
        sample_rate
    """

    print("\n" + "=" * 60)
    print("AUDIO PREPROCESSING STARTED")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: CHECK FILE
    # --------------------------------------------------------

    print("\nLooking for audio file:")
    print(audio_path)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"\nAudio file not found:\n{audio_path}"
        )

    # --------------------------------------------------------
    # STEP 2: LOAD AUDIO
    # --------------------------------------------------------
    # SoundFile is used instead of librosa.load()
    # to avoid audioread backend problems.
    # --------------------------------------------------------

    try:

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

    except Exception as e:

        raise RuntimeError(
            "\nUnable to read audio file.\n"
            f"File: {audio_path}\n"
            f"Error: {e}"
        )

    print("\nAudio loaded successfully!")

    # --------------------------------------------------------
    # STEP 3: AUDIO INFORMATION
    # --------------------------------------------------------

    print("\nOriginal Audio Information")
    print("-" * 40)

    print("Sample Rate      :", sample_rate)
    print("Number of Samples:", len(audio))

    duration = len(audio) / sample_rate

    print(
        "Duration         :",
        round(duration, 2),
        "seconds"
    )

    # --------------------------------------------------------
    # STEP 4: CONVERT STEREO TO MONO
    # --------------------------------------------------------

    if audio.ndim > 1:

        print("\nStereo audio detected.")
        print("Converting stereo audio to mono...")

        audio = np.mean(
            audio,
            axis=1
        )

        print("Stereo to mono conversion completed.")

    else:

        print("\nAudio is already mono.")

    # --------------------------------------------------------
    # STEP 5: FLOAT32 CONVERSION
    # --------------------------------------------------------

    audio = audio.astype(
        np.float32
    )

    # --------------------------------------------------------
    # STEP 6: NORMALIZATION
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = audio / max_value

        print(
            "\nAudio normalized successfully."
        )

    else:

        print(
            "\nWARNING: Audio contains only zero values."
        )

    # --------------------------------------------------------
    # STEP 7: MFCC EXTRACTION
    # --------------------------------------------------------

    print("\nExtracting MFCC features...")

    mfcc_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    print(
        "\nMFCC Features Extracted Successfully!"
    )

    print(
        "MFCC Shape:",
        mfcc_features.shape
    )

    print(
        "Number of MFCC Coefficients:",
        mfcc_features.shape[0]
    )

    print(
        "Number of Time Frames:",
        mfcc_features.shape[1]
    )

    print("\n" + "=" * 60)
    print("AUDIO PREPROCESSING COMPLETED")
    print("=" * 60)

    return mfcc_features, sample_rate


# ============================================================
# AUDIO FEATURE DECOUPLING
# ============================================================

def decouple_audio_features(audio_path=None):
    """
    Decouple the carrier audio into content and style features.

    MFCC representation:
        13 coefficients

    Content features:
        First 8 MFCC coefficients

    Style features:
        Remaining 5 MFCC coefficients

    Returns:
        content_features
        style_features
    """

    print("\n" + "=" * 60)
    print("AUDIO FEATURE DECOUPLING STARTED")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: FIND PROJECT ROOT
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    print("\nProject Root:")
    print(project_root)

    # --------------------------------------------------------
    # STEP 2: SELECT CARRIER AUDIO
    # --------------------------------------------------------

    if audio_path is None:

        print(
            "\nNo audio path supplied."
        )

        print(
            "Selecting carrier audio from VCC2020 dataset..."
        )

        try:

            from modules.carrier_audio_selector import (
                select_carrier_audio
            )

            audio_path = select_carrier_audio()

        except Exception as e:

            raise RuntimeError(
                "\nUnable to select carrier audio "
                "from VCC2020 dataset.\n"
                f"Error: {e}"
            )

    # --------------------------------------------------------
    # STEP 3: CONVERT PATH TO STRING
    # --------------------------------------------------------
    # carrier_audio_selector may return a WindowsPath.
    # Converting it to str prevents:
    #
    # TypeError: write() argument must be str,
    # not WindowsPath
    # --------------------------------------------------------

    audio_path = str(audio_path)

    print("\nSelected Carrier Audio:")
    print(audio_path)

    # --------------------------------------------------------
    # STEP 4: PREPROCESS AUDIO
    # --------------------------------------------------------

    mfcc_features, sample_rate = preprocess_audio(
        audio_path
    )

    # --------------------------------------------------------
    # STEP 5: DISPLAY MFCC INFORMATION
    # --------------------------------------------------------

    print("\nInput MFCC Information")
    print("-" * 40)

    print(
        "MFCC Shape:",
        mfcc_features.shape
    )

    # --------------------------------------------------------
    # STEP 6: EXTRACT CONTENT FEATURES
    # --------------------------------------------------------
    #
    # First 8 MFCC coefficients
    #
    # Example:
    #
    # MFCC = (13, 146)
    #
    # Content = (8, 146)
    #
    # --------------------------------------------------------

    content_features = mfcc_features[:8, :]

    print(
        "\nContent Features Extracted Successfully!"
    )

    print(
        "Content Feature Shape:",
        content_features.shape
    )

    # --------------------------------------------------------
    # STEP 7: EXTRACT STYLE FEATURES
    # --------------------------------------------------------
    #
    # Remaining 5 MFCC coefficients
    #
    # Example:
    #
    # Style = (5, 146)
    #
    # --------------------------------------------------------

    style_features = mfcc_features[8:, :]

    print(
        "\nStyle Features Extracted Successfully!"
    )

    print(
        "Style Feature Shape:",
        style_features.shape
    )

    # --------------------------------------------------------
    # STEP 8: CREATE AUDIO DATA DIRECTORY
    # --------------------------------------------------------

    audio_data_folder = os.path.join(
        project_root,
        "data",
        "audio"
    )

    os.makedirs(
        audio_data_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 9: SAVE CONTENT FEATURES
    # --------------------------------------------------------

    content_output_path = os.path.join(
        audio_data_folder,
        "content_features.npy"
    )

    np.save(
        content_output_path,
        content_features
    )

    print(
        "\nContent Features Saved Successfully!"
    )

    print(
        "Saved at:"
    )

    print(
        content_output_path
    )

    # --------------------------------------------------------
    # STEP 10: SAVE STYLE FEATURES
    # --------------------------------------------------------

    style_output_path = os.path.join(
        audio_data_folder,
        "style_features.npy"
    )

    np.save(
        style_output_path,
        style_features
    )

    print(
        "\nStyle Features Saved Successfully!"
    )

    print(
        "Saved at:"
    )

    print(
        style_output_path
    )

    # --------------------------------------------------------
    # STEP 11: SAVE SELECTED CARRIER AUDIO PATH
    # --------------------------------------------------------

    carrier_path_file = os.path.join(
        audio_data_folder,
        "selected_carrier_audio.txt"
    )

    with open(
        carrier_path_file,
        "w",
        encoding="utf-8"
    ) as file:

        # IMPORTANT:
        # audio_path is converted to str above.
        file.write(
            str(audio_path)
        )

    print(
        "\nSelected Carrier Audio Path Saved Successfully!"
    )

    print(
        "Saved at:"
    )

    print(
        carrier_path_file
    )

    # --------------------------------------------------------
    # STEP 12: FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO FEATURE DECOUPLING COMPLETED")
    print("=" * 60)

    print("\nFINAL OUTPUT")
    print("-" * 40)

    print(
        "Carrier Audio :",
        audio_path
    )

    print(
        "Sample Rate   :",
        sample_rate
    )

    print(
        "MFCC          :",
        mfcc_features.shape
    )

    print(
        "Content       :",
        content_features.shape
    )

    print(
        "Style         :",
        style_features.shape
    )

    print("\nSaved Files:")
    print(
        "1.",
        content_output_path
    )
    print(
        "2.",
        style_output_path
    )
    print(
        "3.",
        carrier_path_file
    )

    print("=" * 60)

    return content_features, style_features


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:

        content_features, style_features = (
            decouple_audio_features()
        )

        print("\n")
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)

        print(
            "\nContent Feature Shape:",
            content_features.shape
        )

        print(
            "Style Feature Shape:",
            style_features.shape
        )

        print(
            "\nAudio preprocessing and feature "
            "decoupling completed successfully."
        )

        print("=" * 60)

    except Exception as e:

        print("\n")
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            "\n",
            str(e)
        )

        print("=" * 60)

        raise