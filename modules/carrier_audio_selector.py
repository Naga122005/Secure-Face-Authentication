from pathlib import Path
import math

import numpy as np
import soundfile as sf
import librosa


# ============================================================
# SETTINGS
# ============================================================

# Number of content MFCC coefficients
CONTENT_COEFFICIENTS = 8

# Number of bits embedded into each content coefficient
BITS_PER_VALUE = 4

# Extra safety margin
MIN_REQUIRED_FRAMES = 550


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def get_project_root():
    return Path(__file__).resolve().parent.parent


# ============================================================
# CALCULATE PAYLOAD SIZE
# ============================================================

def calculate_required_frames(project_root):
    """
    Calculate the number of MFCC frames required to store
    the current encoded biometric payload.

    steganographic_embedding.py adds:
        4 bytes  -> CFD1 magic
        4 bytes  -> payload length

    Therefore 8 extra bytes are added.
    """

    payload_file = (
        project_root
        / "data"
        / "audio"
        / "encoded_biometric_payload.bin"
    )

    if not payload_file.exists():
        print()
        print("WARNING")
        print("----------------------------------------")
        print("Encoded biometric payload was not found.")
        print(f"Expected file:")
        print(payload_file)
        print()
        print("Using default minimum requirement:")
        print(f"{MIN_REQUIRED_FRAMES} MFCC frames")
        print("----------------------------------------")

        return MIN_REQUIRED_FRAMES

    payload_size = payload_file.stat().st_size

    # Feature-domain wrapper:
    # CFD1 = 4 bytes
    # Length = 4 bytes
    wrapper_size = 8

    total_payload_size = payload_size + wrapper_size
    total_payload_bits = total_payload_size * 8

    bits_per_frame = CONTENT_COEFFICIENTS * BITS_PER_VALUE

    required_frames = math.ceil(
        total_payload_bits / bits_per_frame
    )

    # Add safety margin
    required_frames = max(
        required_frames,
        MIN_REQUIRED_FRAMES
    )

    print()
    print("PAYLOAD CAPACITY REQUIREMENT")
    print("=" * 60)
    print(f"Original payload size : {payload_size} bytes")
    print(f"Feature wrapper       : {wrapper_size} bytes")
    print(f"Total payload         : {total_payload_size} bytes")
    print(f"Total payload bits    : {total_payload_bits}")
    print()
    print(f"Content coefficients  : {CONTENT_COEFFICIENTS}")
    print(f"Bits per value        : {BITS_PER_VALUE}")
    print(
        f"Capacity per frame    : "
        f"{bits_per_frame} bits"
    )
    print()
    print(f"Calculated frames     : {required_frames}")
    print("=" * 60)

    return required_frames


# ============================================================
# CHECK ONE WAV FILE
# ============================================================

def get_mfcc_frame_count(audio_file):
    """
    Load a valid WAV file and calculate the exact MFCC
    frame count used by the project.
    """

    try:
        audio, sample_rate = sf.read(
            audio_file,
            dtype="float32"
        )

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Normalize
        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # Use the same MFCC settings as feature_decoupling.py
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13
        )

        return mfcc.shape[1]

    except Exception:
        return None


# ============================================================
# SELECT CARRIER AUDIO
# ============================================================

def select_carrier_audio():

    project_root = get_project_root()

    dataset_path = (
        project_root
        / "Data"
        / "audio_dtaset"
        / "VCC2020-database-master"
        / "vcc2020_database_evaluation"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\nVCC2020 dataset not found:\n"
            f"{dataset_path}"
        )

    # Calculate payload requirement
    required_frames = calculate_required_frames(
        project_root
    )

    print()
    print("=" * 70)
    print("             CARRIER AUDIO SELECTION")
    print("=" * 70)

    print("Dataset location :")
    print(dataset_path)

    # --------------------------------------------------------
    # FIND ALL WAV FILES
    # --------------------------------------------------------

    all_wav_files = list(
        dataset_path.rglob("*.wav")
    )

    print()
    print(f"All WAV files    : {len(all_wav_files)}")

    if not all_wav_files:
        raise FileNotFoundError(
            "\nNo WAV files were found in the VCC2020 dataset."
        )

    # --------------------------------------------------------
    # FILTER INVALID MACOS FILES
    # --------------------------------------------------------

    valid_files = []

    for audio_file in all_wav_files:

        # Ignore Apple metadata files
        if audio_file.name.startswith("._"):
            continue

        if audio_file.name.startswith("."):
            continue

        if "__MACOSX" in audio_file.parts:
            continue

        try:
            info = sf.info(audio_file)

            if info.frames <= 0:
                continue

            if info.samplerate <= 0:
                continue

            valid_files.append(audio_file)

        except Exception:
            continue

    print(f"Valid WAV files  : {len(valid_files)}")

    if not valid_files:
        raise FileNotFoundError(
            "\nNo valid WAV files were found."
        )

    # --------------------------------------------------------
    # CHECK MFCC CAPACITY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING AUDIO CAPACITY")
    print("=" * 70)

    suitable_files = []

    checked_count = 0

    for audio_file in valid_files:

        checked_count += 1

        frame_count = get_mfcc_frame_count(
            audio_file
        )

        if frame_count is None:
            continue

        capacity_bits = (
            frame_count
            * CONTENT_COEFFICIENTS
            * BITS_PER_VALUE
        )

        enough_capacity = (
            frame_count >= required_frames
        )

        if enough_capacity:

            suitable_files.append(
                (
                    audio_file,
                    frame_count,
                    capacity_bits
                )
            )

        print(
            f"[{checked_count:3d}/{len(valid_files):3d}] "
            f"{audio_file.name:15s} "
            f"Frames: {frame_count:4d} "
            f"Capacity: {capacity_bits:6d} bits "
            f"{'OK' if enough_capacity else 'NOT ENOUGH'}"
        )

    # --------------------------------------------------------
    # NO SUITABLE FILE
    # --------------------------------------------------------

    if not suitable_files:

        print()
        print("=" * 70)
        print("NO SUITABLE CARRIER FOUND")
        print("=" * 70)

        print()
        print(
            f"Required MFCC frames : {required_frames}"
        )

        print(
            "No VCC2020 WAV file has enough feature capacity "
            "for the current payload."
        )

        print()
        print(
            "Possible solutions:"
        )

        print(
            "1. Use a longer carrier audio."
        )

        print(
            "2. Reduce the payload size."
        )

        print(
            "3. Use a more efficient embedding method."
        )

        print()
        raise ValueError(
            "No VCC2020 carrier has sufficient feature-domain capacity."
        )

    # --------------------------------------------------------
    # SELECT THE SMALLEST SUITABLE FILE
    # --------------------------------------------------------
    #
    # We select the shortest file that still has enough
    # capacity. This avoids unnecessarily using a very long
    # audio file.
    # --------------------------------------------------------

    suitable_files.sort(
        key=lambda item: item[1]
    )

    selected_audio = suitable_files[0][0]
    selected_frames = suitable_files[0][1]
    selected_capacity = suitable_files[0][2]

    # Get final audio information
    info = sf.info(selected_audio)

    duration = info.frames / info.samplerate

    # --------------------------------------------------------
    # SAVE SELECTED AUDIO PATH
    # --------------------------------------------------------

    output_directory = (
        project_root
        / "data"
        / "audio"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_path_file = (
        output_directory
        / "selected_carrier_audio.txt"
    )

    selected_path_file.write_text(
        str(selected_audio),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("             SELECTED CARRIER")
    print("=" * 70)

    print()
    print("Selected carrier :")
    print(selected_audio)

    print()
    print(f"File name        : {selected_audio.name}")
    print(f"Sample rate      : {info.samplerate} Hz")
    print(f"Channels         : {info.channels}")
    print(f"Samples          : {info.frames}")
    print(f"Duration         : {duration:.2f} seconds")

    print()
    print("FEATURE CAPACITY")
    print("----------------------------------------")
    print(f"MFCC frames      : {selected_frames}")
    print(
        f"Content values   : "
        f"{selected_frames * CONTENT_COEFFICIENTS}"
    )
    print(
        f"Capacity         : "
        f"{selected_capacity} bits"
    )
    print(
        f"Required         : "
        f"{required_frames * CONTENT_COEFFICIENTS * BITS_PER_VALUE} bits"
    )

    print()
    print(
        "Capacity status  : SUFFICIENT"
    )

    print()
    print("Saved carrier path:")
    print(selected_path_file)

    print()
    print("=" * 70)
    print("Carrier audio selected successfully!")
    print("=" * 70)

    return selected_audio


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        selected = select_carrier_audio()

    except Exception as error:

        print()
        print("=" * 70)
        print("CARRIER SELECTION FAILED")
        print("=" * 70)

        print()
        print(error)

        print()
        print("=" * 70)