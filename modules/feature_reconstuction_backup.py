"""
Feature Reconstruction
======================

Modified Content Features + Original Style Features
                    ↓
              MFCC Reconstruction
                    ↓
              stego_audio.wav
"""

import os
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "audio"
)

MODIFIED_CONTENT_PATH = os.path.join(
    AUDIO_DIR,
    "modified_content_features.npy"
)

STYLE_PATH = os.path.join(
    AUDIO_DIR,
    "style_features.npy"
)

SELECTED_CARRIER_PATH = os.path.join(
    AUDIO_DIR,
    "selected_carrier_audio.txt"
)

OUTPUT_AUDIO_PATH = os.path.join(
    AUDIO_DIR,
    "stego_audio.wav"
)


# ============================================================
# AUDIO PARAMETERS
# ============================================================

N_MFCC = 13

N_FFT = 2048

HOP_LENGTH = 512

N_MELS = 128

N_ITER = 128


# ============================================================
# LOAD CARRIER
# ============================================================

def load_selected_carrier():

    if not os.path.exists(
        SELECTED_CARRIER_PATH
    ):

        raise FileNotFoundError(
            "Selected carrier file not found:\n"
            + SELECTED_CARRIER_PATH
        )

    with open(
        SELECTED_CARRIER_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        carrier_path = file.read().strip()

    if not os.path.exists(
        carrier_path
    ):

        raise FileNotFoundError(
            "Carrier audio does not exist:\n"
            + carrier_path
        )

    return carrier_path


# ============================================================
# MAIN RECONSTRUCTION
# ============================================================

def reconstruct_stego_audio():

    print("=" * 70)
    print(
        "FEATURE-DOMAIN STEGO AUDIO RECONSTRUCTION"
    )
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    print("\n" + "-" * 60)
    print("CHECKING INPUT FILES")
    print("-" * 60)

    required_files = [

        MODIFIED_CONTENT_PATH,

        STYLE_PATH,

        SELECTED_CARRIER_PATH
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Required file not found:\n"
                + path
            )

        print(
            "Found:",
            path
        )

    # ========================================================
    # LOAD CONTENT
    # ========================================================

    print("\n" + "-" * 60)
    print("LOADING MODIFIED CONTENT")
    print("-" * 60)

    modified_content = np.load(
        MODIFIED_CONTENT_PATH
    ).astype(np.float32)

    print(
        "Modified content:",
        modified_content.shape
    )

    # ========================================================
    # LOAD STYLE
    # ========================================================

    print("\n" + "-" * 60)
    print("LOADING ORIGINAL STYLE")
    print("-" * 60)

    style = np.load(
        STYLE_PATH
    ).astype(np.float32)

    print(
        "Style:",
        style.shape
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    print("\n" + "-" * 60)
    print("CHECKING FEATURE DIMENSIONS")
    print("-" * 60)

    if modified_content.ndim != 2:

        raise ValueError(
            "Modified content must be 2-D."
        )

    if style.ndim != 2:

        raise ValueError(
            "Style must be 2-D."
        )

    if modified_content.shape[0] != 8:

        raise ValueError(
            "Expected content shape (8, frames)."
        )

    if style.shape[0] != 5:

        raise ValueError(
            "Expected style shape (5, frames)."
        )

    if (
        modified_content.shape[1]
        !=
        style.shape[1]
    ):

        raise ValueError(
            "Content and style frame counts differ."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    print("\n" + "-" * 60)
    print("COMBINING CONTENT + STYLE")
    print("-" * 60)

    reconstructed_mfcc = np.vstack(
        (
            modified_content,
            style
        )
    ).astype(np.float32)

    print(
        "Reconstructed MFCC:",
        reconstructed_mfcc.shape
    )

    # ========================================================
    # LOAD CARRIER
    # ========================================================

    print("\n" + "-" * 60)
    print("LOADING CARRIER AUDIO")
    print("-" * 60)

    carrier_path = (
        load_selected_carrier()
    )

    print(
        "Carrier:",
        carrier_path
    )

    carrier_audio, sample_rate = (
        librosa.load(
            carrier_path,
            sr=None,
            mono=True
        )
    )

    original_length = (
        len(carrier_audio)
    )

    print(
        "Sample rate:",
        sample_rate
    )

    print(
        "Original samples:",
        original_length
    )

    # ========================================================
    # MFCC → MEL
    # ========================================================

    print("\n" + "-" * 60)
    print("CONVERTING MFCC → MEL SPECTROGRAM")
    print("-" * 60)

    mel_spectrogram = (
        librosa.feature.inverse.mfcc_to_mel(
            reconstructed_mfcc,
            n_mels=N_MELS,
            dct_type=2,
            norm="ortho"
        )
    )

    mel_spectrogram = np.nan_to_num(
        mel_spectrogram,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mel_spectrogram = np.maximum(
        mel_spectrogram,
        1e-10
    )

    print(
        "Mel spectrogram:",
        mel_spectrogram.shape
    )

    # ========================================================
    # MEL → AUDIO
    # ========================================================

    print("\n" + "-" * 60)
    print("RECONSTRUCTING AUDIO")
    print("-" * 60)

    reconstructed_audio = (
        librosa.feature.inverse.mel_to_audio(
            mel_spectrogram,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            n_iter=N_ITER
        )
    )

    reconstructed_audio = np.nan_to_num(
        reconstructed_audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # ========================================================
    # MATCH LENGTH
    # ========================================================

    print("\n" + "-" * 60)
    print("MATCHING AUDIO LENGTH")
    print("-" * 60)

    if len(reconstructed_audio) > original_length:

        reconstructed_audio = (
            reconstructed_audio[
                :original_length
            ]
        )

    elif len(reconstructed_audio) < original_length:

        reconstructed_audio = np.pad(
            reconstructed_audio,
            (
                0,
                original_length
                -
                len(reconstructed_audio)
            ),
            mode="constant"
        )

    print(
        "Reconstructed samples:",
        len(reconstructed_audio)
    )

    # ========================================================
    # SAFE CLIPPING
    # ========================================================

    print("\n" + "-" * 60)
    print("PREPARING AUDIO FOR WAV OUTPUT")
    print("-" * 60)

    max_value = np.max(
        np.abs(reconstructed_audio)
    )

    print(
        "Maximum absolute amplitude:",
        float(max_value)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT normalize the audio.
    #
    # Normalization changes signal energy and therefore
    # changes MFCC coefficients.
    #
    # We only scale down if clipping would occur.
    # --------------------------------------------------------

    if max_value > 0.99:

        reconstructed_audio = (
            reconstructed_audio
            / max_value
            * 0.99
        )

        print(
            "Audio exceeded safe range."
        )

        print(
            "Applied minimum necessary scaling."
        )

    else:

        print(
            "No normalization required."
        )

    reconstructed_audio = np.clip(
        reconstructed_audio,
        -0.999,
        0.999
    )

    reconstructed_audio = (
        reconstructed_audio
        .astype(np.float32)
    )

    # ========================================================
    # SAVE WAV
    # ========================================================

    print("\n" + "-" * 60)
    print("SAVING STEGO AUDIO")
    print("-" * 60)

    sf.write(
        OUTPUT_AUDIO_PATH,
        reconstructed_audio,
        sample_rate,
        subtype="PCM_16"
    )

    print(
        "Stego audio saved:"
    )

    print(
        OUTPUT_AUDIO_PATH
    )

    # ========================================================
    # VERIFY
    # ========================================================

    print("\n" + "-" * 60)
    print("VERIFYING STEGO AUDIO")
    print("-" * 60)

    if not os.path.exists(
        OUTPUT_AUDIO_PATH
    ):

        raise RuntimeError(
            "Stego audio was not created."
        )

    check_audio, check_sr = (
        sf.read(
            OUTPUT_AUDIO_PATH
        )
    )

    print(
        "Output sample rate:",
        check_sr
    )

    print(
        "Output samples:",
        len(check_audio)
    )

    print(
        "Output duration:",
        round(
            len(check_audio)
            /
            check_sr,
            3
        ),
        "seconds"
    )

    print(
        "Output file size:",
        round(
            os.path.getsize(
                OUTPUT_AUDIO_PATH
            ) / 1024,
            2
        ),
        "KB"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "STEGO AUDIO RECONSTRUCTION COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Modified content:",
        modified_content.shape
    )

    print(
        "Original style:",
        style.shape
    )

    print(
        "Combined MFCC:",
        reconstructed_mfcc.shape
    )

    print()
    print(
        "Stego audio:"
    )

    print(
        OUTPUT_AUDIO_PATH
    )

    print()
    print(
        "Next:"
    )

    print(
        "Run the feature round-trip test."
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    reconstruct_stego_audio()