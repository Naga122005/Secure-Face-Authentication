"""
FEATURE ROUND-TRIP TEST
=======================

Purpose:
    Verify whether the content-domain embedded payload survives:

    Modified Content Features
            ↓
    Feature Reconstruction
            ↓
        Stego Audio
            ↓
        MFCC Extraction
            ↓
    Recovered Content Features
            ↓
       Payload Extraction

This test compares:
    1. Modified content vs recovered content
    2. Payload embedded in modified content
    3. Payload recovered from stego audio

It does NOT modify any project files.
"""

import os
import sys
import numpy as np
import librosa


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_AUDIO = os.path.join(PROJECT_ROOT, "data", "audio")

ORIGINAL_CONTENT_PATH = os.path.join(
    DATA_AUDIO,
    "content_features.npy"
)

MODIFIED_CONTENT_PATH = os.path.join(
    DATA_AUDIO,
    "modified_content_features.npy"
)

STEGO_AUDIO_PATH = os.path.join(
    DATA_AUDIO,
    "stego_audio.wav"
)


# ============================================================
# PAYLOAD SETTINGS
# ============================================================

BITS_PER_VALUE = 4
QUANTIZATION_STEP = 16.0
QUANTIZATION_LEVELS = 16


# ============================================================
# PRINT HELPERS
# ============================================================

def print_line():
    print("-" * 70)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():
    section("LOADING CONTENT FEATURES")

    if not os.path.exists(ORIGINAL_CONTENT_PATH):
        print("ERROR: Original content features not found:")
        print(ORIGINAL_CONTENT_PATH)
        return None, None

    if not os.path.exists(MODIFIED_CONTENT_PATH):
        print("ERROR: Modified content features not found:")
        print(MODIFIED_CONTENT_PATH)
        print()
        print("Run the content payload embedding step first.")
        return None, None

    original_content = np.load(ORIGINAL_CONTENT_PATH)
    modified_content = np.load(MODIFIED_CONTENT_PATH)

    print("Original content:")
    print(f"  Shape: {original_content.shape}")
    print(f"  Dtype: {original_content.dtype}")

    print()
    print("Modified content:")
    print(f"  Shape: {modified_content.shape}")
    print(f"  Dtype: {modified_content.dtype}")

    if original_content.shape != modified_content.shape:
        print()
        print("WARNING: Original and modified content shapes differ.")

    return original_content, modified_content


# ============================================================
# LOAD STEGO AUDIO
# ============================================================

def load_stego_audio():
    section("LOADING STEGO AUDIO")

    if not os.path.exists(STEGO_AUDIO_PATH):
        print("ERROR: Stego audio not found:")
        print(STEGO_AUDIO_PATH)
        return None, None

    audio, sample_rate = librosa.load(
        STEGO_AUDIO_PATH,
        sr=None,
        mono=True
    )

    print("Stego audio:")
    print(f"  Path: {STEGO_AUDIO_PATH}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Samples: {len(audio)}")
    print(f"  Duration: {len(audio) / sample_rate:.3f} seconds")

    return audio, sample_rate


# ============================================================
# EXTRACT CONTENT FROM AUDIO
# ============================================================

def extract_recovered_content(audio, sample_rate):
    section("EXTRACTING RECOVERED CONTENT FROM STEGO AUDIO")

    print("Extracting MFCC features...")

    # Same basic MFCC configuration used by the project
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    print(f"Recovered MFCC shape: {mfcc.shape}")

    # Content = first 8 MFCC coefficients
    recovered_content = mfcc[:8, :]

    # Style = remaining 5 coefficients
    recovered_style = mfcc[8:, :]

    print(f"Recovered content shape: {recovered_content.shape}")
    print(f"Recovered style shape: {recovered_style.shape}")

    return mfcc, recovered_content, recovered_style


# ============================================================
# FEATURE COMPARISON
# ============================================================

def compare_features(modified_content, recovered_content):
    section("MODIFIED CONTENT → AUDIO → RECOVERED CONTENT")

    rows = min(
        modified_content.shape[0],
        recovered_content.shape[0]
    )

    cols = min(
        modified_content.shape[1],
        recovered_content.shape[1]
    )

    modified = modified_content[:rows, :cols]
    recovered = recovered_content[:rows, :cols]

    difference = recovered - modified

    mse = np.mean(difference ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(difference))
    max_error = np.max(np.abs(difference))

    p90 = np.percentile(np.abs(difference), 90)
    p95 = np.percentile(np.abs(difference), 95)
    p99 = np.percentile(np.abs(difference), 99)

    print(f"Compared shape: {modified.shape}")
    print()
    print(f"MSE       : {mse:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"MAE       : {mae:.6f}")
    print(f"Max Error : {max_error:.6f}")
    print(f"P90 Error : {p90:.6f}")
    print(f"P95 Error : {p95:.6f}")
    print(f"P99 Error : {p99:.6f}")

    return modified, recovered


# ============================================================
# PAYLOAD SYMBOL EXTRACTION
# ============================================================

def content_to_symbols(content):
    """
    Convert content feature values into 4-bit symbols.

    Sender-side scheme:

        quantized_index = floor(value / 4.0)
        symbol = quantized_index % 16

    Therefore every content value stores 4 bits.
    """

    flattened = content.flatten()

    symbols = []

    for value in flattened:

        quantized_index = int(
            np.floor(float(value) / QUANTIZATION_STEP)
        )

        symbol = quantized_index % QUANTIZATION_LEVELS

        symbols.append(symbol)

    return np.array(symbols, dtype=np.uint8)


# ============================================================
# SYMBOLS → BITS
# ============================================================

def symbols_to_bits(symbols):
    """
    Convert each 4-bit symbol into 4 bits.

    Example:

        symbol = 10

        binary = 1010

        stored as:
        1 0 1 0
    """

    bits = []

    for symbol in symbols:

        for bit_position in range(
            BITS_PER_VALUE - 1,
            -1,
            -1
        ):

            bit = (
                int(symbol) >> bit_position
            ) & 1

            bits.append(bit)

    return np.array(bits, dtype=np.uint8)


# ============================================================
# BITS → BYTES
# ============================================================

def bits_to_bytes(bits):
    """
    Convert bit array into bytes.

    Only complete bytes are used.
    """

    usable_length = (
        len(bits) // 8
    ) * 8

    if usable_length == 0:
        return b""

    usable_bits = bits[:usable_length]

    packed = np.packbits(
        usable_bits
    )

    return packed.tobytes()


# ============================================================
# EXTRACT PAYLOAD FROM CONTENT
# ============================================================

def extract_payload_from_content(content):
    symbols = content_to_symbols(content)

    bits = symbols_to_bits(symbols)

    payload_bytes = bits_to_bytes(bits)

    return symbols, bits, payload_bytes


# ============================================================
# DISPLAY BYTE INFORMATION
# ============================================================

def display_bytes(label, data, count=32):

    print()
    print(label)

    if len(data) == 0:
        print("  No bytes recovered.")
        return

    shown = data[:count]

    print(f"  Total bytes available: {len(data)}")
    print(
        "  First bytes (hex):",
        shown.hex(" ")
    )

    try:
        print(
            "  First bytes (ASCII):",
            repr(shown[:16])
        )
    except Exception:
        pass


# ============================================================
# PAYLOAD SURVIVAL TEST
# ============================================================

def test_payload_survival(
    modified_content,
    recovered_content
):

    section("PAYLOAD SURVIVAL TEST")

    print("Extracting payload directly from MODIFIED content...")

    modified_symbols, modified_bits, modified_bytes = (
        extract_payload_from_content(
            modified_content
        )
    )

    print(
        f"Modified content symbols: "
        f"{len(modified_symbols)}"
    )

    print(
        f"Modified content bits: "
        f"{len(modified_bits)}"
    )

    print(
        f"Modified content bytes: "
        f"{len(modified_bytes)}"
    )

    display_bytes(
        "Payload from modified content:",
        modified_bytes
    )

    print()
    print("Extracting payload from RECOVERED content...")

    recovered_symbols, recovered_bits, recovered_bytes = (
        extract_payload_from_content(
            recovered_content
        )
    )

    print(
        f"Recovered content symbols: "
        f"{len(recovered_symbols)}"
    )

    print(
        f"Recovered content bits: "
        f"{len(recovered_bits)}"
    )

    print(
        f"Recovered content bytes: "
        f"{len(recovered_bytes)}"
    )

    display_bytes(
        "Payload from recovered content:",
        recovered_bytes
    )

    # --------------------------------------------------------
    # CHECK CFD1 HEADER
    # --------------------------------------------------------

    print()
    print_line()
    print("FEATURE HEADER CHECK")
    print_line()

    expected_magic = b"CFD1"

    modified_magic = modified_bytes[:4]
    recovered_magic = recovered_bytes[:4]

    print(
        f"Expected magic   : {expected_magic!r}"
    )

    print(
        f"Modified magic   : {modified_magic!r}"
    )

    print(
        f"Recovered magic  : {recovered_magic!r}"
    )

    modified_magic_ok = (
        modified_magic == expected_magic
    )

    recovered_magic_ok = (
        recovered_magic == expected_magic
    )

    print()
    print(
        "Modified content contains CFD1:",
        "PASS" if modified_magic_ok else "FAIL"
    )

    print(
        "Recovered content contains CFD1:",
        "PASS" if recovered_magic_ok else "FAIL"
    )

    # --------------------------------------------------------
    # BYTE-BY-BYTE COMPARISON
    # --------------------------------------------------------

    print()
    print_line()
    print("PAYLOAD BYTE COMPARISON")
    print_line()

    compare_length = min(
        len(modified_bytes),
        len(recovered_bytes)
    )

    if compare_length == 0:

        print("No payload bytes available.")
        return False

    matching_bytes = sum(
        1
        for i in range(compare_length)
        if modified_bytes[i] == recovered_bytes[i]
    )

    byte_accuracy = (
        matching_bytes / compare_length
    ) * 100.0

    print(
        f"Bytes compared : {compare_length}"
    )

    print(
        f"Matching bytes : {matching_bytes}"
    )

    print(
        f"Byte accuracy  : {byte_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # SYMBOL-BY-SYMBOL COMPARISON
    # --------------------------------------------------------

    print()
    print_line()
    print("4-BIT SYMBOL COMPARISON")
    print_line()

    symbol_compare_length = min(
        len(modified_symbols),
        len(recovered_symbols)
    )

    matching_symbols = np.sum(
        modified_symbols[:symbol_compare_length]
        ==
        recovered_symbols[:symbol_compare_length]
    )

    symbol_accuracy = (
        matching_symbols
        /
        symbol_compare_length
    ) * 100.0

    print(
        f"Symbols compared : "
        f"{symbol_compare_length}"
    )

    print(
        f"Matching symbols : "
        f"{matching_symbols}"
    )

    print(
        f"Symbol accuracy  : "
        f"{symbol_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # FIRST DIFFERENCES
    # --------------------------------------------------------

    print()
    print_line()
    print("FIRST PAYLOAD DIFFERENCES")
    print_line()

    differences_found = 0

    for i in range(
        min(
            len(modified_symbols),
            len(recovered_symbols)
        )
    ):

        if (
            modified_symbols[i]
            !=
            recovered_symbols[i]
        ):

            print(
                f"Symbol {i}: "
                f"modified={modified_symbols[i]} "
                f"recovered={recovered_symbols[i]}"
            )

            differences_found += 1

            if differences_found >= 20:
                break

    if differences_found == 0:
        print("No symbol differences found.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PAYLOAD SURVIVAL RESULT")
    print("=" * 70)

    if recovered_magic_ok:

        print()
        print("SUCCESS:")
        print("CFD1 survived the feature → audio → MFCC round trip.")

        if byte_accuracy >= 95:
            print(
                f"Payload byte accuracy is high: "
                f"{byte_accuracy:.2f}%"
            )

        print()
        print(
            "The receiver-side payload extraction "
            "can proceed."
        )

        return True

    else:

        print()
        print("FAILURE:")
        print(
            "CFD1 did NOT survive the feature → "
            "audio → MFCC round trip."
        )

        print()
        print(
            "This means the problem is BEFORE "
            "AES decryption/authentication."
        )

        print(
            "The receiver cannot reliably recover "
            "the encrypted biometric payload yet."
        )

        return False


# ============================================================
# ORIGINAL CONTENT COMPARISON
# ============================================================

def compare_original_and_modified(
    original_content,
    modified_content
):

    section("ORIGINAL CONTENT vs MODIFIED CONTENT")

    rows = min(
        original_content.shape[0],
        modified_content.shape[0]
    )

    cols = min(
        original_content.shape[1],
        modified_content.shape[1]
    )

    original = original_content[:rows, :cols]
    modified = modified_content[:rows, :cols]

    difference = modified - original

    mse = np.mean(difference ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(difference))
    max_error = np.max(np.abs(difference))

    print(f"Compared shape: {original.shape}")
    print()
    print(f"MSE       : {mse:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"MAE       : {mae:.6f}")
    print(f"Max Error : {max_error:.6f}")

    changed_values = np.sum(
        np.abs(difference) > 1e-12
    )

    total_values = difference.size

    changed_percentage = (
        changed_values / total_values
    ) * 100.0

    print()
    print(
        f"Changed values: "
        f"{changed_values}/{total_values}"
    )

    print(
        f"Changed percentage: "
        f"{changed_percentage:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section("FEATURE ROUND-TRIP TEST")

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Test flow:")
    print(
        "Modified Content Features"
        " → Stego Audio"
        " → MFCC"
        " → Recovered Content"
        " → Payload"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    original_content, modified_content = (
        load_features()
    )

    if (
        original_content is None
        or
        modified_content is None
    ):
        sys.exit(1)

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    compare_original_and_modified(
        original_content,
        modified_content
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    audio, sample_rate = load_stego_audio()

    if audio is None:
        sys.exit(1)

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    (
        mfcc,
        recovered_content,
        recovered_style
    ) = extract_recovered_content(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    modified_compared, recovered_compared = (
        compare_features(
            modified_content,
            recovered_content
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    payload_ok = test_payload_survival(
        modified_compared,
        recovered_compared
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    section("FINAL SUMMARY")

    print(
        f"Original content shape : "
        f"{original_content.shape}"
    )

    print(
        f"Modified content shape : "
        f"{modified_content.shape}"
    )

    print(
        f"Recovered content shape: "
        f"{recovered_content.shape}"
    )

    print()

    if payload_ok:

        print("STATUS: PAYLOAD SURVIVED")
        print()
        print(
            "Next step:"
        )
        print(
            "Proceed with receiver-side payload "
            "parsing and AES-GCM decryption."
        )

    else:

        print("STATUS: PAYLOAD DID NOT SURVIVE")
        print()
        print(
            "Do NOT proceed to AES decryption yet."
        )
        print(
            "The feature reconstruction / audio "
            "round-trip must be fixed first."
        )

    print()
    print("=" * 70)
    print("ROUND-TRIP TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()