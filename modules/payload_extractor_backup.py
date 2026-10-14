"""
RECEIVER SIDE - PAYLOAD EXTRACTION

Flow:
    received_stego_audio.wav
            |
            v
       MFCC extraction
            |
            v
     8 content features
            |
            v
    Extract 4 bits/value
            |
            v
       CFD1 header
            |
            v
      FACEAUD1 payload
            |
            v
 encrypted biometric payload
"""

import os
import struct
import hashlib
import numpy as np
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")

STEGO_AUDIO = os.path.join(
    AUDIO_DIR,
    "received_stego_audio.wav"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "receiver"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Sender configuration
SAMPLE_RATE = 24000

N_MFCC = 13

CONTENT_DIM = 8

BITS_PER_VALUE = 4

QUANTIZATION_STEP = 4.0

# Sender embeds:
# CFD1 header + FACEAUD1 payload
FEATURE_HEADER_SIZE = 41

PAYLOAD_HEADER_SIZE = 45

BIOMETRIC_SIZE = 553

TOTAL_PAYLOAD_SIZE = (
    FEATURE_HEADER_SIZE
    + PAYLOAD_HEADER_SIZE
    + BIOMETRIC_SIZE
)


# ============================================================
# UTILITY
# ============================================================

def bytes_to_bits(data):
    """Convert bytes into a list of bits."""

    bits = []

    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits


def bits_to_bytes(bits):
    """Convert a list of bits back into bytes."""

    if len(bits) % 8 != 0:
        raise ValueError(
            "Number of bits must be divisible by 8"
        )

    result = bytearray()

    for i in range(0, len(bits), 8):

        byte = 0

        for bit in bits[i:i + 8]:
            byte = (byte << 1) | int(bit)

        result.append(byte)

    return bytes(result)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_stego_audio():

    print("\n" + "-" * 60)
    print("CHECKING STEGO AUDIO")
    print("-" * 60)

    if not os.path.exists(STEGO_AUDIO):

        raise FileNotFoundError(
            f"Stego audio not found:\n{STEGO_AUDIO}"
        )

    print("Stego audio found:")
    print(STEGO_AUDIO)

    print("\n" + "-" * 60)
    print("LOADING STEGO AUDIO")
    print("-" * 60)

    audio, sr = librosa.load(
        STEGO_AUDIO,
        sr=None,
        mono=True
    )

    print(f"Sample rate: {sr}")
    print(f"Samples: {len(audio)}")
    print(
        f"Duration: {len(audio) / sr:.3f} seconds"
    )

    if sr != SAMPLE_RATE:

        print(
            f"WARNING: Expected {SAMPLE_RATE} Hz "
            f"but received {sr} Hz"
        )

    return audio, sr


# ============================================================
# MFCC EXTRACTION
# ============================================================

def extract_mfcc(audio, sr):

    print("\n" + "-" * 60)
    print("EXTRACTING MFCC FEATURES")
    print("-" * 60)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    print(f"MFCC shape: {mfcc.shape}")

    return mfcc


# ============================================================
# CONTENT EXTRACTION
# ============================================================

def extract_content_features(mfcc):

    print("\n" + "-" * 60)
    print("EXTRACTING CONTENT FEATURES")
    print("-" * 60)

    content = mfcc[:CONTENT_DIM, :]

    print(
        f"Content feature shape: {content.shape}"
    )

    return content


# ============================================================
# SYMBOL EXTRACTION
# ============================================================

def extract_symbols(content):

    """
    IMPORTANT:

    Sender:
        symbol = floor(value / 4.0) % 16

    Therefore receiver MUST use exactly
    the same operation.
    """

    print("\n" + "-" * 60)
    print("EXTRACTING 4-BIT SYMBOLS")
    print("-" * 60)

    flat_content = content.flatten()

    symbols = np.floor(
        flat_content / QUANTIZATION_STEP
    ).astype(np.int64) % 16

    print(
        f"Number of content values: "
        f"{len(flat_content)}"
    )

    print(
        f"Number of symbols: "
        f"{len(symbols)}"
    )

    return symbols


# ============================================================
# SYMBOLS → BITS
# ============================================================

def symbols_to_bits(symbols):

    print("\n" + "-" * 60)
    print("CONVERTING SYMBOLS TO BITS")
    print("-" * 60)

    bits = []

    for symbol in symbols:

        symbol = int(symbol)

        for i in range(
            BITS_PER_VALUE - 1,
            -1,
            -1
        ):

            bits.append(
                (symbol >> i) & 1
            )

    print(
        f"Extracted bits: {len(bits)}"
    )

    print(
        f"Extracted bytes: "
        f"{len(bits) // 8}"
    )

    return bits


# ============================================================
# FEATURE HEADER
# ============================================================

def parse_feature_header(data):

    print("\n" + "-" * 60)
    print("CHECKING CFD1 FEATURE HEADER")
    print("-" * 60)

    if len(data) < FEATURE_HEADER_SIZE:

        raise ValueError(
            "Not enough data for CFD1 header"
        )

    header = data[:FEATURE_HEADER_SIZE]

    magic = header[:4]

    print(f"Recovered magic: {magic}")

    if magic != b"CFD1":

        raise ValueError(
            f"Invalid feature magic: {magic!r}\n"
            f"Expected: b'CFD1'\n\n"
            f"The stego audio did not preserve "
            f"the sender's embedded feature payload."
        )

    print("CFD1 header detected successfully.")

    return header


# ============================================================
# FACEAUD1 PAYLOAD
# ============================================================

def extract_faceaud_payload(data):

    print("\n" + "-" * 60)
    print("EXTRACTING FACEAUD1 PAYLOAD")
    print("-" * 60)

    start = FEATURE_HEADER_SIZE

    end = (
        FEATURE_HEADER_SIZE
        + PAYLOAD_HEADER_SIZE
        + BIOMETRIC_SIZE
    )

    if len(data) < end:

        raise ValueError(
            f"Not enough extracted data.\n"
            f"Required: {end} bytes\n"
            f"Available: {len(data)} bytes"
        )

    payload = data[start:end]

    magic = payload[:8]

    print(
        f"Payload magic: {magic}"
    )

    if magic != b"FACEAUD1":

        raise ValueError(
            f"Invalid payload magic: {magic!r}\n"
            f"Expected: b'FACEAUD1'"
        )

    print(
        "FACEAUD1 payload detected."
    )

    return payload


# ============================================================
# PAYLOAD VERIFICATION
# ============================================================

def verify_payload(payload):

    print("\n" + "-" * 60)
    print("VERIFYING PAYLOAD")
    print("-" * 60)

    header = payload[:PAYLOAD_HEADER_SIZE]

    encrypted_data = payload[
        PAYLOAD_HEADER_SIZE:
    ]

    print(
        f"Encrypted biometric size: "
        f"{len(encrypted_data)} bytes"
    )

    # The SHA-256 value is stored in the
    # FACEAUD1 header after magic/version/length.

    stored_hash = header[-32:]

    calculated_hash = hashlib.sha256(
        encrypted_data
    ).digest()

    print(
        "Stored SHA-256:",
        stored_hash.hex()
    )

    print(
        "Calculated SHA-256:",
        calculated_hash.hex()
    )

    if stored_hash != calculated_hash:

        print(
            "\nWARNING:"
        )

        print(
            "Payload hash verification FAILED."
        )

        return False

    print(
        "Payload hash verification PASSED."
    )

    return True


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_payload():

    print("\n")
    print("=" * 70)
    print("RECEIVER - PAYLOAD EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD AUDIO
    # --------------------------------------------------------

    audio, sr = load_stego_audio()

    # --------------------------------------------------------
    # 2. MFCC
    # --------------------------------------------------------

    mfcc = extract_mfcc(
        audio,
        sr
    )

    # --------------------------------------------------------
    # 3. CONTENT
    # --------------------------------------------------------

    content = extract_content_features(
        mfcc
    )

    # --------------------------------------------------------
    # 4. SYMBOLS
    # --------------------------------------------------------

    symbols = extract_symbols(
        content
    )

    # --------------------------------------------------------
    # 5. BITS
    # --------------------------------------------------------

    bits = symbols_to_bits(
        symbols
    )

    # --------------------------------------------------------
    # 6. BYTES
    # --------------------------------------------------------

    extracted_bytes = bits_to_bytes(
        bits[:TOTAL_PAYLOAD_SIZE * 8]
    )

    print("\n" + "-" * 60)
    print("EXTRACTED DATA")
    print("-" * 60)

    print(
        f"Extracted bytes: "
        f"{len(extracted_bytes)}"
    )

    # --------------------------------------------------------
    # 7. CFD1
    # --------------------------------------------------------

    parse_feature_header(
        extracted_bytes
    )

    # --------------------------------------------------------
    # 8. FACEAUD1
    # --------------------------------------------------------

    payload = extract_faceaud_payload(
        extracted_bytes
    )

    # --------------------------------------------------------
    # 9. VERIFY
    # --------------------------------------------------------

    verify_payload(
        payload
    )

    # --------------------------------------------------------
    # 10. SAVE
    # --------------------------------------------------------

    encrypted_biometric = payload[
        PAYLOAD_HEADER_SIZE:
    ]

    output_file = os.path.join(
        OUTPUT_DIR,
        "extracted_encrypted_biometric.bin"
    )

    with open(
        output_file,
        "wb"
    ) as f:

        f.write(
            encrypted_biometric
        )

    print("\n" + "-" * 60)
    print("EXTRACTION COMPLETED")
    print("-" * 60)

    print(
        "Encrypted biometric saved:"
    )

    print(output_file)

    print(
        f"Size: "
        f"{len(encrypted_biometric)} bytes"
    )

    print("\n" + "=" * 70)
    print("PAYLOAD EXTRACTION COMPLETED")
    print("=" * 70)

    return encrypted_biometric


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        extract_payload()

    except Exception as e:

        print("\n" + "=" * 70)
        print("PAYLOAD EXTRACTION FAILED")
        print("=" * 70)

        print(
            f"\nError: {e}"
        )

        raise