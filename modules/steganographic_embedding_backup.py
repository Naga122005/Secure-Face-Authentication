"""
Robust Content-Domain Steganographic Embedding
==============================================

Encrypted biometric payload
            ↓
       Feature header
            ↓
        Bit stream
            ↓
   Content feature embedding
            ↓
modified_content_features.npy

The payload is embedded only into content features.
Style features are not modified.
"""

import os
import struct
import hashlib
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "audio"
)

CONTENT_PATH = os.path.join(
    AUDIO_DIR,
    "content_features.npy"
)

PAYLOAD_PATH = os.path.join(
    AUDIO_DIR,
    "encoded_biometric_payload.bin"
)

OUTPUT_CONTENT_PATH = os.path.join(
    AUDIO_DIR,
    "modified_content_features.npy"
)

METADATA_PATH = os.path.join(
    AUDIO_DIR,
    "embedding_metadata.npy"
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_DOMAIN = "content_features"

BITS_PER_VALUE = 4

QUANTIZATION_LEVELS = 16

# Larger quantization cell than the previous implementation.
#
# IMPORTANT:
# This value must be large enough that normal MFCC reconstruction
# errors do not immediately change the stored symbol.
#
QUANTIZATION_STEP = 16.0


# ============================================================
# FEATURE PAYLOAD HEADER
# ============================================================

FEATURE_MAGIC = b"CFD1"

FEATURE_VERSION = 1

FEATURE_HEADER_SIZE = 41


# ============================================================
# LOAD PAYLOAD
# ============================================================

def load_payload():

    print("\n" + "-" * 60)
    print("LOADING ENCRYPTED BIOMETRIC PAYLOAD")
    print("-" * 60)

    if not os.path.exists(PAYLOAD_PATH):

        raise FileNotFoundError(
            "Encrypted biometric payload not found:\n"
            + PAYLOAD_PATH
        )

    with open(
        PAYLOAD_PATH,
        "rb"
    ) as file:

        payload = file.read()

    print(
        "Encrypted biometric payload:",
        len(payload),
        "bytes"
    )

    return payload


# ============================================================
# CREATE FEATURE PAYLOAD
# ============================================================

def create_feature_payload(payload):

    print("\n" + "-" * 60)
    print("CREATING CONTENT FEATURE PAYLOAD")
    print("-" * 60)

    payload_length = len(payload)

    payload_hash = hashlib.sha256(
        payload
    ).digest()

    header = (
        FEATURE_MAGIC
        + struct.pack(
            ">B",
            FEATURE_VERSION
        )
        + struct.pack(
            ">I",
            payload_length
        )
        + payload_hash
    )

    if len(header) != FEATURE_HEADER_SIZE:

        raise ValueError(
            "Feature header size is incorrect."
        )

    complete_payload = (
        header
        + payload
    )

    print(
        "Feature magic:",
        FEATURE_MAGIC
    )

    print(
        "Feature version:",
        FEATURE_VERSION
    )

    print(
        "Feature header:",
        len(header),
        "bytes"
    )

    print(
        "Encrypted biometric:",
        payload_length,
        "bytes"
    )

    print(
        "Complete feature payload:",
        len(complete_payload),
        "bytes"
    )

    return complete_payload


# ============================================================
# BYTES → BITS
# ============================================================

def bytes_to_bits(data):

    return np.unpackbits(
        np.frombuffer(
            data,
            dtype=np.uint8
        )
    ).astype(np.uint8)


# ============================================================
# BITS → BYTES
# ============================================================

def bits_to_bytes(bits):

    usable_bits = (
        len(bits) // 8
    ) * 8

    if usable_bits == 0:
        return b""

    return np.packbits(
        bits[:usable_bits]
    ).tobytes()


# ============================================================
# CAPACITY
# ============================================================

def calculate_capacity(content):

    content_values = content.size

    capacity_bits = (
        content_values
        * BITS_PER_VALUE
    )

    capacity_bytes = (
        capacity_bits // 8
    )

    return (
        content_values,
        capacity_bits,
        capacity_bytes
    )


# ============================================================
# EMBED PAYLOAD
# ============================================================

def embed_payload(
    content,
    bits
):

    print("\n" + "-" * 60)
    print("EMBEDDING PAYLOAD INTO CONTENT FEATURES")
    print("-" * 60)

    original_shape = content.shape

    flat = (
        content
        .astype(np.float64)
        .flatten()
    )

    capacity_bits = (
        len(flat)
        * BITS_PER_VALUE
    )

    if len(bits) > capacity_bits:

        raise ValueError(
            "Insufficient content-feature capacity.\n"
            f"Available: {capacity_bits} bits\n"
            f"Required: {len(bits)} bits"
        )

    modified = flat.copy()

    number_of_groups = int(
        np.ceil(
            len(bits)
            / BITS_PER_VALUE
        )
    )

    print(
        "Content values:",
        len(flat)
    )

    print(
        "Embedding groups:",
        number_of_groups
    )

    print(
        "Bits per value:",
        BITS_PER_VALUE
    )

    print(
        "Quantization levels:",
        QUANTIZATION_LEVELS
    )

    print(
        "Quantization step:",
        QUANTIZATION_STEP
    )

    # --------------------------------------------------------
    # Embed each 4-bit symbol
    # --------------------------------------------------------

    for i in range(number_of_groups):

        start = (
            i
            * BITS_PER_VALUE
        )

        end = min(
            start + BITS_PER_VALUE,
            len(bits)
        )

        group = bits[start:end]

        symbol = 0

        for bit in group:

            symbol = (
                symbol << 1
            ) | int(bit)

        # Pad final symbol with zeros
        actual_bits = end - start

        while actual_bits < BITS_PER_VALUE:

            symbol = (
                symbol << 1
            )

            actual_bits += 1

        # ----------------------------------------------------
        # Instead of using a very small 4-unit region,
        # use a larger quantization interval.
        # ----------------------------------------------------

        original_value = flat[i]

        original_index = int(
            np.floor(
                original_value
                / QUANTIZATION_STEP
            )
        )

        # Keep the coefficient around its original
        # large-scale region.
        #
        # We use the symbol as the low-order quantized
        # component.
        base_index = (
            original_index
            // QUANTIZATION_LEVELS
        ) * QUANTIZATION_LEVELS

        target_index = (
            base_index
            + symbol
        )

        # Put value near the center of the cell.
        modified[i] = (
            target_index
            * QUANTIZATION_STEP
            + QUANTIZATION_STEP * 0.5
        )

    modified_content = (
        modified
        .astype(np.float32)
        .reshape(original_shape)
    )

    return modified_content


# ============================================================
# EXTRACT BITS
# ============================================================

def extract_bits_from_content(
    content,
    number_of_bits
):

    flat = (
        content
        .astype(np.float64)
        .flatten()
    )

    required_values = int(
        np.ceil(
            number_of_bits
            / BITS_PER_VALUE
        )
    )

    if required_values > len(flat):

        raise ValueError(
            "Not enough content values."
        )

    extracted_bits = []

    for i in range(required_values):

        value = flat[i]

        quantized_index = int(
            np.floor(
                value
                / QUANTIZATION_STEP
            )
        )

        symbol = (
            quantized_index
            % QUANTIZATION_LEVELS
        )

        for shift in range(
            BITS_PER_VALUE - 1,
            -1,
            -1
        ):

            bit = (
                symbol
                >> shift
            ) & 1

            extracted_bits.append(bit)

    return np.array(
        extracted_bits[:number_of_bits],
        dtype=np.uint8
    )


# ============================================================
# VERIFY
# ============================================================

def verify_payload(
    extracted_payload,
    original_payload
):

    print("\n" + "-" * 60)
    print("VERIFYING CONTENT-DOMAIN EMBEDDING")
    print("-" * 60)

    print(
        "Original payload:",
        len(original_payload),
        "bytes"
    )

    print(
        "Extracted payload:",
        len(extracted_payload),
        "bytes"
    )

    exact_match = (
        extracted_payload
        == original_payload
    )

    print(
        "Payload recovered:",
        exact_match
    )

    if exact_match:

        print(
            "Internal content-domain "
            "verification: PASS"
        )

    else:

        limit = min(
            len(extracted_payload),
            len(original_payload)
        )

        mismatch_index = None

        for i in range(limit):

            if (
                extracted_payload[i]
                != original_payload[i]
            ):

                mismatch_index = i
                break

        if mismatch_index is not None:

            print(
                "First mismatch byte:",
                mismatch_index
            )

            print(
                "Original:",
                original_payload[
                    mismatch_index
                ]
            )

            print(
                "Extracted:",
                extracted_payload[
                    mismatch_index
                ]
            )

        print(
            "Internal content-domain "
            "verification: FAIL"
        )

    return exact_match


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    content,
    encrypted_payload,
    feature_payload,
    bits,
    internal_match
):

    (
        content_values,
        capacity_bits,
        capacity_bytes
    ) = calculate_capacity(content)

    metadata = {

        "embedding_domain":
            EMBEDDING_DOMAIN,

        "content_shape":
            content.shape,

        "content_values":
            content_values,

        "bits_per_value":
            BITS_PER_VALUE,

        "quantization_levels":
            QUANTIZATION_LEVELS,

        "quantization_step":
            QUANTIZATION_STEP,

        "capacity_bits":
            capacity_bits,

        "capacity_bytes":
            capacity_bytes,

        "encrypted_biometric_size_bytes":
            len(encrypted_payload),

        "feature_header_size_bytes":
            FEATURE_HEADER_SIZE,

        "feature_payload_size_bytes":
            len(feature_payload),

        "payload_bits":
            len(bits),

        "internal_verification":
            internal_match,

        "style_modified":
            False,

        "description":
            (
                "Encrypted cancelable biometric "
                "payload embedded into content "
                "features using quantized symbols."
            )
    }

    np.save(
        METADATA_PATH,
        metadata,
        allow_pickle=True
    )

    print(
        "\nEmbedding metadata saved:"
    )

    print(
        METADATA_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "ROBUST CONTENT-DOMAIN "
        "STEGANOGRAPHIC EMBEDDING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD CONTENT
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("LOADING CONTENT FEATURES")
    print("-" * 60)

    if not os.path.exists(CONTENT_PATH):

        raise FileNotFoundError(
            "Content features not found:\n"
            + CONTENT_PATH
        )

    content = np.load(
        CONTENT_PATH
    ).astype(np.float32)

    print(
        "Content shape:",
        content.shape
    )

    # --------------------------------------------------------
    # LOAD PAYLOAD
    # --------------------------------------------------------

    encrypted_payload = (
        load_payload()
    )

    # --------------------------------------------------------
    # CREATE FEATURE PAYLOAD
    # --------------------------------------------------------

    feature_payload = (
        create_feature_payload(
            encrypted_payload
        )
    )

    # --------------------------------------------------------
    # BITS
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("GENERATING PAYLOAD BITSTREAM")
    print("-" * 60)

    bits = bytes_to_bits(
        feature_payload
    )

    print(
        "Payload bytes:",
        len(feature_payload)
    )

    print(
        "Payload bits:",
        len(bits)
    )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    (
        content_values,
        capacity_bits,
        capacity_bytes
    ) = calculate_capacity(
        content
    )

    print("\n" + "-" * 60)
    print("CAPACITY CHECK")
    print("-" * 60)

    print(
        "Content values:",
        content_values
    )

    print(
        "Capacity:",
        capacity_bits,
        "bits"
    )

    print(
        "Capacity:",
        capacity_bytes,
        "bytes"
    )

    print(
        "Required:",
        len(feature_payload),
        "bytes"
    )

    if len(bits) > capacity_bits:

        raise ValueError(
            "Payload does not fit."
        )

    print(
        "Capacity sufficient: True"
    )

    # --------------------------------------------------------
    # EMBED
    # --------------------------------------------------------

    modified_content = embed_payload(
        content,
        bits
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.save(
        OUTPUT_CONTENT_PATH,
        modified_content
    )

    print("\n" + "-" * 60)
    print("SAVING MODIFIED CONTENT")
    print("-" * 60)

    print(
        "Modified content saved:"
    )

    print(
        OUTPUT_CONTENT_PATH
    )

    # --------------------------------------------------------
    # INTERNAL TEST
    # --------------------------------------------------------

    extracted_bits = (
        extract_bits_from_content(
            modified_content,
            len(bits)
        )
    )

    extracted_payload = (
        bits_to_bytes(
            extracted_bits
        )
    )

    internal_match = verify_payload(
        extracted_payload,
        feature_payload
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    save_metadata(
        content,
        encrypted_payload,
        feature_payload,
        bits,
        internal_match
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTENT-DOMAIN EMBEDDING COMPLETED")
    print("=" * 70)

    print(
        "Modified content:",
        OUTPUT_CONTENT_PATH
    )

    print(
        "Internal payload verification:",
        internal_match
    )

    print(
        "Quantization step:",
        QUANTIZATION_STEP
    )


if __name__ == "__main__":

    main()