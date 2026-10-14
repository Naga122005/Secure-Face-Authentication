"""
CONTENT PAYLOAD ENCODER
=======================

Creates a compact payload containing the encrypted
cancelable biometric template.

Architecture:

Encrypted biometric payload
        ↓
Compact feature payload
        ↓
steganographic_embedding.py
        ↓
Modified content features
"""

from pathlib import Path
import hashlib
import struct
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_AUDIO = PROJECT_ROOT / "data" / "audio"

ENCRYPTED_PAYLOAD_FILE = (
    DATA_AUDIO / "encrypted_cancelable_template.bin"
)

OUTPUT_PAYLOAD_FILE = (
    DATA_AUDIO / "encoded_biometric_payload.bin"
)

OUTPUT_METADATA_FILE = (
    DATA_AUDIO / "payload_metadata.npy"
)

# Feature embedding will use 3 bits/value
BITS_PER_VALUE = 4

# Content feature count:
# 8 × 199 = 1592
CONTENT_VALUES = 8 * 199

CAPACITY_BITS = CONTENT_VALUES * BITS_PER_VALUE
CAPACITY_BYTES = CAPACITY_BITS // 8


# ============================================================
# COMPACT HEADER
# ============================================================
#
# 4 bytes : magic = CFD1
# 1 byte  : version
# 4 bytes : payload length
# 4 bytes : SHA-256 checksum prefix
#
# Total = 13 bytes
#
# The complete SHA-256 is NOT stored here because the
# encrypted AES-GCM payload already provides integrity.
#
# ============================================================

MAGIC = b"CFD1"
VERSION = 1
HEADER_SIZE = 13


def create_header(payload):
    """
    Create compact 13-byte feature payload header.
    """

    payload_length = len(payload)

    checksum = hashlib.sha256(payload).digest()

    header = (
        MAGIC
        + struct.pack("<B", VERSION)
        + struct.pack("<I", payload_length)
        + checksum[:4]
    )

    return header


def main():

    print("=" * 70)
    print("COMPACT CONTENT PAYLOAD ENCODER")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CHECKING ENCRYPTED BIOMETRIC PAYLOAD")
    print("-" * 60)

    if not ENCRYPTED_PAYLOAD_FILE.exists():
        print("ERROR:")
        print(f"Missing file: {ENCRYPTED_PAYLOAD_FILE}")
        return

    print(f"Found: {ENCRYPTED_PAYLOAD_FILE}")

    # --------------------------------------------------------
    # LOAD ENCRYPTED PAYLOAD
    # --------------------------------------------------------

    payload = ENCRYPTED_PAYLOAD_FILE.read_bytes()

    print("\nEncrypted biometric payload:")
    print(f"  Size: {len(payload)} bytes")

    # --------------------------------------------------------
    # CREATE HEADER
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CREATING COMPACT FEATURE HEADER")
    print("-" * 60)

    header = create_header(payload)

    print(f"Magic: {MAGIC}")
    print(f"Version: {VERSION}")
    print(f"Header size: {len(header)} bytes")

    # --------------------------------------------------------
    # COMPLETE FEATURE PAYLOAD
    # --------------------------------------------------------

    complete_payload = header + payload

    print("\nComplete feature payload:")
    print(f"  Header: {len(header)} bytes")
    print(f"  Encrypted data: {len(payload)} bytes")
    print(f"  Total: {len(complete_payload)} bytes")

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CHECKING FEATURE CAPACITY")
    print("-" * 60)

    print(f"Content values: {CONTENT_VALUES}")
    print(f"Bits per value: {BITS_PER_VALUE}")
    print(f"Capacity: {CAPACITY_BITS} bits")
    print(f"Capacity: {CAPACITY_BYTES} bytes")
    print(f"Required: {len(complete_payload)} bytes")

    if len(complete_payload) > CAPACITY_BYTES:
        print("\nERROR:")
        print("Payload does not fit into content features.")
        print(
            f"Required: {len(complete_payload)} bytes | "
            f"Available: {CAPACITY_BYTES} bytes"
        )
        return

    remaining = CAPACITY_BYTES - len(complete_payload)

    print(f"Remaining capacity: {remaining} bytes")

    # --------------------------------------------------------
    # SAVE PAYLOAD
    # --------------------------------------------------------

    OUTPUT_PAYLOAD_FILE.write_bytes(complete_payload)

    print("\n" + "-" * 60)
    print("SAVING FEATURE PAYLOAD")
    print("-" * 60)

    print(f"Saved:")
    print(OUTPUT_PAYLOAD_FILE)

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "header_size": HEADER_SIZE,
        "encrypted_payload_size": len(payload),
        "complete_payload_size": len(complete_payload),
        "bits_per_value": BITS_PER_VALUE,
        "capacity_bits": CAPACITY_BITS,
        "capacity_bytes": CAPACITY_BYTES,
        "remaining_bytes": remaining,
    }

    np.save(
        OUTPUT_METADATA_FILE,
        metadata,
        allow_pickle=True
    )

    print(f"\nMetadata saved:")
    print(OUTPUT_METADATA_FILE)

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    saved_payload = OUTPUT_PAYLOAD_FILE.read_bytes()

    print("\n" + "-" * 60)
    print("VERIFYING SAVED PAYLOAD")
    print("-" * 60)

    print(
        "Payload verification:",
        saved_payload == complete_payload
    )

    print("\n" + "=" * 70)
    print("CONTENT PAYLOAD ENCODING COMPLETED")
    print("=" * 70)

    print(f"Payload: {len(complete_payload)} bytes")
    print(f"Capacity: {CAPACITY_BYTES} bytes")
    print("Ready for content-domain embedding.")


if __name__ == "__main__":
    main()