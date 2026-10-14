import os
import struct
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "faces",
    "cancelable_template.npy"
)

KEY_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "faces",
    "encryption_key.bin"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "faces",
    "encrypted_cancelable_template.bin"
)


# ---------------------------------------------------------
# COMPACT QUANTIZATION
# ---------------------------------------------------------

def quantize_template(template):
    """
    Convert 512-D float32 cancelable template
    into compact int8 representation.

    The scale is stored along with the quantized data.
    """

    max_abs = np.max(np.abs(template))

    if max_abs == 0:
        raise ValueError("Template contains only zero values.")

    scale = max_abs / 127.0

    quantized = np.round(
        template / scale
    ).astype(np.int8)

    return quantized, np.float32(scale)


# ---------------------------------------------------------
# MAIN ENCRYPTION
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AES-256-GCM ENCRYPTION")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. LOAD CANCELABLE TEMPLATE
    # -----------------------------------------------------

    print("\n[1] Loading cancelable biometric template...")

    template = np.load(TEMPLATE_PATH)

    print("Template shape:", template.shape)
    print("Template dtype:", template.dtype)

    if template.shape != (512,):
        raise ValueError(
            "Expected cancelable template shape (512,)"
        )

    template = template.astype(np.float32)

    print("Valid 512-D cancelable template.")

    # -----------------------------------------------------
    # 2. QUANTIZE
    # -----------------------------------------------------

    print("\n[2] Compacting template using INT8 quantization...")

    quantized, scale = quantize_template(template)

    print("Original size :", template.nbytes, "bytes")
    print("Quantized size:", quantized.nbytes, "bytes")
    print("Scale         :", scale)

    # Verify reconstruction
    reconstructed = (
        quantized.astype(np.float32) * scale
    )

    cosine_similarity = np.dot(
        template,
        reconstructed
    ) / (
        np.linalg.norm(template) *
        np.linalg.norm(reconstructed)
    )

    print(
        "Quantization cosine similarity:",
        cosine_similarity
    )

    # -----------------------------------------------------
    # 3. CREATE COMPACT PLAINTEXT
    # -----------------------------------------------------

    print("\n[3] Creating compact biometric data...")

    # Format:
    #
    # 4 bytes  : MAGIC = CBQ1
    # 1 byte   : VERSION
    # 4 bytes  : number of template values
    # 4 bytes  : quantization scale (float32)
    # 512 bytes: INT8 template
    #

    MAGIC = b"CBQ1"
    VERSION = 1

    header = (
        MAGIC +
        struct.pack(">B", VERSION) +
        struct.pack(">I", len(quantized)) +
        struct.pack(">f", float(scale))
    )

    compact_data = (
        header +
        quantized.tobytes()
    )

    print("Compact header size:", len(header), "bytes")
    print("Compact biometric size:", len(compact_data), "bytes")

    # -----------------------------------------------------
    # 4. GENERATE AES-256 KEY
    # -----------------------------------------------------

    print("\n[4] Generating AES-256 key...")

    key = AESGCM.generate_key(bit_length=256)

    with open(KEY_PATH, "wb") as f:
        f.write(key)

    print("AES key saved:")
    print(KEY_PATH)

    # -----------------------------------------------------
    # 5. AES-256-GCM ENCRYPTION
    # -----------------------------------------------------

    print("\n[5] Encrypting compact cancelable template...")

    aesgcm = AESGCM(key)

    nonce = os.urandom(12)

    encrypted_data = aesgcm.encrypt(
        nonce,
        compact_data,
        None
    )

    # Store nonce + ciphertext + authentication tag
    final_data = nonce + encrypted_data

    with open(OUTPUT_PATH, "wb") as f:
        f.write(final_data)

    print("Encrypted data size:", len(final_data), "bytes")

    # -----------------------------------------------------
    # 6. FINAL INFORMATION
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ENCRYPTION COMPLETED")
    print("=" * 60)

    print("\nBiometric representation:")
    print("512-D Cancelable Template")

    print("\nCompression:")
    print("Float32 → INT8")

    print("\nOriginal template:")
    print(template.nbytes, "bytes")

    print("\nCompact template:")
    print(len(compact_data), "bytes")

    print("\nEncrypted payload:")
    print(len(final_data), "bytes")

    print("\nSaved at:")
    print(OUTPUT_PATH)

    print("\nAES key:")
    print(KEY_PATH)

    print("\nReady for content-domain steganographic embedding.")


if __name__ == "__main__":
    main()