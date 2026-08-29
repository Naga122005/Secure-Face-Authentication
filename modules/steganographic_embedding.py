import numpy as np
from pathlib import Path
import sys


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))


# ==========================================
# FILE PATHS
# ==========================================

ENCRYPTED_FACE_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "encrypted_face_embedding.bin"
)


OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "embedded_face_data.npy"
)


# ==========================================
# LOAD ENCRYPTED FACE DATA
# ==========================================

def load_encrypted_face():

    if not ENCRYPTED_FACE_PATH.exists():

        print("\nERROR: Encrypted face data not found!")

        print("Expected location:")
        print(ENCRYPTED_FACE_PATH)

        return None

    with open(ENCRYPTED_FACE_PATH, "rb") as file:

        encrypted_data = file.read()

    print("\nEncrypted face data loaded successfully.")

    print("Encrypted data size:")
    print(len(encrypted_data), "bytes")

    return encrypted_data


# ==========================================
# CONVERT BYTES TO NUMBERS
# ==========================================

def bytes_to_array(data):

    # Convert bytes to unsigned 8-bit integers
    data_array = np.frombuffer(
        data,
        dtype=np.uint8
    ).astype(np.float32)

    # Convert values from 0-255 to approximately
    # -1 to +1
    data_array = (
        data_array / 127.5
    ) - 1.0

    return data_array


# ==========================================
# EMBEDDING FUNCTION
# ==========================================

def embed_data(
    content_features,
    encrypted_data,
    alpha=0.01
):
    """
    Embed encrypted face information
    into audio content features.

    alpha controls embedding strength.
    """

    if content_features is None:

        print("Content features are unavailable.")

        return None

    # Convert encrypted bytes
    secret_data = bytes_to_array(
        encrypted_data
    )

    # Flatten content features
    content = content_features.flatten()

    print("\nContent feature size:")
    print(len(content))

    print("Encrypted data size:")
    print(len(secret_data))

    # Check capacity
    if len(secret_data) > len(content):

        print("\nERROR: Content feature capacity is too small.")

        print(
            "Need:",
            len(secret_data),
            "values"
        )

        print(
            "Available:",
            len(content),
            "values"
        )

        return None

    # Create a copy
    modified_content = content.copy()

    # Embed secret information
    modified_content[
        :len(secret_data)
    ] = (
        modified_content[
            :len(secret_data)
        ]
        + alpha * secret_data
    )

    # Restore original shape
    modified_content = modified_content.reshape(
        content_features.shape
    )

    return modified_content


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("STEGANOGRAPHIC EMBEDDING STARTED")
    print("=" * 55)

    # --------------------------------------
    # Import feature decoupling
    # --------------------------------------

    from modules.feature_decoupling import (
        decouple_audio_features
    )

    # --------------------------------------
    # Get content and style
    # --------------------------------------

    content_features, style_features = (
        decouple_audio_features()
    )

    if content_features is None:

        print("\nContent extraction failed.")

        sys.exit()

    # --------------------------------------
    # Load encrypted face data
    # --------------------------------------

    encrypted_data = load_encrypted_face()

    if encrypted_data is None:

        sys.exit()

    # --------------------------------------
    # Embed encrypted face data
    # --------------------------------------

    modified_content = embed_data(
        content_features,
        encrypted_data,
        alpha=0.01
    )

    if modified_content is None:

        sys.exit()

    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\nEmbedding completed successfully!")

    print("\nOriginal content shape:")
    print(content_features.shape)

    print("\nModified content shape:")
    print(modified_content.shape)

    print("\nStyle feature shape:")
    print(style_features.shape)

    # --------------------------------------
    # Save modified content
    # --------------------------------------

    np.save(
        OUTPUT_PATH,
        modified_content
    )

    print("\nModified content saved at:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("STEGANOGRAPHIC EMBEDDING COMPLETED")
    print("=" * 55)