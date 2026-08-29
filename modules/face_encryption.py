import numpy as np
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "face_embedding.npy"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "encrypted_face_embedding.bin"
)

KEY_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "encryption_key.bin"
)


# ==========================================
# LOAD FACE EMBEDDING
# ==========================================

def load_face_embedding():

    if not EMBEDDING_PATH.exists():

        print("\nERROR: Face embedding not found!")
        print("Expected location:")
        print(EMBEDDING_PATH)

        return None

    embedding = np.load(EMBEDDING_PATH)

    print("\nFace embedding loaded successfully.")
    print("Embedding shape:", embedding.shape)

    return embedding


# ==========================================
# GENERATE AES-256 KEY
# ==========================================

def generate_key():

    key = AESGCM.generate_key(bit_length=256)

    with open(KEY_PATH, "wb") as file:
        file.write(key)

    print("\nAES-256 encryption key generated.")
    print("Key saved at:")
    print(KEY_PATH)

    return key


# ==========================================
# ENCRYPT FACE EMBEDDING
# ==========================================

def encrypt_embedding(embedding, key):

    # Convert NumPy array into bytes
    embedding_bytes = embedding.astype(
        np.float32
    ).tobytes()

    # Create AES-GCM object
    aes = AESGCM(key)

    # Generate random nonce
    nonce = secrets.token_bytes(12)

    # Encrypt
    encrypted_data = aes.encrypt(
        nonce,
        embedding_bytes,
        None
    )

    # Store nonce + encrypted data
    final_data = nonce + encrypted_data

    with open(OUTPUT_PATH, "wb") as file:
        file.write(final_data)

    print("\nFace embedding encrypted successfully.")

    print("Encrypted data saved at:")
    print(OUTPUT_PATH)

    print("\nEncrypted data size:")
    print(len(final_data), "bytes")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("FACE EMBEDDING ENCRYPTION")
    print("=" * 50)

    # Load embedding
    embedding = load_face_embedding()

    if embedding is None:
        exit()

    # Generate encryption key
    key = generate_key()

    # Encrypt embedding
    encrypt_embedding(
        embedding,
        key
    )

    print("\n" + "=" * 50)
    print("ENCRYPTION COMPLETED")
    print("=" * 50)