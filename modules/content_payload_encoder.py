import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

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
    / "embedded_content.npy"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# CONTENT ENCODER
# ============================================================

class ContentEncoder(nn.Module):

    def __init__(self, input_size, latent_size):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_size),
            nn.Tanh()
        )

    def forward(self, x):

        return self.encoder(x)


# ============================================================
# PAYLOAD ENCODER
# ============================================================

class PayloadEncoder(nn.Module):

    def __init__(self, input_size, latent_size):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_size),
            nn.Tanh()
        )

    def forward(self, x):

        return self.encoder(x)


# ============================================================
# LOAD ENCRYPTED FACE DATA
# ============================================================

def load_encrypted_face():

    if not ENCRYPTED_FACE_PATH.exists():

        print("\nERROR: Encrypted face file not found!")
        print("Expected location:")
        print(ENCRYPTED_FACE_PATH)

        return None

    with open(
        ENCRYPTED_FACE_PATH,
        "rb"
    ) as file:

        encrypted_data = file.read()

    print("\nEncrypted face data loaded successfully.")
    print(
        "Encrypted data size:",
        len(encrypted_data),
        "bytes"
    )

    return encrypted_data


# ============================================================
# CONVERT ENCRYPTED DATA TO ARRAY
# ============================================================

def encrypted_data_to_array(encrypted_data):

    data = np.frombuffer(
        encrypted_data,
        dtype=np.uint8
    ).astype(np.float32)

    data = data / 255.0

    return data


# ============================================================
# LOAD AUDIO CONTENT FEATURES
# ============================================================

def load_content_features():

    from modules.feature_decoupling import (
        decouple_audio_features
    )

    content_features, style_features = (
        decouple_audio_features()
    )

    return content_features, style_features


# ============================================================
# CREATE CONTENT LATENT
# ============================================================

def create_content_latent(content_features):

    content_vector = content_features.flatten()

    input_size = len(content_vector)

    latent_size = 512

    model = ContentEncoder(
        input_size=input_size,
        latent_size=latent_size
    ).to(device)

    content_tensor = torch.tensor(
        content_vector,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        content_latent = model(
            content_tensor
        )

    return content_latent


# ============================================================
# CREATE PAYLOAD LATENT
# ============================================================

def create_payload_latent(encrypted_data):

    payload = encrypted_data_to_array(
        encrypted_data
    )

    input_size = len(payload)

    latent_size = 512

    model = PayloadEncoder(
        input_size=input_size,
        latent_size=latent_size
    ).to(device)

    payload_tensor = torch.tensor(
        payload,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        payload_latent = model(
            payload_tensor
        )

    return payload_latent


# ============================================================
# FUSE CONTENT AND PAYLOAD
# ============================================================

def fuse_features(
    content_latent,
    payload_latent
):

    combined = torch.cat(
        (
            content_latent,
            payload_latent
        ),
        dim=1
    )

    return combined


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONTENT + PAYLOAD ENCODER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load audio features
    # --------------------------------------------------------

    content_features, style_features = (
        load_content_features()
    )

    if content_features is None:

        print("\nContent features unavailable.")
        exit()

    print("\nOriginal content shape:")
    print(content_features.shape)

    print("\nStyle feature shape:")
    print(style_features.shape)

    # --------------------------------------------------------
    # Load encrypted face
    # --------------------------------------------------------

    encrypted_data = load_encrypted_face()

    if encrypted_data is None:

        exit()

    # --------------------------------------------------------
    # Content encoder
    # --------------------------------------------------------

    print("\nCreating content latent...")

    content_latent = create_content_latent(
        content_features
    )

    print(
        "Content latent shape:",
        content_latent.shape
    )

    # --------------------------------------------------------
    # Payload encoder
    # --------------------------------------------------------

    print("\nCreating payload latent...")

    payload_latent = create_payload_latent(
        encrypted_data
    )

    print(
        "Payload latent shape:",
        payload_latent.shape
    )

    # --------------------------------------------------------
    # Fuse
    # --------------------------------------------------------

    print("\nCombining content and payload...")

    combined_features = fuse_features(
        content_latent,
        payload_latent
    )

    print(
        "Combined feature shape:",
        combined_features.shape
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined_numpy = (
        combined_features
        .cpu()
        .numpy()
    )

    np.save(
        OUTPUT_PATH,
        combined_numpy
    )

    print("\nCombined latent representation saved.")

    print("Saved at:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETED")
    print("=" * 60)