import torch
import numpy as np
from pathlib import Path
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

FACE_PATH = BASE_DIR / "data" / "faces" / "person1.jpg"


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# FACE DETECTOR
# ==========================================

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device
)


# ==========================================
# FACE FEATURE MODEL
# ==========================================

face_model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


# ==========================================
# FACE FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_face_features(image_path):

    # Check whether image exists
    if not image_path.exists():

        print("\nERROR: Face image not found!")
        print("Expected location:")
        print(image_path)

        return None

    try:

        # ----------------------------------
        # Load image
        # ----------------------------------

        print("\nLoading face image...")
        print(image_path)

        image = Image.open(image_path).convert("RGB")

        print("Image loaded successfully.")

        # ----------------------------------
        # Detect face
        # ----------------------------------

        face = mtcnn(image)

        if face is None:

            print("\nERROR: No face detected in the image.")

            return None

        print("Face detected successfully.")

        # ----------------------------------
        # Add batch dimension
        # ----------------------------------

        face = face.unsqueeze(0)

        face = face.to(device)

        # ----------------------------------
        # Generate face embedding
        # ----------------------------------

        with torch.no_grad():

            embedding = face_model(face)

        # ----------------------------------
        # Convert tensor to NumPy
        # ----------------------------------

        embedding = embedding.cpu().numpy()[0]

        # ----------------------------------
        # Normalize embedding
        # ----------------------------------

        norm = np.linalg.norm(embedding)

        if norm != 0:

            embedding = embedding / norm

        return embedding

    except Exception as error:

        print("\nERROR while processing face image:")
        print(error)

        return None


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("FACE FEATURE EXTRACTION STARTED")
    print("=" * 50)

    # Extract face features
    face_embedding = extract_face_features(FACE_PATH)

    if face_embedding is not None:

        print("\nFace feature extraction successful!")

        # ----------------------------------
        # Display embedding information
        # ----------------------------------

        print("\nEmbedding shape:")
        print(face_embedding.shape)

        print("\nEmbedding dimension:")
        print(len(face_embedding))

        print("\nFirst 10 embedding values:")
        print(face_embedding[:10])

        # ----------------------------------
        # Save embedding
        # ----------------------------------

        output_path = (
            BASE_DIR
            / "data"
            / "faces"
            / "face_embedding.npy"
        )

        np.save(output_path, face_embedding)

        print("\nFace embedding saved successfully!")

        print("Saved location:")
        print(output_path)

    else:

        print("\nFace feature extraction failed.")

    print("\n" + "=" * 50)
    print("FACE FEATURE EXTRACTION COMPLETED")
    print("=" * 50)