from pathlib import Path
import numpy as np

from .face_feature_extraction import extract_face_features
from .cancelable_biometrics import CancelableBiometrics


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FACE_PATH = BASE_DIR / "data" / "faces" / "person1.jpg"

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "faces"
    / "cancelable_template.npy"
)


# ============================================================
# CANCELABLE BIOMETRIC SECRET
# ============================================================

TRANSFORMATION_SEED = "PERSON1-CANCELABLE-KEY-001"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CANCELABLE BIOMETRIC PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1 — FACE IMAGE → FACENET EMBEDDING
    # --------------------------------------------------------

    print("\n[1] Extracting FaceNet embedding...")

    embedding = extract_face_features(FACE_PATH)

    if embedding is None:
        print("\nERROR: Face feature extraction failed.")
        return

    print("FaceNet embedding successfully extracted.")
    print("Embedding shape:", embedding.shape)
    print("Embedding dimension:", len(embedding))
    print("Embedding dtype:", embedding.dtype)

    if embedding.shape != (512,):
        print(
            "\nERROR: FaceNet did not produce "
            "a 512-dimensional embedding."
        )
        return

    print("✓ FaceNet produced a valid 512-D embedding.")

    # --------------------------------------------------------
    # STEP 2 — CREATE CANCELABLE BIOMETRIC OBJECT
    # --------------------------------------------------------

    print("\n[2] Initializing cancelable biometrics...")

    cancelable = CancelableBiometrics(dimension=512)

    print("Cancelable biometric method: Random Projection")

    # --------------------------------------------------------
    # STEP 3 — GENERATE CANCELABLE TEMPLATE
    # --------------------------------------------------------

    print("\n[3] Applying cancelable transformation...")

    cancelable_template = cancelable.transform(
        embedding,
        TRANSFORMATION_SEED
    )

    print("Cancelable transformation completed.")
    print("Cancelable template shape:", cancelable_template.shape)
    print(
        "Cancelable template dimension:",
        len(cancelable_template)
    )
    print("Cancelable template dtype:", cancelable_template.dtype)

    # --------------------------------------------------------
    # STEP 4 — DISPLAY SAMPLE VALUES
    # --------------------------------------------------------

    print("\nFirst 10 FaceNet embedding values:")
    print(embedding[:10])

    print("\nFirst 10 cancelable template values:")
    print(cancelable_template[:10])

    # --------------------------------------------------------
    # STEP 5 — SAVE CANCELABLE TEMPLATE
    # --------------------------------------------------------

    print("\n[4] Saving cancelable template...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        cancelable_template
    )

    print("Cancelable template saved successfully.")
    print("Saved location:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # STEP 6 — VERIFY SAVED TEMPLATE
    # --------------------------------------------------------

    print("\n[5] Verifying saved template...")

    loaded_template = np.load(OUTPUT_PATH)

    print("Loaded template shape:", loaded_template.shape)
    print("Loaded template dtype:", loaded_template.dtype)

    # --------------------------------------------------------
    # STEP 7 — VERIFY DATA INTEGRITY
    # --------------------------------------------------------

    identical = np.array_equal(
        cancelable_template,
        loaded_template
    )

    print(
        "Saved template matches original:",
        identical
    )

   
    # ========================================================
# STEP 8: CALCULATE COSINE SIMILARITY
# ========================================================

    vector_a = np.asarray(cancelable_template)
    vector_b = np.asarray(loaded_template)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        similarity = 0.0
    else:
        similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    print("\nCosine similarity:")
    print(similarity)

    print(
        "\nCosine similarity between "
        "original and loaded template:"
    )

    print(similarity)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CANCELABLE BIOMETRIC PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

