import cv2
import torch
import numpy as np
import mysql.connector

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "naga*"
DB_NAME = "secure_face_authentication"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# MODELS
# ============================================================

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


# ============================================================
# DATABASE
# ============================================================

def connect_database():

    try:

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        print("\nDatabase connected successfully!")

        return connection

    except mysql.connector.Error as error:

        print("\nDatabase connection failed!")
        print("Error:", error)

        return None


# ============================================================
# CAPTURE FACE
# ============================================================

def capture_face():

    print("\n" + "=" * 60)
    print("FACE CAPTURE")
    print("=" * 60)

    print("\nOpening webcam...")
    print("Look at the camera.")
    print("Press SPACE to capture.")
    print("Press Q to cancel.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("\nERROR: Webcam could not be opened.")

        return None

    captured_face = None

    while True:

        ret, frame = camera.read()

        if not ret:

            print("\nERROR: Could not read webcam frame.")

            break

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Detect face
        # ----------------------------------------------------

        boxes, probabilities = mtcnn.detect(
            rgb_frame
        )

        # ----------------------------------------------------
        # Draw face box
        # ----------------------------------------------------

        if boxes is not None:

            # Only use first detected face
            box = boxes[0]

            x1, y1, x2, y2 = [
                int(v) for v in box
            ]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "FACE DETECTED",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "SPACE = CAPTURE",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = QUIT",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Secure Face Registration",
            frame
        )

        # ----------------------------------------------------
        # KEYBOARD INPUT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # SPACE → CAPTURE
        # ----------------------------------------------------

        if key == 32:

            print("\nSPACE pressed.")

            if boxes is None:

                print(
                    "No face detected. "
                    "Please try again."
                )

                continue

            print(
                "Face detected. Capturing..."
            )

            # Get first detected face
            box = boxes[0]

            x1, y1, x2, y2 = [
                int(v) for v in box
            ]

            # Add small margin
            margin = 20

            x1 = max(
                0,
                x1 - margin
            )

            y1 = max(
                0,
                y1 - margin
            )

            x2 = min(
                rgb_frame.shape[1],
                x2 + margin
            )

            y2 = min(
                rgb_frame.shape[0],
                y2 + margin
            )

            # ------------------------------------------------
            # CROP FACE
            # ------------------------------------------------

            captured_face = rgb_frame[
                y1:y2,
                x1:x2
            ].copy()

            if captured_face.size == 0:

                print(
                    "ERROR: Face crop is empty."
                )

                captured_face = None

                continue

            print(
                "Face captured successfully!"
            )

            print(
                "Captured face shape:",
                captured_face.shape
            )

            # ------------------------------------------------
            # SAVE CAPTURED FACE IMAGE
            # ------------------------------------------------

            save_path = (
                "data/faces/captured_face.jpg"
            )

            # RGB → BGR for OpenCV
            captured_bgr = cv2.cvtColor(
                captured_face,
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                save_path,
                captured_bgr
            )

            print(
                "Captured face saved at:"
            )

            print(save_path)

            break

        # ----------------------------------------------------
        # Q → QUIT
        # ----------------------------------------------------

        elif key == ord("q"):

            print(
                "\nFace capture cancelled."
            )

            break

    camera.release()

    cv2.destroyAllWindows()

    return captured_face


# ============================================================
# FACE FEATURE EXTRACTION
# ============================================================

def extract_face_features(face_image):

    print("\n" + "=" * 60)
    print("FACIAL FEATURE EXTRACTION")
    print("=" * 60)

    if face_image is None:

        print(
            "ERROR: Face image is empty."
        )

        return None

    # Convert NumPy → PIL
    pil_image = Image.fromarray(
        face_image
    )

    # FaceNet expects aligned face
    face_tensor = mtcnn(
        pil_image
    )

    if face_tensor is None:

        print(
            "\nERROR: Face could not be processed."
        )

        return None

    # Add batch dimension
    face_tensor = (
        face_tensor
        .unsqueeze(0)
        .to(device)
    )

    print(
        "\nGenerating facial embedding..."
    )

    with torch.no_grad():

        embedding = facenet(
            face_tensor
        )

    # Normalize
    embedding = embedding / embedding.norm(
        p=2,
        dim=1,
        keepdim=True
    )

    embedding = (
        embedding
        .cpu()
        .numpy()[0]
    )

    print(
        "\nFacial feature extraction completed."
    )

    print(
        "Feature shape:",
        embedding.shape
    )

    print(
        "Feature dimension:",
        embedding.size
    )

    return embedding


# ============================================================
# STORE FACE FEATURES
# ============================================================

def store_features(
    username,
    face_features
):

    print("\n" + "=" * 60)
    print("DATABASE STORAGE")
    print("=" * 60)

    connection = connect_database()

    if connection is None:

        return False

    try:

        cursor = connection.cursor()

        # Convert NumPy array → bytes
        feature_bytes = (
            face_features
            .astype(np.float32)
            .tobytes()
        )

        query = """
        INSERT INTO users
        (username, face_features)
        VALUES (%s, %s)
        """

        cursor.execute(
            query,
            (
                username,
                feature_bytes
            )
        )

        connection.commit()

        print(
            "\nFace feature stored successfully!"
        )

        print(
            "Username:",
            username
        )

        print(
            "Feature size:",
            len(feature_bytes),
            "bytes"
        )

        cursor.close()
        connection.close()

        return True

    except mysql.connector.Error as error:

        print(
            "\nDatabase storage failed."
        )

        print(
            "Error:",
            error
        )

        connection.close()

        return False


# ============================================================
# MAIN REGISTRATION
# ============================================================

def register_user():

    print("\n")
    print("=" * 60)
    print("SECURE FACE AUTHENTICATION")
    print("USER REGISTRATION")
    print("=" * 60)

    username = input(
        "\nEnter username: "
    ).strip()

    if username == "":

        print(
            "\nERROR: Username cannot be empty."
        )

        return

    # --------------------------------------------------------
    # CAPTURE FACE
    # --------------------------------------------------------

    face_image = capture_face()

    if face_image is None:

        print(
            "\nRegistration failed."
        )

        return

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    face_features = extract_face_features(
        face_image
    )

    if face_features is None:

        print(
            "\nRegistration failed."
        )

        return

    # --------------------------------------------------------
    # STORE FEATURES
    # --------------------------------------------------------

    success = store_features(
        username,
        face_features
    )

    if success:

        print("\n")
        print("=" * 60)
        print("REGISTRATION SUCCESSFUL")
        print("=" * 60)

        print(
            "\nUser:",
            username
        )

        print(
            "Face feature dimension:",
            face_features.shape
        )

    else:

        print(
            "\nREGISTRATION FAILED"
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    register_user()