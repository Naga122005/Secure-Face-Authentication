import os
import numpy as np

from modules.face_feature_extraction import (
    extract_face_features
)

from modules.cancelable_biometrics import (
    CancelableBiometrics
)

from modules.database_audio_loader import (
    AudioDatabase
)


class FaceRegistration:

    def __init__(self):

        self.cancelable = (
            CancelableBiometrics(512)
        )

        self.database = AudioDatabase()

    def register(
        self,
        username,
        face_image,
        stego_audio_path
    ):

        print("\n==============================")
        print("FACE REGISTRATION")
        print("==============================")

        # ----------------------------------------
        # STEP 1: FACE FEATURE EXTRACTION
        # ----------------------------------------

        face_embedding = (
            extract_face_features(
                face_image
            )
        )

        face_embedding = np.asarray(
            face_embedding,
            dtype=np.float32
        ).flatten()

        print(
            "Face embedding shape:",
            face_embedding.shape
        )

        # ----------------------------------------
        # STEP 2: CANCELABLE BIOMETRICS
        # ----------------------------------------

        cancelable_template = (
            self.cancelable.transform(
                face_embedding,
                username
            )
        )

        print(
            "Cancelable template shape:",
            cancelable_template.shape
        )

        # ----------------------------------------
        # STEP 3:
        # ENCRYPT CANCELABLE TEMPLATE
        # ----------------------------------------

        # IMPORTANT:
        # Call your EXISTING ML-KEM + AES
        # encryption function here.

        encrypted_template = (
            self.encrypt_template(
                cancelable_template,
                username
            )
        )

        print(
            "Encrypted template generated."
        )

        # ----------------------------------------
        # STEP 4:
        # HIFI STEGO EMBEDDING
        # ----------------------------------------

        self.create_stego_audio(
            encrypted_template,
            stego_audio_path
        )

        print(
            "High-Fidelity Stego Audio created."
        )

        # ----------------------------------------
        # STEP 5:
        # STORE STEGO AUDIO ONLY
        # ----------------------------------------

        self.database.store_stego_audio(
            username,
            stego_audio_path
        )

        print("\nRegistration completed.")
        print(
            "Database stores: HIGH-FIDELITY STEGO AUDIO"
        )

    # --------------------------------------------
    # ENCRYPTION
    # --------------------------------------------

    def encrypt_template(
        self,
        template,
        username
    ):

        """
        Connect this function to your existing
        ML-KEM + AES-256 implementation.
        """

        template_bytes = template.tobytes()

        return template_bytes

    # --------------------------------------------
    # HIFI STEGO
    # --------------------------------------------

    def create_stego_audio(
        self,
        encrypted_template,
        output_path
    ):

        """
        Connect this function to your existing
        HIFI-Stego embedding pipeline.

        encrypted_template must be embedded
        into the content representation.
        """

        raise NotImplementedError(
            "Connect your existing "
            "feature coupling / payload encoder "
            "code here."
        )