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


class FaceAuthentication:

    def __init__(self):

        self.cancelable = (
            CancelableBiometrics(512)
        )

        self.database = AudioDatabase()

    def authenticate(
        self,
        username,
        new_face_image
    ):

        print("\n==============================")
        print("FACE AUTHENTICATION")
        print("==============================")

        # ========================================
        # STEP 1
        # EXTRACT NEW FACE FEATURES
        # ========================================

        new_embedding = (
            extract_face_features(
                new_face_image
            )
        )

        new_embedding = np.asarray(
            new_embedding,
            dtype=np.float32
        ).flatten()

        print(
            "New face embedding shape:",
            new_embedding.shape
        )

        # ========================================
        # STEP 2
        # CREATE QUERY CANCELABLE TEMPLATE
        # ========================================

        query_template = (
            self.cancelable.transform(
                new_embedding,
                username
            )
        )

        print(
            "Query cancelable template:",
            query_template.shape
        )

        # ========================================
        # STEP 3
        # RETRIEVE STEGO AUDIO
        # ========================================

        retrieved_audio = (
            "data/audio/retrieved_stego_audio.wav"
        )

        self.database.retrieve_stego_audio(
            username,
            retrieved_audio
        )

        # ========================================
        # STEP 4
        # EXTRACT ENCRYPTED PAYLOAD
        # ========================================

        encrypted_template = (
            self.extract_payload(
                retrieved_audio
            )
        )

        print(
            "Encrypted payload extracted."
        )

        # ========================================
        # STEP 5
        # DECRYPT
        # ========================================

        recovered_template = (
            self.decrypt_template(
                encrypted_template,
                username
            )
        )

        recovered_template = np.asarray(
            recovered_template,
            dtype=np.float32
        ).flatten()

        print(
            "Recovered template shape:",
            recovered_template.shape
        )

        # ========================================
        # STEP 6
        # COMPARE
        # ========================================

        similarity = (
            self.cosine_similarity(
                query_template,
                recovered_template
            )
        )

        print(
            f"Similarity score: {similarity:.4f}"
        )

        # ========================================
        # STEP 7
        # DECISION
        # ========================================

        threshold = 0.70

        if similarity >= threshold:

            print("\n✓ AUTHENTICATION SUCCESS")

            return True, similarity

        else:

            print("\n✗ AUTHENTICATION FAILURE")

            return False, similarity

    # ============================================
    # PAYLOAD EXTRACTION
    # ============================================

    def extract_payload(
        self,
        stego_audio_path
    ):

        """
        Connect to your existing
        payload_extractor.py / feature_decoupling.py.
        """

        raise NotImplementedError(
            "Connect your existing HIFI "
            "payload extraction code here."
        )

    # ============================================
    # DECRYPTION
    # ============================================

    def decrypt_template(
        self,
        encrypted_template,
        username
    ):

        """
        Connect to your existing
        ML-KEM + AES decryption.
        """

        raise NotImplementedError(
            "Connect your existing "
            "ML-KEM + AES decryption code here."
        )

    # ============================================
    # COSINE SIMILARITY
    # ============================================

    @staticmethod
    def cosine_similarity(
        vector1,
        vector2
    ):

        vector1 = np.asarray(
            vector1,
            dtype=np.float32
        )

        vector2 = np.asarray(
            vector2,
            dtype=np.float32
        )

        denominator = (
            np.linalg.norm(vector1)
            *
            np.linalg.norm(vector2)
        )

        if denominator == 0:

            return 0.0

        return float(
            np.dot(vector1, vector2)
            / denominator
        )