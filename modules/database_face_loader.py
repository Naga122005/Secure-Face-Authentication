import mysql.connector
import numpy as np


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "naga*"
DB_NAME = "secure_face_authentication"


# ============================================================
# LOAD FACE FEATURE
# ============================================================

def load_face_feature(username):

    print("=" * 60)
    print("DATABASE FACE FEATURE LOADER")
    print("=" * 60)

    try:

        # Connect to MySQL
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        print("\nDatabase connected successfully.")

        cursor = connection.cursor()

        # Retrieve feature
        query = """
        SELECT face_features
        FROM users
        WHERE username = %s
        """

        cursor.execute(
            query,
            (username,)
        )

        result = cursor.fetchone()

        if result is None:

            print(
                f"\nERROR: User '{username}' not found."
            )

            cursor.close()
            connection.close()

            return None

        # Retrieve binary data
        feature_bytes = result[0]

        print(
            "\nFace feature retrieved successfully."
        )

        print(
            "Retrieved bytes:",
            len(feature_bytes)
        )

        # Convert bytes back to float32
        face_features = np.frombuffer(
            feature_bytes,
            dtype=np.float32
        ).copy()

        print(
            "Feature shape:",
            face_features.shape
        )

        print(
            "Feature dtype:",
            face_features.dtype
        )

        cursor.close()
        connection.close()

        return face_features

    except mysql.connector.Error as error:

        print(
            "\nDatabase error:"
        )

        print(error)

        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    username = input(
        "Enter username: "
    ).strip()

    features = load_face_feature(
        username
    )

    if features is not None:

        print("\n" + "=" * 60)
        print("FEATURE LOADING SUCCESSFUL")
        print("=" * 60)

        print(
            "\nUser:",
            username
        )

        print(
            "Feature dimensions:",
            features.shape
        )

        print(
            "Number of features:",
            len(features)
        )

        print(
            "First 5 values:",
            features[:5]
        )