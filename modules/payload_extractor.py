import os
import numpy as np
import librosa
import torch
import torch.nn as nn


class PayloadExtractor(nn.Module):
    """
    Extracts the embedded payload from the content
    representation received from the stego audio.
    """

    def __init__(self):
        super(PayloadExtractor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Linear(8 * 146, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 2076)
        )

    def forward(self, content_features):

        # Flatten content features
        content_features = content_features.reshape(
            content_features.shape[0],
            -1
        )

        payload = self.extractor(
            content_features
        )

        return payload


def extract_payload():

    print("=" * 60)
    print("PAYLOAD EXTRACTION STARTED")
    print("=" * 60)

    # --------------------------------------------------
    # PROJECT ROOT
    # --------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # --------------------------------------------------
    # STEGO AUDIO PATH
    # --------------------------------------------------

    stego_audio_path = os.path.join(
        project_root,
        "data",
        "audio",
        "stego_audio.wav"
    )

    if not os.path.exists(stego_audio_path):

        print("\nERROR: stego_audio.wav not found!")

        print("Expected:")
        print(stego_audio_path)

        return

    # --------------------------------------------------
    # LOAD STEGO AUDIO
    # --------------------------------------------------

    print("\nLoading stego audio...")

    audio, sample_rate = librosa.load(
        stego_audio_path,
        sr=None
    )

    print("Stego audio loaded successfully.")

    print("Sample rate:", sample_rate)

    print(
        "Number of samples:",
        len(audio)
    )

    print(
        "Duration:",
        round(
            len(audio) / sample_rate,
            2
        ),
        "seconds"
    )

    # --------------------------------------------------
    # NORMALIZE AUDIO
    # --------------------------------------------------

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = audio / max_value

    print("\nAudio normalized successfully.")

    # --------------------------------------------------
    # EXTRACT MFCC
    # --------------------------------------------------

    print("\nExtracting MFCC features...")

    mfcc_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    print(
        "MFCC shape:",
        mfcc_features.shape
    )

    # --------------------------------------------------
    # HANDLE FRAME SIZE
    # --------------------------------------------------

    target_frames = 146

    if mfcc_features.shape[1] < target_frames:

        padding = target_frames - mfcc_features.shape[1]

        mfcc_features = np.pad(
            mfcc_features,
            ((0, 0), (0, padding)),
            mode="constant"
        )

    elif mfcc_features.shape[1] > target_frames:

        mfcc_features = mfcc_features[
            :, :target_frames
        ]

    print(
        "Adjusted MFCC shape:",
        mfcc_features.shape
    )

    # --------------------------------------------------
    # DECOUPLE CONTENT
    # --------------------------------------------------

    content_features = mfcc_features[:8, :]

    print("\nContent features extracted.")

    print(
        "Content shape:",
        content_features.shape
    )

    # --------------------------------------------------
    # CONVERT TO TORCH
    # --------------------------------------------------

    content_tensor = torch.tensor(
        content_features,
        dtype=torch.float32
    ).unsqueeze(0)

    # --------------------------------------------------
    # CREATE EXTRACTOR
    # --------------------------------------------------

    extractor = PayloadExtractor()

    extractor.eval()

    # --------------------------------------------------
    # EXTRACT PAYLOAD
    # --------------------------------------------------

    print("\nExtracting embedded payload...")

    with torch.no_grad():

        extracted_payload = extractor(
            content_tensor
        )

    extracted_payload = (
        extracted_payload
        .cpu()
        .numpy()
        .squeeze()
    )

    print(
        "\nExtracted payload successfully."
    )

    print(
        "Extracted payload shape:",
        extracted_payload.shape
    )

    # --------------------------------------------------
    # SAVE EXTRACTED PAYLOAD
    # --------------------------------------------------

    output_path = os.path.join(
        project_root,
        "data",
        "faces",
        "extracted_face_payload.npy"
    )

    np.save(
        output_path,
        extracted_payload
    )

    print("\nExtracted payload saved successfully.")

    print("Saved at:")
    print(output_path)

    print("\n" + "=" * 60)
    print("STEP 9 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    extract_payload()