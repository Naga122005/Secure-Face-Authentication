import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf


class AudioDecoder(nn.Module):
    """
    Decoder that converts the coupled latent representation
    into an audio waveform.
    """

    def __init__(self, output_length):
        super(AudioDecoder, self).__init__()

        self.output_length = output_length

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.ReLU(),

            nn.Linear(2048, 4096),
            nn.ReLU(),

            nn.Linear(4096, output_length),
            nn.Tanh()
        )

    def forward(self, latent):
        return self.decoder(latent)


def reconstruct_audio():

    print("=" * 60)
    print("AUDIO RECONSTRUCTION STARTED")
    print("=" * 60)

    # --------------------------------------------------
    # PROJECT ROOT
    # --------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # --------------------------------------------------
    # INPUT PATHS
    # --------------------------------------------------

    coupled_path = os.path.join(
        project_root,
        "data",
        "audio",
        "coupled_representation.npy"
    )

    original_audio_path = os.path.join(
        project_root,
        "data",
        "audio",
        "sample.wav"
    )

    # --------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------

    if not os.path.exists(coupled_path):

        print("\nERROR: coupled_representation.npy not found!")
        print("Expected:")
        print(coupled_path)

        return

    if not os.path.exists(original_audio_path):

        print("\nERROR: sample.wav not found!")
        print("Expected:")
        print(original_audio_path)

        return

    # --------------------------------------------------
    # LOAD COUPLED REPRESENTATION
    # --------------------------------------------------

    print("\nLoading coupled representation...")

    coupled_latent = np.load(
        coupled_path
    )

    print(
        "Coupled representation loaded successfully."
    )

    print(
        "Coupled latent shape:",
        coupled_latent.shape
    )

    # --------------------------------------------------
    # LOAD ORIGINAL AUDIO INFORMATION
    # --------------------------------------------------

    print("\nLoading original audio...")

    original_audio, sample_rate = sf.read(
        original_audio_path
    )

    print("Original audio loaded successfully.")

    print("Sample rate:", sample_rate)

    print(
        "Original audio samples:",
        len(original_audio)
    )

    print(
        "Original duration:",
        round(
            len(original_audio) / sample_rate,
            2
        ),
        "seconds"
    )

    # --------------------------------------------------
    # CONVERT LATENT TO TORCH
    # --------------------------------------------------

    coupled_tensor = torch.tensor(
        coupled_latent,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # CREATE DECODER
    # --------------------------------------------------

    print("\nCreating audio decoder...")

    decoder = AudioDecoder(
        output_length=len(original_audio)
    )

    # --------------------------------------------------
    # AUDIO RECONSTRUCTION
    # --------------------------------------------------

    print("\nReconstructing audio...")

    decoder.eval()

    with torch.no_grad():

        reconstructed_audio = decoder(
            coupled_tensor
        )

    # Convert tensor to NumPy
    reconstructed_audio = (
        reconstructed_audio
        .cpu()
        .numpy()
        .squeeze()
    )

    # --------------------------------------------------
    # NORMALIZE AUDIO
    # --------------------------------------------------

    max_value = np.max(
        np.abs(reconstructed_audio)
    )

    if max_value > 0:

        reconstructed_audio = (
            reconstructed_audio / max_value
        )

    # --------------------------------------------------
    # SAVE STEGO AUDIO
    # --------------------------------------------------

    output_path = os.path.join(
        project_root,
        "data",
        "audio",
        "stego_audio.wav"
    )

    sf.write(
        output_path,
        reconstructed_audio,
        sample_rate
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nAudio reconstruction completed!")

    print(
        "Reconstructed audio samples:",
        len(reconstructed_audio)
    )

    print(
        "Reconstructed duration:",
        round(
            len(reconstructed_audio) / sample_rate,
            2
        ),
        "seconds"
    )

    print("\nStego audio saved successfully!")

    print("Saved at:")
    print(output_path)

    print("\n" + "=" * 60)
    print("STEP 8 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    reconstruct_audio()