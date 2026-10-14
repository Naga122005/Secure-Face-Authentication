import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# FEATURE COUPLING MODEL
# ============================================================

class FeatureCouplingNetwork(nn.Module):
    """
    Feature coupling network.

    Input:
        Content features
        Style features

    Output:
        512-dimensional coupled latent representation.

    The network dynamically adapts to the number of
    time frames in the selected VCC2020 audio.
    """

    def __init__(
        self,
        content_size,
        style_size,
        latent_size=512
    ):

        super().__init__()

        self.content_size = content_size
        self.style_size = style_size
        self.latent_size = latent_size

        # ----------------------------------------------------
        # CONTENT PROJECTION
        # ----------------------------------------------------

        self.content_projection = nn.Sequential(

            nn.Linear(
                content_size,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                latent_size
            )
        )

        # ----------------------------------------------------
        # STYLE PROJECTION
        # ----------------------------------------------------

        self.style_projection = nn.Sequential(

            nn.Linear(
                style_size,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                latent_size
            )
        )

        # ----------------------------------------------------
        # COUPLING LAYER
        # ----------------------------------------------------

        self.coupling_layer = nn.Sequential(

            nn.Linear(
                latent_size * 2,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                latent_size
            )
        )

    def forward(
        self,
        content,
        style
    ):

        # ----------------------------------------------------
        # CONTENT PROJECTION
        # ----------------------------------------------------

        content_latent = self.content_projection(
            content
        )

        # ----------------------------------------------------
        # STYLE PROJECTION
        # ----------------------------------------------------

        style_latent = self.style_projection(
            style
        )

        # ----------------------------------------------------
        # COMBINE CONTENT + STYLE
        # ----------------------------------------------------

        combined = torch.cat(
            (
                content_latent,
                style_latent
            ),
            dim=1
        )

        # ----------------------------------------------------
        # FINAL COUPLING
        # ----------------------------------------------------

        coupled_latent = self.coupling_layer(
            combined
        )

        return coupled_latent


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    # --------------------------------------------------------
    # PROJECT ROOT
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    audio_folder = os.path.join(
        project_root,
        "data",
        "audio"
    )

    # --------------------------------------------------------
    # FEATURE FILES
    # --------------------------------------------------------

    content_path = os.path.join(
        audio_folder,
        "content_features.npy"
    )

    style_path = os.path.join(
        audio_folder,
        "style_features.npy"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(content_path):

        raise FileNotFoundError(
            "\nContent feature file not found:\n"
            + content_path
        )

    if not os.path.exists(style_path):

        raise FileNotFoundError(
            "\nStyle feature file not found:\n"
            + style_path
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    content_features = np.load(
        content_path
    )

    style_features = np.load(
        style_path
    )

    print("\nFeatures loaded successfully!")

    print(
        "Content feature shape:",
        content_features.shape
    )

    print(
        "Style feature shape:",
        style_features.shape
    )

    return (
        content_features,
        style_features
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    content_features,
    style_features
):
    """
    Convert 2D feature matrices into flattened
    PyTorch tensors.

    Example:

        Content:
            (8, 146)

        becomes:
            (1, 1168)

        Style:
            (5, 146)

        becomes:
            (1, 730)
    """

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content_flat = content_features.flatten()

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    style_flat = style_features.flatten()

    # --------------------------------------------------------
    # CONVERT TO TENSOR
    # --------------------------------------------------------

    content_tensor = torch.tensor(
        content_flat,
        dtype=torch.float32
    ).unsqueeze(0)

    style_tensor = torch.tensor(
        style_flat,
        dtype=torch.float32
    ).unsqueeze(0)

    print("\nPrepared feature tensors:")

    print(
        "Content tensor:",
        content_tensor.shape
    )

    print(
        "Style tensor:",
        style_tensor.shape
    )

    return (
        content_tensor,
        style_tensor
    )


# ============================================================
# GENERATE COUPLED REPRESENTATION
# ============================================================

def generate_coupled_representation():

    print("\n" + "=" * 60)
    print("FEATURE COUPLING STARTED")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: LOAD FEATURES
    # --------------------------------------------------------

    content_features, style_features = (
        load_features()
    )

    # --------------------------------------------------------
    # STEP 2: VALIDATE DIMENSIONS
    # --------------------------------------------------------

    if content_features.ndim != 2:

        raise ValueError(
            "\nContent features must be a 2D matrix."
        )

    if style_features.ndim != 2:

        raise ValueError(
            "\nStyle features must be a 2D matrix."
        )

    # --------------------------------------------------------
    # STEP 3: PREPARE TENSORS
    # --------------------------------------------------------

    content_tensor, style_tensor = (
        prepare_features(
            content_features,
            style_features
        )
    )

    # --------------------------------------------------------
    # STEP 4: DETERMINE INPUT DIMENSIONS
    # --------------------------------------------------------

    content_size = (
        content_tensor.shape[1]
    )

    style_size = (
        style_tensor.shape[1]
    )

    print("\nDynamic Model Dimensions")
    print("-" * 40)

    print(
        "Content input size:",
        content_size
    )

    print(
        "Style input size:",
        style_size
    )

    # --------------------------------------------------------
    # STEP 5: CREATE MODEL
    # --------------------------------------------------------

    model = FeatureCouplingNetwork(
        content_size=content_size,
        style_size=style_size,
        latent_size=512
    )

    model.eval()

    print(
        "\nFeature coupling network created successfully."
    )

    # --------------------------------------------------------
    # STEP 6: COUPLE CONTENT + STYLE
    # --------------------------------------------------------

    print(
        "\nCoupling embedded content with style features..."
    )

    with torch.no_grad():

        coupled_latent = model(
            content_tensor,
            style_tensor
        )

    # --------------------------------------------------------
    # STEP 7: DISPLAY OUTPUT
    # --------------------------------------------------------

    print(
        "\nFeature coupling completed successfully!"
    )

    print(
        "Coupled latent shape:",
        coupled_latent.shape
    )

    # --------------------------------------------------------
    # STEP 8: SAVE OUTPUT
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    audio_folder = os.path.join(
        project_root,
        "data",
        "audio"
    )

    os.makedirs(
        audio_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        audio_folder,
        "coupled_representation.npy"
    )

    np.save(
        output_path,
        coupled_latent.numpy()
    )

    print(
        "\nCoupled representation saved successfully!"
    )

    print(
        "Saved at:"
    )

    print(
        output_path
    )

    # --------------------------------------------------------
    # STEP 9: FINAL INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE COUPLING COMPLETED")
    print("=" * 60)

    print("\nFINAL OUTPUT")
    print("-" * 40)

    print(
        "Content features:",
        content_features.shape
    )

    print(
        "Style features:",
        style_features.shape
    )

    print(
        "Content tensor:",
        content_tensor.shape
    )

    print(
        "Style tensor:",
        style_tensor.shape
    )

    print(
        "Coupled latent:",
        coupled_latent.shape
    )

    print(
        "\nSaved file:"
    )

    print(
        output_path
    )

    print("=" * 60)

    return coupled_latent


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        generate_coupled_representation()

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            "\n",
            str(e)
        )

        print("=" * 60)

        raise