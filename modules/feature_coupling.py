import os
import numpy as np
import torch
import torch.nn as nn


class FeatureCoupling(nn.Module):
    """
    Couples the embedded content representation
    with the original style representation.
    """

    def __init__(self):
        super(FeatureCoupling, self).__init__()

        # Embedded content is expected to be 1024 dimensions
        self.content_projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # Style features are projected into a 512-dimensional latent space
        self.style_projection = nn.Sequential(
            nn.Linear(5 * 146, 512),
            nn.ReLU()
        )

        # Couple content and style
        self.coupling_network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 512)
        )

    def forward(self, embedded_content, style_features):

        # Project embedded content
        content_latent = self.content_projection(
            embedded_content
        )

        # Flatten and project style features
        style_features = style_features.reshape(
            style_features.shape[0], -1
        )

        style_latent = self.style_projection(
            style_features
        )

        # Combine content and style
        combined = torch.cat(
            (content_latent, style_latent),
            dim=1
        )

        # Generate final coupled representation
        coupled_latent = self.coupling_network(
            combined
        )

        return coupled_latent


def generate_coupled_representation():

    print("=" * 60)
    print("FEATURE COUPLING STARTED")
    print("=" * 60)

    # Get project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Path of embedded content
    embedded_path = os.path.join(
        project_root,
        "data",
        "faces",
        "embedded_content.npy"
    )

    # Path where style features will be loaded
    style_path = os.path.join(
        project_root,
        "data",
        "audio",
        "style_features.npy"
    )

    # Check embedded content
    if not os.path.exists(embedded_path):
        print("\nERROR: embedded_content.npy not found!")
        print(embedded_path)
        return

    # Check style features
    if not os.path.exists(style_path):
        print("\nERROR: style_features.npy not found!")
        print(style_path)
        return

    # Load embedded content
    print("\nLoading embedded content...")

    embedded_content = np.load(
        embedded_path
    )

    print("Embedded content loaded successfully.")
    print("Embedded content shape:",
          embedded_content.shape)

    # Load style features
    print("\nLoading style features...")

    style_features = np.load(
        style_path
    )

    print("Style features loaded successfully.")
    print("Style feature shape:",
          style_features.shape)

    # Convert to tensors
    embedded_tensor = torch.tensor(
        embedded_content,
        dtype=torch.float32
    )

    style_tensor = torch.tensor(
        style_features,
        dtype=torch.float32
    )

    # Add batch dimension if necessary
    if style_tensor.dim() == 2:
        style_tensor = style_tensor.unsqueeze(0)

    # Create model
    model = FeatureCoupling()

    print("\nCoupling embedded content with style features...")

    with torch.no_grad():

        coupled_latent = model(
            embedded_tensor,
            style_tensor
        )

    print("\nFeature coupling completed successfully.")

    print(
        "Final coupled latent shape:",
        coupled_latent.shape
    )

    # Save coupled representation
    output_path = os.path.join(
        project_root,
        "data",
        "audio",
        "coupled_representation.npy"
    )

    np.save(
        output_path,
        coupled_latent.numpy()
    )

    print("\nCoupled representation saved successfully.")
    print("Saved at:")
    print(output_path)

    print("\n" + "=" * 60)
    print("STEP 7 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    generate_coupled_representation()