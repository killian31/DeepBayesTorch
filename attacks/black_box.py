import random

import torch


def gaussian_perturbation_attack(
    images: torch.Tensor,
    eps: float,
    mean: float = 0.0,
    seed: int = 29,
) -> torch.Tensor:
    """
    Applies a Gaussian perturbation attack to a batch of images.

    This attack adds Gaussian noise to each pixel in the images, where the noise
    is sampled from a normal distribution with specified mean and standard deviation.
    The perturbation is controlled by the `eps` parameter, which represents the
    standard deviation of the noise.

    Args:
        images (torch.Tensor): A batch of images with shape (B, C, H, W).
        eps (float, optional): Standard deviation of the Gaussian noise. Default is 0.1.
        mean (float, optional): Mean of the Gaussian noise. Default is 0.0.
        seed (int, optional): Random seed for reproducibility. Default is 29.

    Returns:
        torch.Tensor: A batch of images with Gaussian perturbations applied.
    """
    if eps == 0:
        return images
    adv_images = images.clone()
    torch.manual_seed(seed)  # For reproducibility

    # Generate Gaussian noise
    noise = torch.randn_like(adv_images) * eps + mean

    # Apply the noise to the images
    adv_images += noise

    # Optionally, clamp the images to maintain valid pixel range
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    return adv_images


def sticker_attack(
    images: torch.Tensor,
    sticker_size: float = 0.1,
    placement: str = "center",
    seed: int = 29,
) -> torch.Tensor:
    """
    Simulates a sticker attack on a batch of images, with the option to place the sticker at the center.

    Args:
        images (torch.Tensor): A batch of images with shape (B, C, H, W).
        sticker_size (float): Fraction of the image dimension for the sticker area.
        placement (str): Placement of the sticker ("center" or "random").
        color (list): RGB values for the sticker color.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: A batch of images with the sticker applied.
    """
    adv_images = images.clone()
    batch_size, c, h, w = images.shape
    sticker_dim = int(min(h, w) * sticker_size)  # Sticker size in pixels
    random.seed(seed)  # For reproducibility
    flashy_colors = [
        [1.0, 1.0, 0.0],  # Bright yellow
        [0.0, 1.0, 0.0],  # Neon green
        [1.0, 0.0, 1.0],  # Neon pink
        [0.0, 1.0, 1.0],  # Bright cyan
        [1.0, 0.5, 0.0],  # Bright orange
    ]
    for i in range(batch_size):
        color = random.choice(flashy_colors)
        if placement == "center":
            # Calculate top-left corner for a centered sticker
            top = (h - sticker_dim) // 2 + random.randint(-1, 1)
            left = (w - sticker_dim) // 2 + random.randint(-1, 1)
        elif placement == "random":
            # Random placement
            top = random.randint(0, h - sticker_dim)
            left = random.randint(0, w - sticker_dim)
        else:
            raise ValueError(f"Unsupported placement option: {placement}")

        # Apply the sticker by setting pixel values in the specified region
        adv_images[i, :, top : top + sticker_dim, left : left + sticker_dim] = (
            torch.tensor(color).view(-1, 1, 1)
        )

    return adv_images
