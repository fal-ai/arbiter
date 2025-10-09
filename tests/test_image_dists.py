import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_dists() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url) as image_path:
            image = read_media(image_path)
            dists = get_test_measurement(
                media_type=("image", "image"),
                unique_name="deep_image_structure_and_texture_similarity",
                alias="dists",
            )

            # Test 1: Identical images should have DISTS = 0.0
            dists_identical = dists.calculate((image, image))
            assert (
                abs(dists_identical) < 1e-5
            ), f"Expected DISTS of 0.0 for identical images, got {dists_identical}"

            # Test 2: Add small noise - DISTS should be small but > 0
            noise = torch.randn_like(image) * 0.05
            noisy_image = torch.clamp(image + noise, 0, 1)
            dists_noise = dists.calculate((image, noisy_image))
            assert (
                0.0 < dists_noise < 0.2
            ), f"Expected DISTS between 0.0 and 0.2 for slightly noisy image, got {dists_noise}"

            # Test 3: Add more noise - should have higher DISTS
            noise_large = torch.randn_like(image) * 0.2
            very_noisy_image = torch.clamp(image + noise_large, 0, 1)
            dists_noise_large = dists.calculate((image, very_noisy_image))
            assert (
                dists_noise < dists_noise_large
            ), f"Expected larger noise to have higher DISTS: {dists_noise} >= {dists_noise_large}"
            assert (
                dists_noise_large < 0.5
            ), f"Expected DISTS < 0.5 for noisy image, got {dists_noise_large}"

            # Test 4: Gaussian blur should affect texture more than structure
            # Use built-in Gaussian blur from torchvision transforms
            from torchvision import transforms

            blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
            blurred = blur_transform(image)
            dists_blur = dists.calculate((image, blurred))
            assert (
                0.0 < dists_blur < 0.3
            ), f"Expected DISTS for blurred image between 0.0 and 0.3, got {dists_blur}"

            # Test 5: Test different image sizes (DISTS should handle this)
            small_image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            small_noisy = torch.nn.functional.interpolate(
                noisy_image.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            dists_small = dists.calculate((small_image, small_noisy))
            assert (
                0.0 < dists_small < 0.3
            ), f"Expected reasonable DISTS for smaller images, got {dists_small}"


def test_image_dists() -> None:
    asyncio.run(_test_image_dists())
