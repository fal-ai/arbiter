import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_lpips() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)
            lpips = get_test_measurement(
                media_type=("image", "image"),
                unique_name="learned_perceptual_image_patch_similarity",
                alias="lpips",
            )

            # Test 1: Identical images should have LPIPS = 0.0
            lpips_identical = lpips.calculate((image, image))
            assert (
                abs(lpips_identical) < 1e-5
            ), f"Expected LPIPS of 0.0 for identical images, got {lpips_identical}"

            # Test 2: Add small noise - LPIPS should be small but > 0
            noise = torch.randn_like(image) * 0.05
            noisy_image = torch.clamp(image + noise, 0, 1)
            lpips_noise = lpips.calculate((image, noisy_image))
            assert (
                0.0 < lpips_noise < 0.3
            ), f"Expected LPIPS between 0.0 and 0.3 for slightly noisy image, got {lpips_noise}"

            # Test 3: Add more noise - should have higher LPIPS
            noise_large = torch.randn_like(image) * 0.2
            very_noisy_image = torch.clamp(image + noise_large, 0, 1)
            lpips_noise_large = lpips.calculate((image, very_noisy_image))
            assert (
                lpips_noise < lpips_noise_large
            ), f"Expected larger noise to have higher LPIPS: {lpips_noise} >= {lpips_noise_large}"
            assert (
                lpips_noise_large < 1.0
            ), f"Expected LPIPS < 1.0 for noisy image, got {lpips_noise_large}"

            # Test 4: Inverted image should have high LPIPS
            inverted = 1.0 - image
            lpips_inverted = lpips.calculate((image, inverted))
            assert (
                lpips_inverted > 0.3
            ), f"Expected LPIPS > 0.3 for inverted image, got {lpips_inverted}"

            # Test 5: Random image should have very high LPIPS
            random_image = torch.rand_like(image)
            lpips_random = lpips.calculate((image, random_image))
            assert (
                lpips_random > 0.6
            ), f"Expected high LPIPS for random image, got {lpips_random}"

            # Test 6: Test different image sizes (LPIPS should handle this)
            # Resize to different size
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
            lpips_small = lpips.calculate((small_image, small_noisy))
            assert (
                0.0 < lpips_small < 0.4
            ), f"Expected reasonable LPIPS for smaller images, got {lpips_small}"


def test_image_lpips() -> None:
    asyncio.run(_test_image_lpips())
