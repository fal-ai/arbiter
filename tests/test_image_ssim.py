import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_ssim() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)
            ssim = get_test_measurement(
                media_type=("image", "image"),
                unique_name="structural_similarity_index_measure",
                alias="ssim",
            )

            # Test 1: Identical images should have SSIM = 1.0
            ssim_identical = ssim.calculate((image, image))
            assert (
                abs(ssim_identical - 1.0) < 1e-5
            ), f"Expected SSIM of 1.0 for identical images, got {ssim_identical}"

            # Test 2: Add small noise - SSIM should be high but < 1
            noise = torch.randn_like(image) * 0.05
            noisy_image = torch.clamp(image + noise, 0, 1)
            ssim_noise = ssim.calculate((image, noisy_image))
            assert (
                0.5 < ssim_noise < 1.0
            ), f"Expected SSIM between 0.5 and 1.0 for slightly noisy image, got {ssim_noise}"

            # Test 3: Inverted image should have lower SSIM
            inverted = 1.0 - image
            ssim_inverted = ssim.calculate((image, inverted))
            assert (
                ssim_inverted < 0.5
            ), f"Expected SSIM < 0.5 for inverted image, got {ssim_inverted}"

            # Test 4: Completely different image (random) should have very low SSIM
            random_image = torch.rand_like(image)
            ssim_random = ssim.calculate((image, random_image))
            assert (
                -0.2 < ssim_random < 0.3
            ), f"Expected low SSIM for random image, got {ssim_random}"


def test_image_ssim() -> None:
    asyncio.run(_test_image_ssim())
