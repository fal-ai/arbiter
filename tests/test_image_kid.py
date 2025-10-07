import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_kid() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url) as image_path:
            image = read_media(image_path)
            kid = get_test_measurement(
                media_type=("image", "image"),
                unique_name="kernel_inception_distance",
                alias="kid",
                aggregate=True,  # This is an aggregate measurement
            )

            # Test 1: Identical distributions should have very low KID
            # Create a set of image pairs where both sets are identical
            identical_pairs = []
            for i in range(10):
                # Add some variation to create a distribution
                noise = torch.randn_like(image) * 0.01 * i
                img_variant = torch.clamp(image + noise, 0, 1)
                identical_pairs.append((img_variant, img_variant))

            kid_identical = kid.calculate(identical_pairs)
            assert isinstance(kid_identical, dict), "KID should return a dictionary"
            assert (
                "mean" in kid_identical and "std" in kid_identical
            ), "KID should return mean and std"
            assert (
                kid_identical["mean"] < 0.1
            ), f"Expected low KID for identical distributions, got {kid_identical['mean']}"

            # Test 2: Slightly different distributions
            # Create pairs where the second image has small noise added
            noisy_pairs = []
            for i in range(10):
                base_noise = torch.randn_like(image) * 0.02 * i
                img_base = torch.clamp(image + base_noise, 0, 1)

                added_noise = torch.randn_like(image) * 0.25
                img_noisy = torch.clamp(img_base + added_noise, 0, 1)
                noisy_pairs.append((img_base, img_noisy))

            kid_noisy = kid.calculate(noisy_pairs)
            assert (
                kid_noisy["mean"] > kid_identical["mean"]
            ), f"Expected higher KID for noisy distribution: {kid_noisy['mean']} <= {kid_identical['mean']}"
            assert (
                kid_noisy["mean"] < 1.0
            ), f"Expected reasonable KID for slightly noisy distribution, got {kid_noisy['mean']}"

            # Test 3: Very different distributions
            # Create pairs where real and generated are very different
            different_pairs = []
            for i in range(10):
                # Real images with slight variations
                real_noise = torch.randn_like(image) * 0.02 * i
                img_real = torch.clamp(image + real_noise, 0, 1)

                # Generated images are random
                img_random = torch.rand_like(image)
                different_pairs.append((img_real, img_random))

            kid_different = kid.calculate(different_pairs)
            assert (
                kid_different["mean"] > kid_noisy["mean"]
            ), f"Expected higher KID for very different distributions: {kid_different['mean']} <= {kid_noisy['mean']}"

            # Test 4: Test with different image sizes
            # KID should handle this by resizing internally
            mixed_size_pairs = []
            for i in range(5):
                # Create images of different sizes
                size = 256 + i * 32  # 256, 288, 320, 352, 384
                resized = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                noise = torch.randn_like(resized) * 0.05
                noisy_resized = torch.clamp(resized + noise, 0, 1)
                mixed_size_pairs.append((resized, noisy_resized))

            # This should work without errors
            kid_mixed = kid.calculate(mixed_size_pairs)
            assert isinstance(kid_mixed, dict), "KID should handle mixed sizes"


def test_image_kid() -> None:
    asyncio.run(_test_image_kid())
