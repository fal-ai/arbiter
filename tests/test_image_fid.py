import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_fid() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url) as image_path:
            image = read_media(image_path)
            fid = get_test_measurement(
                media_type=("image", "image"),
                unique_name="frechet_inception_distance",
                alias="fid",
                aggregate=True,  # This is an aggregate measurement
            )

            # Test 1: Identical distributions should have very low FID
            # Create a set of image pairs where both sets are identical
            identical_pairs = []
            for i in range(20):  # FID needs more samples than KID
                # Add some variation to create a distribution
                noise = torch.randn_like(image) * 0.01 * i
                img_variant = torch.clamp(image + noise, 0, 1)
                identical_pairs.append((img_variant, img_variant))

            fid_identical = fid.calculate(identical_pairs)
            assert isinstance(fid_identical, float), "FID should return a float"
            assert (
                fid_identical < 10.0
            ), f"Expected low FID for identical distributions, got {fid_identical}"

            # Test 2: Slightly different distributions
            # Create pairs where the second image has small noise added
            noisy_pairs = []
            for i in range(20):
                base_noise = torch.randn_like(image) * 0.02 * i
                img_base = torch.clamp(image + base_noise, 0, 1)

                added_noise = torch.randn_like(image) * 0.1
                img_noisy = torch.clamp(img_base + added_noise, 0, 1)
                noisy_pairs.append((img_base, img_noisy))

            fid_noisy = fid.calculate(noisy_pairs)
            assert (
                fid_noisy > fid_identical
            ), f"Expected higher FID for noisy distribution: {fid_noisy} <= {fid_identical}"
            assert (
                fid_noisy < 150.0
            ), f"Expected reasonable FID for slightly noisy distribution, got {fid_noisy}"

            # Test 3: Very different distributions
            # Create pairs where real and generated are very different
            different_pairs = []
            for i in range(20):
                # Real images with slight variations
                real_noise = torch.randn_like(image) * 0.02 * i
                img_real = torch.clamp(image + real_noise, 0, 1)

                # Generated images are random
                img_random = torch.rand_like(image)
                different_pairs.append((img_real, img_random))

            fid_different = fid.calculate(different_pairs)
            assert (
                fid_different > fid_noisy
            ), f"Expected higher FID for very different distributions: {fid_different} <= {fid_noisy}"

            # Test 4: Inverted images should have moderate FID
            inverted_pairs = []
            for i in range(20):
                noise = torch.randn_like(image) * 0.02 * i
                img_base = torch.clamp(image + noise, 0, 1)
                img_inverted = 1.0 - img_base
                inverted_pairs.append((img_base, img_inverted))

            fid_inverted = fid.calculate(inverted_pairs)
            assert (
                fid_inverted > 30.0
            ), f"Expected moderate FID for inverted distributions, got {fid_inverted}"

            # Test 5: Test with different image sizes
            # FID should handle this by resizing internally
            mixed_size_pairs = []
            for i in range(10):
                # Create images of different sizes
                size = 256 + i * 32  # 256, 288, 320, 352, 384, etc.
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
            fid_mixed = fid.calculate(mixed_size_pairs)
            assert isinstance(fid_mixed, float), "FID should handle mixed sizes"


def test_image_fid() -> None:
    asyncio.run(_test_image_fid())
