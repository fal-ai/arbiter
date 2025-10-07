import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_sdi() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)
            sdi = get_test_measurement(
                media_type=("image", "image"),
                unique_name="spectral_distortion_index",
                alias="d_lambda",
            )

            # Identical images should have SDI ~ 0.0
            sdi_identical = sdi.calculate((image, image))
            assert (
                abs(sdi_identical) < 1e-5
            ), f"Expected SDI of 0.0, got {sdi_identical}"

            # Slight noise should increase SDI a little but not too much
            noise = torch.randn_like(image) * 0.01
            noisy_image = torch.clamp(image + noise, 0, 1)
            sdi_noise = sdi.calculate((image, noisy_image))
            assert (
                0.0 < sdi_noise < 0.3
            ), f"Expected SDI between 0.0 and 0.3, got {sdi_noise}"

            # More noise should increase SDI further
            noise_large = torch.randn_like(image) * 0.2
            very_noisy_image = torch.clamp(image + noise_large, 0, 1)
            sdi_noise_large = sdi.calculate((image, very_noisy_image))
            assert (
                sdi_noise < sdi_noise_large
            ), f"Expected larger noise to have higher SDI: {sdi_noise} >= {sdi_noise_large}"
            assert (
                sdi_noise_large < 1.5
            ), f"Expected SDI < 1.5 for noisy image, got {sdi_noise_large}"

            # Random image should have high SDI
            random_image = torch.rand_like(image)
            sdi_random = sdi.calculate((image, random_image))
            assert (
                sdi_random > 0.6
            ), f"Expected high SDI for random image, got {sdi_random}"

            # Check resized images still compute
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
            sdi_small = sdi.calculate((small_image, small_noisy))
            assert (
                0.0 < sdi_small < 0.5
            ), f"Expected reasonable SDI for smaller images, got {sdi_small}"


def test_image_sdi() -> None:
    asyncio.run(_test_image_sdi())
