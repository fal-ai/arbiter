import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_vol() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)

            vol = get_test_measurement(
                media_type=("image",),
                unique_name="variance_of_laplacian",
                alias="vol",
            )

            base = vol.calculate((image,))

            # Blur should reduce VoL
            down = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            blurred = torch.nn.functional.interpolate(
                down.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            blurred_score = vol.calculate((blurred,))
            assert blurred_score <= base + 1e-4

            # Add noise should typically increase VoL
            noise = (torch.randn_like(image) * 0.2).clamp(-1, 1)
            noisy = torch.clamp(image + noise, 0, 1)
            noisy_score = vol.calculate((noisy,))
            assert noisy_score >= base - 1e-4


def test_image_vol() -> None:
    asyncio.run(_test_image_vol())
