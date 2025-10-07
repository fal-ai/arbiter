import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_musiq() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)

            try:
                musiq = get_test_measurement(
                    media_type=("image",),
                    unique_name="musiq",
                    alias="musiq",
                )
            except Exception as e:
                print(f"Skipping MUSIQ test due to missing deps: {e}")
                return

            score_clean = musiq.calculate((image,))

            # Blur degradation
            down = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            blurred = torch.nn.functional.interpolate(
                down.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            score_blurred = musiq.calculate((blurred,))

            # Noise degradation
            noise = torch.randn_like(image) * 0.15
            noisy = torch.clamp(image + noise, 0, 1)
            score_noisy = musiq.calculate((noisy,))

            # Expect degradations to generally not improve score
            assert score_blurred <= score_clean + 0.2
            assert score_noisy <= score_clean + 0.2


def test_image_musiq() -> None:
    asyncio.run(_test_image_musiq())
