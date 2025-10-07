import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_arniqa() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)

            try:
                arniqa = get_test_measurement(
                    media_type=("image",),
                    unique_name="arniqa",
                    alias="arniqa",
                )
            except Exception as e:
                # Optional dependency not available; skip test
                print(f"Skipping ARNIQA test due to missing deps: {e}")
                return

            # Same image score should be finite in [0, 1]
            score_clean = arniqa.calculate((image,))
            assert (
                0.0 <= score_clean <= 1.0
            ), f"Expected score in [0,1], got {score_clean}"

            # Add small gaussian noise -> quality should decrease
            noise_small = torch.randn_like(image) * 0.03
            noisy_small = torch.clamp(image + noise_small, 0, 1)
            score_noisy_small = arniqa.calculate((noisy_small,))
            assert (
                score_noisy_small <= score_clean + 0.1
            ), f"Expected small noise not to improve quality: clean {score_clean}, noisy {score_noisy_small}"

            # Add stronger noise -> quality should further decrease
            noise_large = torch.randn_like(image) * 0.15
            noisy_large = torch.clamp(image + noise_large, 0, 1)
            score_noisy_large = arniqa.calculate((noisy_large,))
            assert (
                score_noisy_large < score_noisy_small + 0.05
            ), f"Expected larger noise to reduce quality: small {score_noisy_small}, large {score_noisy_large}"

            # Apply heavy JPEG-like blur (simulate with gaussian blur via interpolate down/up)
            down = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            blurred = torch.nn.functional.interpolate(
                down.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            score_blurred = arniqa.calculate((blurred,))
            assert (
                score_blurred < score_clean
            ), f"Expected blur to reduce quality: clean {score_clean}, blurred {score_blurred}"


def test_image_arniqa() -> None:
    asyncio.run(_test_image_arniqa())
