import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_clipiqa() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)

            try:
                clip_iqa = get_test_measurement(
                    media_type=("image",),
                    unique_name="clip_image_quality_assessment",
                    alias="clip_iqa",
                )
            except Exception as e:
                print(f"Skipping CLIP-IQA test due to missing deps: {e}")
                return

            # Basic run with default prompts
            scores_clean = clip_iqa.calculate((image,))
            assert isinstance(scores_clean, dict) and len(scores_clean) > 0

            # Provide a custom prompt pair and verify keys and output type
            clip_iqa.set_custom_prompts(
                {
                    "sharpness": (
                        "a blurry out-of-focus photo",
                        "a sharp detailed photo",
                    ),
                    "brightness": (
                        "a dark underexposed photo",
                        "a well-lit bright photo",
                    ),
                }
            )
            scores_custom = clip_iqa.calculate((image,))
            assert set(scores_custom.keys()) == {"sharpness", "brightness"}
            for v in scores_custom.values():
                assert isinstance(v, float)

            # Degrade image and ensure at least one score changes noticeably
            down = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            degraded = torch.nn.functional.interpolate(
                down.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            scores_degraded = clip_iqa.calculate((degraded,))
            diffs = [abs(scores_degraded[k] - scores_custom[k]) for k in scores_custom]
            assert any(
                d > 0.01 for d in diffs
            ), "Expected some change in scores after degradation"


def test_image_clipiqa() -> None:
    asyncio.run(_test_image_clipiqa())
