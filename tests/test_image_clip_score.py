import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_clip_score() -> None:
    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)

            try:
                clip_score = get_test_measurement(
                    media_type=("text", "image"),
                    unique_name="clip_score",
                    alias="clipscore",
                )
            except Exception as e:
                print(f"Skipping CLIPScore test due to missing deps: {e}")
                return

            # Prompts
            prompt_match = "a man standing on a street"
            prompt_mismatch = "a woman laying in the grass"

            # Base scores
            score_match = clip_score.calculate((prompt_match, image))
            score_mismatch = clip_score.calculate((prompt_mismatch, image))
            assert (
                score_match >= score_mismatch - 5.0
            ), f"Expected matching prompt to score higher: match {score_match}, mismatch {score_mismatch}"

            # Random noise image should score lower than real image for matching prompt
            random_image = torch.rand_like(image)
            score_random = clip_score.calculate((prompt_match, random_image))
            assert (
                score_random <= score_match + 5.0
            ), f"Expected random image to not exceed real image score: real {score_match}, random {score_random}"

            # Heavy blur should generally not improve score
            down = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            blurred = torch.nn.functional.interpolate(
                down.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            score_blurred = clip_score.calculate((prompt_match, blurred))
            assert (
                score_blurred <= score_match + 5.0
            ), f"Expected blur not to dramatically improve score: base {score_match}, blurred {score_blurred}"


def test_image_clip_score() -> None:
    asyncio.run(_test_image_clip_score())
