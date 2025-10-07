import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

# Use a short test video if available, otherwise create synthetic videos
test_video_url = "https://storage.googleapis.com/falserverless/web-examples/wan/t2v.mp4"


async def _test_video_clip_iqa() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            clip_iqa = get_test_measurement(
                media_type=("video",),
                unique_name="video_clip_image_quality_assessment",
                alias="video_clip_iqa",
            )

            clip_iqa_baseline = clip_iqa.calculate((video,))
            assert all([f["quality"] > 0.0 for f in clip_iqa_baseline])

            video.add_(torch.randn_like(video) * 0.85)
            video.clamp_(0, 1)
            clip_iqa_degraded = clip_iqa.calculate((video,))
            assert all(
                [
                    frame["quality"] > degraded_frame["quality"]
                    for frame, degraded_frame in zip(
                        clip_iqa_baseline, clip_iqa_degraded
                    )
                ]
            )

            print(f"CLIP-IQA baseline: {clip_iqa_baseline}")
            print(f"CLIP-IQA degraded: {clip_iqa_degraded}")


def test_video_clip_iqa() -> None:
    asyncio.run(_test_video_clip_iqa())
