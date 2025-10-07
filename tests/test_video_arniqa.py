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


async def _test_video_arniqa() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            arniqa = get_test_measurement(
                media_type=("video",),
                unique_name="video_arniqa",
                alias="arnvqa",
            )

            arniqa_baseline = arniqa.calculate((video,))
            assert all([f > 0.0 for f in arniqa_baseline])

            video.add_(torch.randn_like(video) * 0.5)
            video.clamp_(0, 1)
            arniqa_degraded = arniqa.calculate((video,))
            assert all(
                [
                    arniqa_degraded[i] < arniqa_baseline[i]
                    for i in range(len(arniqa_degraded))
                ]
            )

            print(f"ARNIQA baseline: {arniqa_baseline}")
            print(f"ARNIQA degraded: {arniqa_degraded}")


def test_video_arniqa() -> None:
    asyncio.run(_test_video_arniqa())
