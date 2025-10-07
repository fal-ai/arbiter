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


async def _test_video_nima() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            nima = get_test_measurement(
                media_type=("video",),
                unique_name="video_nima",
                alias="video_nima",
            )

            nima_baseline = nima.calculate((video,))
            assert all([f > 0.0 for f in nima_baseline])

            video.add_(torch.randn_like(video) * 0.5)
            video.clamp_(0, 1)
            nima_degraded = nima.calculate((video,))
            assert all(
                [nima_degraded[i] < nima_baseline[i] for i in range(len(nima_degraded))]
            )

            print(f"NIMA baseline: {nima_baseline}")
            print(f"NIMA degraded: {nima_degraded}")


def test_video_nima() -> None:
    asyncio.run(_test_video_nima())
