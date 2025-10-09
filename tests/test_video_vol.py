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


async def _test_video_vol() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            vol = get_test_measurement(
                media_type=("video",),
                unique_name="video_variance_of_laplacian",
                alias="video_vol",
            )

            vol_baseline = vol.calculate((video,))
            assert all([f > 0.0 for f in vol_baseline])

            # Blur video to decrease sharpness
            video_blurred = torch.nn.functional.avg_pool3d(
                video, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)
            )
            vol_blurred = vol.calculate((video_blurred,))
            assert all(
                [vol_blurred[i] < vol_baseline[i] for i in range(len(vol_blurred))]
            )

            print(f"VOL baseline: {vol_baseline}")
            print(f"VOL blurred: {vol_blurred}")


def test_video_vol() -> None:
    asyncio.run(_test_video_vol())
