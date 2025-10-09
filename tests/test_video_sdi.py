import asyncio

import torch
from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

# Use a short test video if available
test_video_url = "https://storage.googleapis.com/falserverless/web-examples/wan/t2v.mp4"


async def _test_video_sdi() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            sdi = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_spectral_distortion_index",
                alias="video_sdi",
            )

            # Test identical videos - should have very low SDI
            sdi_identical = sdi.calculate((video, video))
            assert all([abs(d) < 1e-5 for d in sdi_identical])

            # Test with slightly different video
            noisy_video = video + torch.randn_like(video) * 0.1
            noisy_video.clamp_(0, 1)
            sdi_noisy = sdi.calculate((video, noisy_video))
            assert all([d > 0.0 for d in sdi_noisy])

            print(f"SDI identical: {sdi_identical}")
            print(f"SDI noisy: {sdi_noisy}")


def test_video_sdi() -> None:
    asyncio.run(_test_video_sdi())
