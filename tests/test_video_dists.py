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


async def _test_video_dists() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            dists = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_deep_image_structure_and_texture_similarity",
                alias="video_dists",
            )

            # Test identical videos - should have very low DISTS
            dists_identical = dists.calculate((video, video))
            assert all([abs(d) < 1e-5 for d in dists_identical])

            # Test with slightly noisy video
            noisy_video = video + torch.randn_like(video) * 0.1
            noisy_video.clamp_(0, 1)
            dists_noisy = dists.calculate((video, noisy_video))
            assert all([d > 0.0 for d in dists_noisy])

            print(f"DISTS identical: {dists_identical}")
            print(f"DISTS noisy: {dists_noisy}")


def test_video_dists() -> None:
    asyncio.run(_test_video_dists())
