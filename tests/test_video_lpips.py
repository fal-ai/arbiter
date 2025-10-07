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


async def _test_video_lpips() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            lpips = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_learned_perceptual_image_patch_similarity",
                alias="video_lpips",
            )

            # Test identical videos - should have very low LPIPS
            lpips_identical = lpips.calculate((video, video))
            assert all([abs(d) < 1e-5 for d in lpips_identical])

            # Test with slightly noisy video
            noisy_video = video + torch.randn_like(video) * 0.1
            noisy_video.clamp_(0, 1)
            lpips_noisy = lpips.calculate((video, noisy_video))
            assert all([d > 0.0 for d in lpips_noisy])

            # Test with very noisy video - should have higher LPIPS
            very_noisy_video = video + torch.randn_like(video) * 0.3
            very_noisy_video.clamp_(0, 1)
            lpips_very_noisy = lpips.calculate((video, very_noisy_video))
            assert all(
                [lpips_very_noisy[i] > lpips_noisy[i] for i in range(len(lpips_noisy))]
            )

            print(f"LPIPS identical: {lpips_identical}")
            print(f"LPIPS noisy: {lpips_noisy}")
            print(f"LPIPS very noisy: {lpips_very_noisy}")


def test_video_lpips() -> None:
    asyncio.run(_test_video_lpips())
