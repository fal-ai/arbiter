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


async def _test_video_ssim() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            ssim = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_structural_similarity_index_measure",
                alias="video_ssim",
            )

            # Test identical videos - should have SSIM = 1.0
            ssim_identical = ssim.calculate((video, video))
            assert all([abs(d - 1.0) < 1e-5 for d in ssim_identical])

            # Test with slightly noisy video - should have lower SSIM
            noisy_video = video + torch.randn_like(video) * 0.05
            noisy_video.clamp_(0, 1)
            ssim_noisy = ssim.calculate((video, noisy_video))
            assert all([d < 1.0 and d > 0.5 for d in ssim_noisy])

            # Test with very noisy video - should have even lower SSIM
            very_noisy_video = video + torch.randn_like(video) * 0.3
            very_noisy_video.clamp_(0, 1)
            ssim_very_noisy = ssim.calculate((video, very_noisy_video))
            assert all(
                [ssim_very_noisy[i] < ssim_noisy[i] for i in range(len(ssim_noisy))]
            )

            print(f"SSIM identical: {ssim_identical}")
            print(f"SSIM noisy: {ssim_noisy}")
            print(f"SSIM very noisy: {ssim_very_noisy}")


def test_video_ssim() -> None:
    asyncio.run(_test_video_ssim())
