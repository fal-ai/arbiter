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


async def _test_video_mse() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            mse = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_mean_squared_error",
                alias="video_mse",
            )

            # Test identical videos - should have MSE = 0
            mse_identical = mse.calculate((video, video))
            assert all([abs(d) < 1e-7 for d in mse_identical])

            # Test with slightly different video
            noisy_video = video + torch.randn_like(video) * 0.1
            noisy_video.clamp_(0, 1)
            mse_noisy = mse.calculate((video, noisy_video))
            assert all([d > 0.0 for d in mse_noisy])

            # Test with very different video - should have higher MSE
            very_noisy_video = video + torch.randn_like(video) * 0.3
            very_noisy_video.clamp_(0, 1)
            mse_very_noisy = mse.calculate((video, very_noisy_video))
            assert all(
                [mse_very_noisy[i] > mse_noisy[i] for i in range(len(mse_noisy))]
            )

            print(f"MSE identical: {mse_identical}")
            print(f"MSE noisy: {mse_noisy}")
            print(f"MSE very noisy: {mse_very_noisy}")


def test_video_mse() -> None:
    asyncio.run(_test_video_mse())
