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


async def _test_video_vmaf() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            vmaf = get_test_measurement(
                media_type=("video", "video"),
                unique_name="video_multi_method_assessment_fusion",
                alias="vmaf",
            )

            # Test 1: Identical videos should have high VMAF score for all frames
            vmaf_identical = vmaf.calculate((video, video))
            assert all([f > 95 for f in vmaf_identical])

            # Test 2: Add small noise - VMAF should be high but < 100
            # For simplicity, we'll use the same video but add noise to simulate degradation
            noise = torch.randn_like(video) * 0.1
            noisy_video = torch.clamp(video + noise, 0, 1)
            vmaf_noise = vmaf.calculate((video, noisy_video))
            assert all(
                [vmaf_noise[i] < vmaf_identical[i] for i in range(len(vmaf_noise))]
            ), f"Expected VMAF for noisy video to be less than identical, got {vmaf_noise} vs {vmaf_identical}"

            print(f"VMAF identical: {vmaf_identical}")
            print(f"VMAF with noise: {vmaf_noise}")


def test_video_vmaf() -> None:
    asyncio.run(_test_video_vmaf())
