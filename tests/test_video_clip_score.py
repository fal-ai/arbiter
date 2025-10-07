import asyncio

from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

# Use a short test video if available
test_video_url = "https://storage.googleapis.com/falserverless/web-examples/wan/t2v.mp4"


async def _test_video_clip_score() -> None:
    with debug_logger():
        async with maybe_download_file(test_video_url, timeout=10.0) as video_path:
            video = read_media(video_path)
            clip_score = get_test_measurement(
                media_type=("video", "text"),
                unique_name="video_clip_score",
                alias="video_clipscore",
            )

            # Test with relevant text
            relevant_text = "a woman walking on a street"
            clip_score_relevant = clip_score.calculate((video, relevant_text))
            assert all([f > 0.0 for f in clip_score_relevant])

            # Test with unrelated text - should have lower score
            unrelated_text = "a purple elephant eating pizza on the moon"
            clip_score_unrelated = clip_score.calculate((video, unrelated_text))
            assert all([f > 0.0 for f in clip_score_unrelated])

            print(f"CLIPScore relevant: {clip_score_relevant}")
            print(f"CLIPScore unrelated: {clip_score_unrelated}")


def test_video_clip_score() -> None:
    asyncio.run(_test_video_clip_score())
