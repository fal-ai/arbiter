import asyncio

from arbiter.util import (
    debug_logger,
    get_test_measurement,
    maybe_download_file,
    read_media,
)

test_image_url = "https://storage.googleapis.com/falserverless/example_inputs/nano-banana-edit-input.png"


async def _test_image_mse() -> None:
    import torch

    with debug_logger():
        async with maybe_download_file(test_image_url, timeout=5.0) as image_path:
            image = read_media(image_path)
            mse = get_test_measurement(
                media_type=("image", "image"),
                unique_name="image_mean_squared_error",
                alias="mse",
            )
            assert mse.calculate((image, image)) == 0.0
            image_2 = torch.ones_like(image) - image
            error = mse.calculate((image, image_2))
            assert 0 < error < 1.0


def test_image_mse() -> None:
    asyncio.run(_test_image_mse())
