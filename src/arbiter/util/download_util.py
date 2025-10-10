import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx


def get_filename_from_url(url: str) -> str:
    """
    Get the filename from a URL.
    """
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


async def download_file(
    url: str,
    output_path: str,
    timeout: float | None = None,
    retries: int = 3,
    retry_delay: float = 0.5,
) -> None:
    """
    Download a file from a URL.
    """
    for _ in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", url) as response:
                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
            return
        except Exception as e:
            print(f"Error downloading file {url}: {e}")
            await asyncio.sleep(retry_delay)
    raise Exception(f"Failed to download file {url} after {retries} retries")


@asynccontextmanager
async def maybe_download_file(
    url: str, timeout: float | None = None
) -> AsyncIterator[str]:
    """
    Downloads a file from a URL if it doesn't exist locally.
    """
    if url.startswith("http://") or url.startswith("https://"):
        filename = get_filename_from_url(url)
        _, extension = os.path.splitext(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file_path = temp_file.name
            await download_file(url, temp_file_path, timeout)
            yield temp_file_path
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    elif os.path.exists(url):
        yield url
    else:
        raise ValueError(f"File {url} does not exist and is not a URL")
