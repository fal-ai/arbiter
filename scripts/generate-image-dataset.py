import argparse
import asyncio
import base64
import json
import os
import random
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import datasets
import fal_client
import httpx
import tqdm

CLIENT_LOCK = asyncio.Lock()
CLIENTS: list[fal_client.AsyncClient] = []


@asynccontextmanager
async def get_client() -> AsyncIterator[fal_client.AsyncClient]:
    """
    Gets a client from the pool.
    """
    global CLIENTS

    async with CLIENT_LOCK:
        while not CLIENTS:
            await asyncio.sleep(0.01)

        client = CLIENTS.pop()

    try:
        yield client
    finally:
        CLIENTS.append(client)


async def generate_for_prompt(
    model_id: str,
    prompt: str,
    output_dir: str,
    progress_bar: tqdm.tqdm | None = None,
    sync_mode: bool = False,
    **kwargs: Any,
) -> tuple[str, int]:
    """
    Generate an image for a given prompt.
    """
    async with get_client() as client:
        seed = random.randint(0, 2**32 - 1)
        params = {"prompt": prompt, "seed": seed, **kwargs}
        if sync_mode:
            params["sync_mode"] = True

        result = await client.subscribe(
            model_id,
            params,
        )

        if "image" not in result and "images" not in result:
            raise ValueError(f"Invalid result: {result}")

        if "image" in result:
            image_url = result["image"]["url"]
        else:
            image_url = result["images"][0]["url"]

        if image_url.startswith("data:image/"):
            image_data_parts = image_url.split(",")
            image_ext = image_data_parts[0].split(";")[0].split("/")[1]
            image_data = base64.b64decode(image_data_parts[1])
        else:
            image_ext = image_url.split(".")[-1]
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                image_data = response.content

        image_path = os.path.join(output_dir, f"{uuid.uuid4()}.{image_ext}")

        with open(image_path, "wb") as f:
            f.write(image_data)

        if progress_bar is not None:
            progress_bar.update(1)

        return image_path, seed


async def generate_for_prompts(
    model_id: str,
    prompts: List[str],
    output_dir: str,
    sync_mode: bool = False,
    **kwargs: Any,
) -> List[Tuple[str, int]]:
    """
    Generate images for a list of prompts.
    """
    progress_bar = tqdm.tqdm(total=len(prompts), desc="Generating images")
    tasks = [
        generate_for_prompt(
            model_id,
            prompt,
            output_dir,
            progress_bar=progress_bar,
            sync_mode=sync_mode,
            **kwargs,
        )
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks)
    progress_bar.close()
    return results


async def generate_dataset(
    repo_id: str,
    model_id: str,
    prompts: list[str],
    public: bool = False,
    sync_mode: bool = False,
    **kwargs: Any,
) -> datasets.Dataset:
    """
    Generate a dataset of images for a list of prompts.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths_seeds = await generate_for_prompts(
            model_id, prompts, temp_dir, sync_mode, **kwargs
        )
        dataset = datasets.Dataset.from_dict(
            {
                "image": [image_path for image_path, _ in image_paths_seeds],
                "seed": [seed for _, seed in image_paths_seeds],
                "prompt": prompts,
            }
        ).cast_column("image", datasets.Image())
        dataset.push_to_hub(repo_id, private=not public)
        return dataset


async def generate_dataset_from_prompt_dataset(
    prompt_dataset_repo_id: str,
    prompt_dataset_split: str,
    prompt_dataset_num_prompts: int | None,
    output_repo_id: str,
    model_id: str,
    public: bool = False,
    sync_mode: bool = False,
    **kwargs: Any,
) -> None:
    """
    Generate a dataset of images from a prompt dataset.
    """
    prompt_dataset = datasets.load_dataset(
        prompt_dataset_repo_id, split=prompt_dataset_split
    )
    prompt_dataset = prompt_dataset.shuffle()
    if prompt_dataset_num_prompts is not None:
        prompt_dataset = prompt_dataset.select(range(prompt_dataset_num_prompts))
    prompts = prompt_dataset["prompt"]
    await generate_dataset(
        output_repo_id, model_id, prompts, public, sync_mode, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image prompts and optionally push to hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo id to push dataset to",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="fal-ai/flux-1/schnell",
        help="Fal model id to use for generation",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        default="{}",
        help="JSON string of kwargs to pass to the model",
    )
    parser.add_argument(
        "--prompt-dataset-repo-id",
        type=str,
        default="fal/image-generation-prompts",
        help="HuggingFace Hub repo id of prompt dataset",
    )
    parser.add_argument(
        "--prompt-dataset-split",
        type=str,
        default="train",
        help="Split of prompt dataset to use",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of prompts to use from prompt dataset. If not provided, all prompts will be used.",
    )
    parser.add_argument(
        "--sync-mode",
        action="store_true",
        help="Use sync mode for generation",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Set repo as public on push (default is private)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of clients to use for parallel generation",
    )

    args = parser.parse_args()

    for _ in range(args.num_clients):
        CLIENTS.append(fal_client.AsyncClient())

    try:
        kwargs = json.loads(args.kwargs)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {args.kwargs}")

    asyncio.run(
        generate_dataset_from_prompt_dataset(
            output_repo_id=args.repo_id,
            prompt_dataset_repo_id=args.prompt_dataset_repo_id,
            prompt_dataset_split=args.prompt_dataset_split,
            prompt_dataset_num_prompts=args.num_images,
            model_id=args.model_id,
            public=args.public,
            sync_mode=args.sync_mode,
            **kwargs,
        )
    )
