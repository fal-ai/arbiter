# Generates varied and challenging prompts for image generation
import argparse
import asyncio
import csv
import json
import os
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import fal_client
import tqdm

CATEGORIES_SUBCATEGORIES = {
    "people": [
        "headshot",
        "modeling photoshoot",
        "action shot",
        "slice of life",
    ],
    "animals": [
        "household pet",
        "common outdoor animal",
        "exotic animal",
        "alien animal",
    ],
    "food": [
        "fresh produce",
        "restaurant meal",
        "home-cooked meal",
        "wrapped food product",
    ],
    "objects": [
        "common household object",
        "exotic object",
        "alien object",
        "product packaging",
    ],
    "vehicles": [
        "land vehicle",
        "marine vehicle",
        "aircraft",
        "spacecraft",
    ],
    "architecture": [
        "modern building",
        "ancient building",
        "futuristic building",
        "industrial building",
    ],
    "places": [
        "cityscape",
        "countryside",
        "beach",
        "forest",
    ],
}

SYSTEM_PROMPT = """You are a helpful assistant that generates varied and meticulously crafted prompts for image generation.

You will be given a category and subcategory to generate a prompt for. You will also be given a list of prompts that are already in the dataset. You should generate a unique prompt with elements that are not already over-represented in the dataset.

Your prompts should be descriptive, natural language describing a visual scene. You should begin by focusing on a singular subject, describing it/them in detail, then move on to describing what it/they are doing, the environment, context, and finally describe feeling and the overall mood. Focus on specific colors and textures, and relationships between objects and the environment. Use up to 150 words.

Your prompt should not only vary in the subject, but also in the environment, context, and mood. Try to keep balanced blends of artificial and natural elements, lighting and colors, representing a wide variety of potential scenes in a wide variety of contexts. Images can be exciting or bland, grounded or fantastical, staged professionally or candid - try to represent all of these and more.

Respond only with the prompt, no other text."""

SYSTEM_PROMPT_MAKE_TAGS = """You are a helpful assistant that takes plain-english image generation prompts turns them into a shorter list of tags.

Given a prompt that describes a visual scene, generate a list of comma-separated tags that describe the scene. You can use short multi-word sentences, but keep them concise. Focus on the most important elements of the scene first, and gradually reference the less important elements. You must use no more than 75 total tokens, so be concise while still maintaining the essence of the original prompt.

Respond only with the list of tags, no other text."""

GPT5_GENERATE_MODEL = "openai/gpt-5-mini"
GOOGLE_GENERATE_MODEL = "google/gemini-2.0-flash-001"
TAG_MODEL = "google/gemini-2.0-flash-001"

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


def prompt_already_exists(
    prompt: str,
    existing_prompts: list[str],
) -> bool:
    """
    Checks if a prompt is already in the dataset.
    """
    prompt_word_counts = Counter(prompt.split())
    prompt_words = set(prompt_word_counts.keys())
    total_prompt_words = sum(prompt_word_counts.values())

    for existing_prompt in existing_prompts:
        existing_prompt_word_counts = Counter(existing_prompt.split())
        existing_prompt_words = set(existing_prompt_word_counts.keys())
        total_existing_words = sum(existing_prompt_word_counts.values())

        # If words are 90% similar, return True
        # That means that 90% of the words (non-unique) are the same
        # This is a very rough heuristic, but it's good enough for our purposes
        intersecting_words = prompt_words.intersection(existing_prompt_words)
        intersecting_word_count = sum(
            prompt_word_counts[word] for word in intersecting_words
        )
        existing_intersecting_word_count = sum(
            existing_prompt_word_counts[word] for word in intersecting_words
        )

        if (
            intersecting_word_count / total_prompt_words > 0.9
            or existing_intersecting_word_count / total_existing_words > 0.9
        ):
            return True

    return False


async def generate_tags(
    prompt: str,
    progress_bar: tqdm.tqdm,
) -> list[str]:
    """
    Generates a list of tags for a given prompt.
    """
    num_tries = 0
    async with get_client() as client:
        params = {
            "model": TAG_MODEL,
            "system_prompt": SYSTEM_PROMPT_MAKE_TAGS,
            "prompt": prompt,
        }

        response = await client.subscribe(
            "fal-ai/any-llm",
            params,
        )

        if "output" in response:
            progress_bar.update(1)
            return response["output"]

        if num_tries > 10:
            raise Exception("Failed to generate tags")

        await asyncio.sleep(0.5)
        num_tries += 1


async def generate_prompt(
    category: str,
    subcategory: str,
    existing_prompts: list[str] = [],
    use_gpt5: bool = False,
) -> str:
    """
    Generates a single prompt for a given category and subcategory.
    """
    params = {
        "model": GPT5_GENERATE_MODEL if use_gpt5 else GOOGLE_GENERATE_MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": f"Category: {category}\nSubcategory: {subcategory}\nExisting prompts: {existing_prompts}",
    }

    result_prompt = None

    async with get_client() as client:
        while result_prompt is None:
            response = await client.subscribe(
                "fal-ai/any-llm",
                params,
            )
            if "output" in response and not prompt_already_exists(
                response["output"], existing_prompts
            ):
                result_prompt = response["output"]
                break

            await asyncio.sleep(0.5)

    return result_prompt


async def generate_prompts_for_category(
    category: str,
    subcategory: str,
    num_prompts: int,
    progress_bar: tqdm.tqdm,
    use_gpt5: bool = False,
) -> list[str]:
    """
    Generates a list of prompts for a given category and subcategory.
    """
    prompts = []
    while len(prompts) < num_prompts:
        prompt = await generate_prompt(category, subcategory, prompts, use_gpt5)
        prompts.append(prompt)
        progress_bar.update(1)

    return prompts


async def generate_dataset(
    output_file: str,
    num_prompts_per_category: int = 1,
    repo_id: str | None = None,
    repo_private: bool = True,
    use_gpt5: bool = False,
) -> None:
    """
    Generates a dataset of prompts and tags.
    """
    num_category_pairs = sum(
        len(subcategories) for subcategories in CATEGORIES_SUBCATEGORIES.values()
    )
    total_prompts = num_category_pairs * num_prompts_per_category
    progress_bar = tqdm.tqdm(total=total_prompts, desc="Generating prompts")
    tasks = []

    for category, subcategories in CATEGORIES_SUBCATEGORIES.items():
        for subcategory in subcategories:
            tasks.append(
                generate_prompts_for_category(
                    category,
                    subcategory,
                    num_prompts_per_category,
                    progress_bar,
                    use_gpt5,
                )
            )

    all_results = await asyncio.gather(*tasks)
    result_rows = []

    for category, subcategories in CATEGORIES_SUBCATEGORIES.items():
        for subcategory in subcategories:
            prompts = all_results.pop(0)
            result_rows.extend(
                [
                    {
                        "category": category,
                        "subcategory": subcategory,
                        "prompt": prompt,
                    }
                    for prompt in prompts
                ]
            )

    tag_progress_bar = tqdm.tqdm(total=len(result_rows), desc="Generating tags")
    all_tags = await asyncio.gather(
        *[generate_tags(row["prompt"], tag_progress_bar) for row in result_rows]
    )
    tag_progress_bar.close()

    for row, tags in zip(result_rows, all_tags):
        row["tags"] = tags

    _, ext = os.path.splitext(output_file)
    if ext == "":
        ext = ".jsonl"

    if ext == ".jsonl":
        with open(output_file, "w") as f:
            for row in result_rows:
                f.write(json.dumps(row) + "\n")
    elif ext == ".json":
        with open(output_file, "w") as f:
            json.dump(result_rows, f)
    elif ext == ".csv":
        with open(output_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
            writer.writeheader()
            writer.writerows(result_rows)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if repo_id is not None:
        from datasets import ClassLabel, Dataset, Features, Value

        category_label = ClassLabel(names=list(CATEGORIES_SUBCATEGORIES.keys()))
        subcategory_label = ClassLabel(
            names=list(
                {
                    subcategory
                    for subcategories in CATEGORIES_SUBCATEGORIES.values()
                    for subcategory in subcategories
                }
            )
        )

        def row_iterator() -> Iterator[dict[str, Any]]:
            for row in result_rows:
                yield {
                    "category": category_label.str2int(row["category"]),
                    "subcategory": subcategory_label.str2int(row["subcategory"]),
                    "prompt": row["prompt"],
                    "tags": row["tags"],
                }

        dataset = Dataset.from_generator(
            row_iterator,
            features=Features(
                {
                    "category": category_label,
                    "subcategory": subcategory_label,
                    "prompt": Value("string"),
                    "tags": Value("string"),
                }
            ),
        )

        dataset.push_to_hub(repo_id, private=repo_private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image prompts and optionally push to hub."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (.jsonl, .json, .csv)",
        default="prompts.jsonl",
    )
    parser.add_argument(
        "--num-prompts-per-category",
        type=int,
        default=1,
        help="Number of prompts to generate per category",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace Hub repo id to push dataset to",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Set repo as public on push (default is private)",
    )
    parser.add_argument(
        "--use-gpt5",
        action="store_true",
        help="Use GPT-5 mini instead of Gemini 2.0 Flash",
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

    asyncio.run(
        generate_dataset(
            output_file=args.output_file,
            repo_id=args.repo_id,
            repo_private=not args.public,
            num_prompts_per_category=args.num_prompts_per_category,
            use_gpt5=args.use_gpt5,
        )
    )
