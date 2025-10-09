# Evaluates a prompt + image(s) dataset, assuming all images in
# a row are generated from the same prompt and should be compared
# against a baseline/reference/real image.
import sys

sys.path.insert(0, "..")

import csv
import json
import os
import tempfile
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

import numpy as np
from arbiter.util import (
    cyan,
    debug_logger,
    green,
    magenta,
    red,
    run_gpu_multiprocessing,
    yellow,
)
from datasets import Features, Image, load_dataset


def get_prompt_field_from_features(features: Features) -> str:
    """
    Gets the name of the prompt field from a list of fields.
    """
    for maybe_field_name in ["prompt", "text", "caption"]:
        for maybe_suffix in ["", "_text"]:
            maybe_field = maybe_field_name + maybe_suffix
            if maybe_field in features.keys():
                return maybe_field
    raise ValueError(f"No prompt field found in fields: {features.keys()}")


def get_image_fields_from_features(features: Features) -> list[str]:
    """
    Gets the names of the image fields from a list of fields.
    """
    return [field for field in features.keys() if isinstance(features[field], Image)]


def get_baseline_image_field_from_fields(fields: list[str]) -> str:
    """
    Gets the name of the baseline image field from a list of fields.
    """
    for maybe_field_name in [
        "baseline",
        "reference",
        "real",
        "ground",
        "ground_truth",
        "truth",
    ]:
        for maybe_suffix in ["", "_image"]:
            maybe_field = maybe_field_name + maybe_suffix
            if maybe_field in fields:
                return maybe_field
    raise ValueError(f"No baseline image field found in fields: {fields}")


def get_image_name_from_field(field: str) -> str:
    """
    Gets the name of an image field.
    """
    if field.endswith("_image"):
        return field[:-6]
    return field


def summarize_measures(measures: list[float]) -> dict[str, float]:
    """
    Summarizes a list of measures.
    """
    if len(measures) == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "q1": None,
            "q3": None,
        }
    if len(measures) == 1:
        return {
            "mean": measures[0],
            "median": measures[0],
            "min": measures[0],
            "max": measures[0],
            "std": 0,
            "q1": measures[0],
            "q3": measures[0],
        }
    return {
        "mean": float(np.mean(measures)),
        "median": float(np.median(measures)),
        "min": float(np.min(measures)),
        "max": float(np.max(measures)),
        "std": float(np.std(measures)),
        "q1": float(np.percentile(measures, 25)),
        "q3": float(np.percentile(measures, 75)),
    }


def write_rows(dataset_id: str, name: str, rows: list[dict[str, Any]]) -> None:
    """
    Writes rows to a JSONL and CSV file.
    """
    jsonl_file = f"{dataset_id.replace('/', '_')}_{name}.jsonl"
    csv_file = f"{dataset_id.replace('/', '_')}_{name}.csv"
    with open(jsonl_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(csv_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(green(f"Wrote {len(rows)} rows to {jsonl_file} and {csv_file}"))


def adaptive_float_format(value: float) -> str:
    """
    Formats a float value for display.
    """
    if value > 1e5:
        return f"{value:0.3e}"
    elif value > 1e3:
        return f"{value:0.1f}"
    elif value > 1:
        return f"{value:0.2f}"
    elif value > 1e-3:
        return f"{value:0.3f}"
    else:
        return f"{value:0.3e}"


def process_dataset(
    dataset_id: str,
    split: str = "train",
    max_rows: int | None = None,
    shuffle: bool = False,
    max_workers: int = 4,
) -> None:
    """
    Processes a dataset of images.
    """
    # Initialize the groups
    dataset = load_dataset(dataset_id, split=split)
    if shuffle:
        dataset = dataset.shuffle()
    if max_rows is not None:
        dataset = dataset.select(range(max_rows))

    prompt_field = get_prompt_field_from_features(dataset.features)
    all_image_fields = get_image_fields_from_features(dataset.features)
    baseline_image_field = get_baseline_image_field_from_fields(all_image_fields)
    comparison_image_fields = [
        field for field in all_image_fields if field != baseline_image_field
    ]
    comparison_image_names = [
        get_image_name_from_field(field) for field in comparison_image_fields
    ]
    assert len(comparison_image_fields) > 0, "No comparison image fields found"

    with tempfile.TemporaryDirectory() as tmp_dir:

        def preprocess_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
            import torch

            id = uuid4().hex
            images_dir = os.path.join(tmp_dir, id)
            os.makedirs(images_dir, exist_ok=True)

            baseline_image = row.pop(baseline_image_field).float() / 255.0
            comparison_images = [
                row.pop(field).float() / 255.0 for field in comparison_image_fields
            ]

            baseline_image_path = os.path.join(images_dir, "baseline.pt")
            torch.save(baseline_image, baseline_image_path)
            comparison_images_paths = [
                os.path.join(images_dir, f"{name}.pt")
                for name in comparison_image_names
            ]

            for comparison_image, comparison_image_path in zip(
                comparison_images, comparison_images_paths
            ):
                torch.save(comparison_image, comparison_image_path)

            row["baseline_image_path"] = baseline_image_path
            row["comparison_images_paths"] = comparison_images_paths
            return row

        preprocessed_dataset = (
            dataset.with_format("torch")
            .map(preprocess_row, num_proc=max_workers)
            .select_columns(
                ["prompt", "baseline_image_path", "comparison_images_paths"]
            )
        )

        def process_row(
            prompt: str,
            baseline_image_path: str,
            comparison_images_paths: list[str],
            **kwargs: Any,
        ) -> dict[str, Any]:
            import torch
            from arbiter.measurements import MeasurementGroup

            # relevant measurement groups
            single_image_group = MeasurementGroup.get("image")
            image_comparison_group = MeasurementGroup.get("image_comparison")
            labeled_image_group = MeasurementGroup.get("labeled_image")

            current_device_id = torch.cuda.current_device()
            baseline_image = torch.load(
                baseline_image_path, f"cuda:{current_device_id}"
            )
            comparison_images = [
                torch.load(path, f"cuda:{current_device_id}")
                for path in comparison_images_paths
            ]
            results = {
                "reference": {},
                "reference_free": {},
                "semantic": {},
            }
            results["reference_free"]["baseline"] = single_image_group().calculate(
                (baseline_image,),
                processed=True,
            )
            results["semantic"]["baseline"] = labeled_image_group().calculate(
                (prompt, baseline_image),
                processed=True,
            )
            for comparison_image, comparison_image_name in zip(
                comparison_images, comparison_image_names
            ):
                results["reference_free"][
                    comparison_image_name
                ] = single_image_group().calculate((comparison_image,))
                results["semantic"][
                    comparison_image_name
                ] = labeled_image_group().calculate((prompt, comparison_image))
                results["reference"][
                    comparison_image_name
                ] = image_comparison_group().calculate(
                    (baseline_image, comparison_image)
                )

            return results

        # Calculate row-wise results
        results = run_gpu_multiprocessing(
            preprocessed_dataset,
            process_row,
            use_tqdm=True,
            total=len(preprocessed_dataset),
        )

        # Reduce row-results to a list of measurements
        result_rows = []
        for result in results:
            result_dict = {
                "baseline_reference_free": result["reference_free"]["baseline"],
                "baseline_semantic": result["semantic"]["baseline"],
            }
            for comparison_image_name in comparison_image_names:
                result_dict[f"{comparison_image_name}_reference"] = result["reference"][
                    comparison_image_name
                ]
                result_dict[f"{comparison_image_name}_reference_free"] = result[
                    "reference_free"
                ][comparison_image_name]
                result_dict[f"{comparison_image_name}_semantic"] = result["semantic"][
                    comparison_image_name
                ]
            result_rows.append(result_dict)

        write_rows(dataset_id, "rows", result_rows)

        # Calculate set-wise results
        baseline_image_urls = [
            row["baseline_image_path"] for row in preprocessed_dataset
        ]
        comparison_image_urls = {
            comparison_image_name: [
                row["comparison_images_paths"][i] for row in preprocessed_dataset
            ]
            for i, comparison_image_name in enumerate(comparison_image_names)
        }

        set_rows = [
            (baseline_image_urls, comparison_image_urls[comparison_image_name])
            for comparison_image_name in comparison_image_names
        ]

        def process_set(
            baseline_image_urls: list[str],
            comparison_image_urls: list[str],
            **kwargs: Any,
        ) -> dict[str, Any]:
            import torch
            from arbiter.measurements import MeasurementGroup

            current_device_id = torch.cuda.current_device()
            # relevant measurement groups
            image_set_comparison_group = MeasurementGroup.get("image_set_comparison")
            baseline_images = [
                torch.load(baseline_image_url, f"cuda:{current_device_id}")
                for baseline_image_url in baseline_image_urls
            ]
            comparison_images = [
                torch.load(comparison_image_url, f"cuda:{current_device_id}")
                for comparison_image_url in comparison_image_urls
            ]
            assert len(baseline_images) == len(comparison_images)

            def image_iterator():
                yield from zip(baseline_images, comparison_images)

            return image_set_comparison_group().calculate(image_iterator())

        set_results = run_gpu_multiprocessing(
            set_rows,
            process_set,
            use_tqdm=True,
            total=len(set_rows),
        )

        # Format set comparisons like row-wise results
        set_comparisons = {}
        for comparison_image_name, set_result in zip(
            comparison_image_names, set_results
        ):
            set_comparisons[comparison_image_name] = set_result

        # Now we only need to calculate and aggregate row-wise results
        measures = {}

        def add_to_measures(result: float | dict[str, float], *names: str) -> None:
            if isinstance(result, dict):
                for key, value in result.items():
                    add_to_measures(value, *names + (key,))
            else:
                current_dict = measures
                names = list(names)
                for name in names[:-1]:
                    if name not in current_dict:
                        current_dict[name] = {}
                    current_dict = current_dict[name]
                if names[-1] not in current_dict:
                    current_dict[names[-1]] = []
                current_dict[names[-1]].append(result)

        for result_row in result_rows:
            baseline_reference_free = result_row["baseline_reference_free"]
            add_to_measures(baseline_reference_free, "baseline")
            baseline_semantic = result_row["baseline_semantic"]
            add_to_measures(baseline_semantic, "baseline")
            for comparison_image_name in comparison_image_names:
                reference_free = result_row[f"{comparison_image_name}_reference_free"]
                add_to_measures(reference_free, comparison_image_name)
                semantic = result_row[f"{comparison_image_name}_semantic"]
                add_to_measures(semantic, comparison_image_name)
                reference = result_row[f"{comparison_image_name}_reference"]
                add_to_measures(reference, comparison_image_name)

        # Summarize measures
        all_measurement_names = set()
        all_set_measurement_names = set()
        for category in measures.keys():
            all_measurement_names.update(measures[category].keys())
        for category in set_comparisons.keys():
            all_set_measurement_names.update(set_comparisons[category].keys())
        summarized_measures = []

        for name in ["baseline"] + comparison_image_names:
            measurement_summary_dict = OrderedDict({"name": name})
            for measurement_name in sorted(all_measurement_names):
                if measurement_name not in measures[name]:
                    measurement_summary_dict[measurement_name] = None
                else:
                    measurement_summary_dict[measurement_name] = summarize_measures(
                        measures[name][measurement_name]
                    )
            for measurement_name in sorted(all_set_measurement_names):
                if (
                    name not in set_comparisons
                    or measurement_name not in set_comparisons[name]
                ):
                    measurement_summary_dict[measurement_name] = None
                else:
                    measurement_summary_dict[measurement_name] = set_comparisons[name][
                        measurement_name
                    ]
            summarized_measures.append(measurement_summary_dict)

        write_rows(dataset_id, "summary_detailed", summarized_measures)

        # Turn detailed measures into means or medians
        target_measure = "median"
        for measurement_summary_dict in summarized_measures:
            for measurement_name in measurement_summary_dict.keys():
                if (
                    isinstance(measurement_summary_dict[measurement_name], dict)
                    and target_measure in measurement_summary_dict[measurement_name]
                ):
                    measurement_summary_dict[measurement_name] = (
                        measurement_summary_dict[measurement_name][target_measure]
                    )

        write_rows(dataset_id, "summary", summarized_measures)

        # Print summary
        print(cyan(f"Summary for {dataset_id}"))
        baseline_dict = None
        for measurement_summary_dict in summarized_measures:
            name = measurement_summary_dict["name"]
            if name == "baseline":
                baseline_dict = measurement_summary_dict
            print(f"{magenta(name)}:")
            for measurement_name, measurement_value in measurement_summary_dict.items():
                if measurement_name == "name" or measurement_value is None:
                    continue
                if (
                    name != "baseline"
                    and baseline_dict.get(measurement_name) is not None
                    and isinstance(measurement_value, float)
                ):
                    delta = measurement_value - baseline_dict[measurement_name]
                    value_str = adaptive_float_format(measurement_value)
                    delta_str = adaptive_float_format(abs(delta))
                    delta_color_fn = green if delta > 0 else red
                    delta_sign = "+" if delta > 0 else "-"
                    print(
                        f"  {yellow(measurement_name)}: {value_str} ({delta_color_fn(delta_sign + delta_str)})"
                    )
                else:
                    if isinstance(measurement_value, float):
                        value_str = adaptive_float_format(measurement_value)
                    else:
                        value_str = str(measurement_value)
                    print(f"  {yellow(measurement_name)}: {value_str}")
            print()


# process_dataset("benjamin-paine/wan-2.2-image-cache-comparison", max_rows=2)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an image dataset with various metrics."
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="The HuggingFace dataset identifier (e.g., 'user/dataset').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Maximum number of rows to process."
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle the dataset before processing."
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of worker threads."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )

    args = parser.parse_args()
    context = debug_logger() if args.debug else nullcontext()
    with context:
        process_dataset(
            dataset_id=args.dataset_id,
            split=args.split,
            max_rows=args.max_rows,
            shuffle=args.shuffle,
            max_workers=args.max_workers,
        )
