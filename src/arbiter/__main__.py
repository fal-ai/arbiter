import csv
import enum
import io
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import click

from arbiter.measurements import Measurement, MeasurementGroup
from arbiter.util import debug_logger, get_media_type_from_path


@contextmanager
def cli_context(debug: bool = False) -> Iterator[None]:
    if debug:
        with debug_logger():
            yield
    else:
        yield


@click.group()
def cli() -> None:
    pass


class OutputFormat(enum.Enum):
    JSON = enum.auto()
    CSV = enum.auto()
    TSV = enum.auto()
    HTML = enum.auto()
    MARKDOWN = enum.auto()
    TABLE = enum.auto()
    PLAIN = enum.auto()


class InputMediaType(enum.Enum):
    IMAGE = enum.auto()
    VIDEO = enum.auto()
    AUDIO = enum.auto()
    TEXT = enum.auto()


def format_output(
    outputs: list[dict[str, Any]],
    output_format: OutputFormat,
) -> str:
    """
    Formats a list of outputs into a string.
    """
    if output_format == OutputFormat.TABLE:
        try:
            import tabulate
        except ImportError:
            output_format = OutputFormat.PLAIN
    if output_format == OutputFormat.JSON:
        return json.dumps(outputs)
    elif output_format == OutputFormat.CSV:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)
        return buffer.getvalue()
    elif output_format == OutputFormat.TSV:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=outputs[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(outputs)
        return buffer.getvalue()
    elif output_format == OutputFormat.HTML:
        html = "<table><thead><tr>"
        for key in outputs[0].keys():
            html += f"<th>{key}</th>"
        html += "</tr></thead><tbody>"
        for output in outputs:
            html += "<tr>"
            for key in output.keys():
                html += f"<td>{output[key]}</td>"
        html += "</tbody></table>"
        return html
    elif output_format == OutputFormat.MARKDOWN:
        markdown = "|"
        for key in outputs[0].keys():
            markdown += f" {key} |"
        markdown += "\n|"
        for key in outputs[0].keys():
            markdown += "-" * (len(key) + 2) + "|"
        markdown += "\n"
        for i, output in enumerate(outputs):
            markdown += "|"
            for key in output.keys():
                markdown += f" {output[key]} |"
            if i < len(outputs) - 1:
                markdown += "\n"
        return markdown
    elif output_format == OutputFormat.PLAIN:
        return "\n".join(
            [", ".join(list([str(key) for key in outputs[0].keys()]))]
            + [
                ", ".join([f'"{value}"' for value in output.values()])
                for output in outputs
            ]
        )
    elif output_format == OutputFormat.TABLE:
        import tabulate

        return tabulate.tabulate(
            outputs,
            headers="keys",
            tablefmt="grid",
        )
    else:
        raise ValueError(f"Invalid output format: {output_format}")


@cli.command("list")
@click.argument(
    "input_media_type",
    type=click.Choice(InputMediaType, case_sensitive=False),
    nargs=-1,
)
@click.option("--debug", is_flag=True)
@click.option(
    "--output-format",
    type=click.Choice(OutputFormat, case_sensitive=False),
    default=OutputFormat.TABLE,
)
def list_measures(
    input_media_type: list[InputMediaType] = [],
    debug: bool = False,
    output_format: OutputFormat = OutputFormat.TABLE,
) -> None:
    """
    Lists all measurements.
    """
    with cli_context(debug):
        if not input_media_type:
            measurements = Measurement.enumerate()
        else:
            measurements = Measurement.for_media_type(
                tuple([m.name.lower() for m in input_media_type])
            )
            measurements.extend(
                Measurement.for_media_type(
                    tuple([m.name.lower() for m in input_media_type]),
                    aggregate=True,
                )
            )

        outputs = []

        for measurement in measurements:
            outputs.append(
                {
                    "Name": measurement.name,
                    "Aliases": (
                        ", ".join(measurement.aliases) if measurement.aliases else ""
                    ),
                    "Media Type": ", ".join(measurement.media_type),
                    "Aggregate": measurement.aggregate,
                }
            )

        outputs.sort(key=lambda x: x["Name"])

        print(
            format_output(outputs, output_format),
        )


@cli.command("measure")
@click.argument("measurement_name")
@click.argument("inputs", nargs=-1)
@click.option(
    "--output-format",
    type=click.Choice(OutputFormat, case_sensitive=False),
    default=OutputFormat.TABLE,
)
@click.option("--debug", is_flag=True)
def measure(
    measurement_name: str,
    inputs: list[str],
    debug: bool = False,
    output_format: OutputFormat = OutputFormat.TABLE,
) -> None:
    """
    Measures a given measurement with the given inputs.
    """
    with cli_context(debug):
        measurement = Measurement.get(measurement_name)
        if measurement is None:
            print(f"Could not find measurement {measurement_name}")
            return
        measurement_instance = measurement()
        result = measurement_instance.calculate(tuple(inputs))
        print(
            format_output(
                [
                    {
                        "Measurement": measurement_name,
                        "Input": ", ".join(inputs),
                        "Output": result,
                    }
                ],
                output_format,
            )
        )


@cli.command("multimeasure")
@click.argument("inputs", nargs=-1)
@click.option("--include-pattern", type=str, multiple=True)
@click.option("--exclude-pattern", type=str, multiple=True)
@click.option(
    "--output-format",
    type=click.Choice(OutputFormat, case_sensitive=False),
    default=OutputFormat.TABLE,
)
@click.option("--debug", is_flag=True)
def multimeasure(
    inputs: list[str],
    include_pattern: list[str] = [],
    exclude_pattern: list[str] = [],
    debug: bool = False,
    output_format: OutputFormat = OutputFormat.TABLE,
) -> None:
    """
    Measures all measurements for the given inputs.
    """
    with cli_context(debug):
        input_media_type = tuple([get_media_type_from_path(i) for i in inputs])
        measurements = Measurement.for_media_type(input_media_type)

        if include_pattern:
            measurements = [
                m
                for m in measurements
                if any(re.match(pattern, m.name) for pattern in include_pattern)
            ]
        if exclude_pattern:
            measurements = [
                m
                for m in measurements
                if not any(re.match(pattern, m.name) for pattern in exclude_pattern)
            ]

        measurement_group = MeasurementGroup.from_measurements(measurements)
        result = measurement_group.calculate(tuple(inputs))
        print(
            format_output(
                [
                    {
                        "Measurement": measurement_name,
                        "Input": ", ".join(inputs),
                        "Output": measurement_result,
                    }
                    for measurement_name, measurement_result in result.items()
                ],
                output_format,
            )
        )


if __name__ == "__main__":
    cli()
