from collections import Counter

from ..annotations import (
    InputMediaTypeName,
    MeasurementInputType,
    MediaTypeName,
    MultiMediaTypeName,
)

__all__ = [
    "to_multi_media_type",
    "is_compatible_multi_media_type",
    "get_media_type_from_input",
    "arrange_input",
]


def to_multi_media_type(media_type: InputMediaTypeName) -> MultiMediaTypeName:
    """
    Convert a single media type to a multi-media type.
    """
    if isinstance(media_type, tuple):
        return media_type
    return (media_type,)


def is_compatible_multi_media_type(
    multi_media_type: MultiMediaTypeName,
    media_type: InputMediaTypeName,
) -> bool:
    """
    Check if a media type is compatible with a multi-media type.
    """
    other_media_type = to_multi_media_type(media_type)
    # Just ensure the count of each media type is the same, order will be fudged
    counts = Counter(multi_media_type)
    for m in other_media_type:
        counts[m] -= 1
        if counts[m] < 0:
            return False
    return all(count == 0 for count in counts.values())


def get_media_type_from_input(
    input: MeasurementInputType,
) -> MultiMediaTypeName:
    """
    Get the media type from the input.
    """
    type_names: list[MediaTypeName] = []
    for i in input:
        if isinstance(i, str):
            type_names.append("text")
        else:
            if i.ndim <= 2:
                type_names.append("audio")
            elif i.ndim == 3:
                type_names.append("image")
            elif i.ndim == 4:
                if i.shape[0] == 1:
                    type_names.append("image")
                else:
                    type_names.append("video")
            else:
                raise ValueError(f"Invalid input type: {type(i)}")
    return tuple(type_names)


def arrange_input(
    input: MeasurementInputType,
    media_type: MultiMediaTypeName,
) -> MeasurementInputType:
    """
    Arrange the input for a measurement.
    """
    input_media_type = get_media_type_from_input(input)
    if not is_compatible_multi_media_type(media_type, input_media_type):
        raise ValueError(
            f"Input media types {input_media_type} are not compatible with measurement media type {media_type}."
        )

    index_bag = list(range(len(input)))
    arranged_input: list[MeasurementInputType] = []
    remaining_input_type = media_type

    while len(remaining_input_type) > 0:
        next_input_type = remaining_input_type[0]
        found = False

        for i in index_bag:
            input_type = input_media_type[i]
            input_value = input[i]
            if input_type == next_input_type:
                arranged_input.append(input_value)
                index_bag = [j for j in index_bag if j != i]
                remaining_input_type = remaining_input_type[1:]
                found = True
                break

        if not found:
            raise ValueError(f"Input type {next_input_type} not found in input.")

    return tuple(arranged_input)
