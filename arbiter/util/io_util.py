from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..constants import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray

    from ..annotations import MediaType, MediaTypeName, ProcessedMeasurableType

__all__ = [
    "resize_and_center_crop",
    "resize_stretch",
    "to_bchw_tensor",
    "process_media",
    "get_media_type_from_path",
    "read_media",
    "read_text",
]


def resize_stretch(
    video: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Resizes and stretches an image/video tensor to the specified width and height.
    """
    import torch.nn.functional as F

    squeeze = False
    if video.ndim == 3:
        video = video.unsqueeze(0)  # Add batch dimension if missing
        squeeze = True
    elif video.ndim != 4:
        raise ValueError("Input tensor must be of shape (T, C, H, W) or (C, H, W)")

    video = F.interpolate(
        video,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    if squeeze:
        video = video.squeeze(0)

    return video


def resize_and_center_crop(
    video: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Resizes and center-crops an image/video tensor to the specified width and height.

    This is a special case so we don't move between devices unnecessarily.
    """
    import torch.nn.functional as F

    squeeze = False
    if video.ndim == 3:
        squeeze = True
        video = video.unsqueeze(0)  # Add batch dimension if missing
    elif video.ndim != 4:
        raise ValueError("Input tensor must be of shape (T, C, H, W) or (C, H, W)")

    # Calculate center crop coordinates
    h, w = video.shape[2:4]

    # Resize if necessary
    if h != height or w != width:
        scale = max(height / h, width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        video = F.interpolate(
            video,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

        x_offset = (new_w - width) // 2
        y_offset = (new_h - height) // 2

        video = video[:, :, y_offset : y_offset + height, x_offset : x_offset + width]

    if squeeze:
        video = video.squeeze(0)

    return video


def to_bchw_tensor(media: MediaType) -> torch.Tensor:
    """
    Converts a media to a tensor of shape (B, C, H, W).

    :param media: The media to convert.
    :return: A tensor of shape (B, C, H, W).
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(media, torch.Tensor):
        if media.dtype is torch.uint8:
            media = media.float() / 255.0
        if media.ndim == 3:
            return media.unsqueeze(0)
        return media
    elif isinstance(media, Image.Image):
        media = np.array(media)
        media = media.astype(np.float32) / 255.0
        media = media.transpose(2, 0, 1)  # H W C -> C H W
        return torch.from_numpy(media).unsqueeze(0)
    elif isinstance(media, list):
        return torch.stack([to_bchw_tensor(f) for f in media])
    elif isinstance(media, np.ndarray):
        if media.dtype is np.uint8:
            media = media.astype(np.float32) / 255.0
        if media.ndim == 3:
            media = media[None, ...]
        return torch.from_numpy(media).permute(0, 3, 1, 2)  # B H W C -> B C H W
    else:
        raise ValueError(f"Unsupported media type: {type(media)}")


def process_media(media: MediaType) -> ProcessedMeasurableType:
    """
    Standardizes a media to a tensor of shape (B, C, H, W) (visual), a tensor of shape (C, T) (audio), or a string (text).
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(media, (torch.Tensor, np.ndarray)):
        if media.ndim in [1, 2]:
            # Audio
            if media.ndim == 1:
                media = media.unsqueeze(0)
            return media.unsqueeze(0)
        elif media.ndim == 3:
            return to_bchw_tensor(media)[0]
        elif media.ndim == 4:
            return to_bchw_tensor(media)
        else:
            raise ValueError(f"Unsupported media tensor shape: {media.shape}")
    elif isinstance(media, Image.Image):
        return to_bchw_tensor(media)[0]
    elif isinstance(media, list):
        return torch.stack([to_bchw_tensor(f) for f in media])
    elif isinstance(media, str):
        if os.path.isfile(media):
            return read_media(media)
        else:
            return media
    else:
        raise ValueError(f"Unsupported media type: {type(media)}")


def get_media_type_from_path(path: str) -> MediaTypeName:
    """
    Gets the media type from the path.
    """
    _, ext = os.path.splitext(path)
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    else:
        raise ValueError(f"Unsupported media type: {path}")


def read_media(path: str) -> torch.Tensor | str:
    """
    Reads a media file and returns a tensor of shape (B, C, H, W) (visual), a tensor of shape (C, T) (audio), or a string (text).

    :param path: The path to the media file.
    :return: A tensor of shape (B, C, H, W) (visual), a tensor of shape (C, T) (audio), or a string (text).
    """
    media_type = get_media_type_from_path(path)
    if media_type == "video":
        return read_video(path)
    elif media_type == "image":
        return read_image(path)
    elif media_type == "audio":
        return read_audio(path)
    elif media_type == "text":
        return read_text(path)
    else:
        raise ValueError(f"Unsupported media type: {path}")


def read_text(path: str) -> str:
    """
    Reads a text file and returns the text.

    :param path: The path to the text file.
    :return: The text.
    """
    with open(path) as f:
        return f.read()


def read_video(path: str) -> torch.Tensor:
    """
    Reads a video file and returns a tensor of shape (B, C, H, W).

    :param path: The path to the video file.
    :return: A tensor of shape (B, C, H, W).
    """
    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(path)

    frames: list[NDArray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    video = np.array(frames).astype(np.float32) / 255.0

    return torch.from_numpy(video).permute(0, 3, 1, 2)  # B H W C -> B C H W


def read_image(path: str) -> torch.Tensor:
    """
    Reads an image file and returns a tensor of shape (C, H, W).

    :param path: The path to the image file.
    :return: A tensor of shape (C, H, W).
    """
    import cv2
    import numpy as np
    import torch

    image = cv2.imread(path)
    assert image is not None, f"Failed to read image from {path}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)  # H W C -> C H W


def read_audio(path: str) -> torch.Tensor:
    """
    Reads an audio file and returns a tensor of shape (C, T).

    :param path: The path to the audio file.
    :return: A tensor of shape (C, T).
    """
    raise NotImplementedError("Audio reading is not implemented yet.")
