import random
from typing import Literal

from ...annotations import ProcessedMeasurementInputType
from ..base import Measurement


class VideoImageMeasurementBase(Measurement):
    """
    A measurement that uses an image measurement to measure a video.
    """

    image_cls: type[Measurement]
    sampling_mode: Literal["all", "first", "last", "uniform", "random"] = "uniform"
    num_samples: int = 3

    def setup(self) -> None:
        """
        Get the sampling indices for the video.
        """
        self.image_measurement = self.image_cls()
        self.image_measurement.setup()

    def get_sampling_indices(self, num_frames: int) -> list[int]:
        """
        Get the sampling indices for the video.
        """
        import numpy as np

        if self.sampling_mode == "all":
            return list(range(num_frames))
        elif self.sampling_mode == "first":
            return [0]
        elif self.sampling_mode == "last":
            return [num_frames - 1]
        elif self.sampling_mode == "uniform":
            if self.num_samples == 0:
                return []
            elif self.num_samples == 1:
                return [0]
            else:
                indices = np.linspace(0, num_frames - 1, self.num_samples)
                return indices.astype(int).tolist()
        elif self.sampling_mode == "random":
            return [random.randint(0, num_frames) for _ in range(self.num_samples)]
        else:
            raise ValueError(f"Invalid sampling mode: {self.sampling_mode}")


class VideoImageMeasurement(VideoImageMeasurementBase):
    """
    A measurement that uses an image measurement to measure a video.
    """

    media_type = ("video",)

    def __call__(self, input: ProcessedMeasurementInputType) -> list[float]:
        """
        :param input: A tuple of a video as [f, c, h, w] float32 tensor in range [0, 1].
        :return: The result of the image measurement.
        """
        (video,) = input
        num_frames = video.shape[0]
        sampling_indices = self.get_sampling_indices(num_frames)
        frames = video[sampling_indices]
        results = [self.image_measurement.calculate((frame,)) for frame in frames]
        return results


class VideoImageComparisonMeasurement(VideoImageMeasurementBase):
    """
    A measurement that uses an image comparison measurement to measure a video.
    """

    media_type = ("video", "video")

    def __call__(self, input: ProcessedMeasurementInputType) -> float:
        """
        :param input: A tuple of two videos as [f, c, h, w] float32 tensors in range [0, 1].
        :return: The result of the image comparison measurement.
        """
        (video1, video2) = input
        num_frames = video1.shape[0]
        sampling_indices = self.get_sampling_indices(num_frames)
        frames1 = video1[sampling_indices]
        frames2 = video2[sampling_indices]
        results = [
            self.image_measurement.calculate((frame1, frame2))
            for frame1, frame2 in zip(frames1, frames2)
        ]
        return results


class VideoImageTextAlignmentMeasurement(VideoImageMeasurementBase):
    """
    A measurement that uses an image text alignment measurement to measure a video.
    """

    media_type = ("video", "text")

    def __call__(self, input: ProcessedMeasurementInputType) -> float:
        """
        :param input: A tuple of a video as [f, c, h, w] float32 tensor in range [0, 1] and a text.
        :return: The result of the image text alignment measurement.
        """
        (video, text) = input
        num_frames = video.shape[0]
        sampling_indices = self.get_sampling_indices(num_frames)
        frames = video[sampling_indices]
        results = [self.image_measurement.calculate((frame, text)) for frame in frames]
        return results
