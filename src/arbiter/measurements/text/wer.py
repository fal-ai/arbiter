from ...annotations import ProcessedMeasurementInputType
from ...util import get_distance_matrix
from ..base import Measurement


class WordErrorRate(Measurement):
    """
    Word Error Rate (WER) is a measure of the difference between a reference text and a hypothesis text.
    """

    media_type = ("text", "text")
    name = "word_error_rate"
    aliases = ["wer"]

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param media: A list of two strings, the reference and the hypothesis.
        :return: The word error rate.
        """
        (reference, hypothesis) = input
        distance = get_distance_matrix(reference, hypothesis)
        num_reference_words, num_hypothesis_words = distance.shape
        num_reference_words -= 1
        num_hypothesis_words -= 1
        num_substitutions = distance[num_reference_words][num_hypothesis_words]
        wer = num_substitutions / num_reference_words
        return wer.item()
