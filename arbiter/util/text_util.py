from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_distance_matrix(reference: str, hypothesis: str) -> NDArray:
    """
    Calculates the distance matrix between the reference and hypothesis texts.

    Shared between WER and MER.

    :param reference: The reference text.
    :param hypothesis: The hypothesis text.
    :return: The distance matrix.
    """
    import numpy as np

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    num_reference_words = len(reference_words)
    num_hypothesis_words = len(hypothesis_words)

    distance = np.zeros((num_reference_words + 1, num_hypothesis_words + 1))

    # Initialize the distance matrix
    for i in range(num_reference_words + 1):
        distance[i][0] = i
    for j in range(num_hypothesis_words + 1):
        distance[0][j] = j

    # Compute the distance matrix
    for i in range(1, num_reference_words + 1):
        for j in range(1, num_hypothesis_words + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = min(
                    distance[i - 1][j] + 1,  # Deletion
                    distance[i][j - 1] + 1,  # Insertion
                    distance[i - 1][j - 1] + 1,  # Substitution
                )

    return distance


def count_n_grams(sentence: str, ngram: int = 4) -> dict[tuple[str, ...], int]:
    """
    Counts the n-grams in the sentence.

    :param sentence: The sentence to count the n-grams in.
    :param ngram: The size of the n-grams to count.
    :return: A dictionary of n-grams and their counts.
    """
    return dict(
        Counter(
            tuple(sentence[i : i + ngram])  # type: ignore[misc]
            for i in range(len(sentence) - ngram + 1)
        )
    )
