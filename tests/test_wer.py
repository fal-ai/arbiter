from arbiter.util import get_test_measurement


def test_wer() -> None:
    wer = get_test_measurement(
        media_type=("text", "text"),
        unique_name="word_error_rate",
        alias="wer",
    )
    assert wer.calculate(("Hello, world!", "Hello, world!")) == 0.0
    assert wer.calculate(("Hello, world!", "Hello, planet!")) == 0.5
