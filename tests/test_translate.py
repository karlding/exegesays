from exegesays.context import (
    LOW_LOGPROB,
    looks_context_dependent,
    should_retry_with_context,
)


def test_pronouns_look_context_dependent() -> None:
    assert looks_context_dependent("那是什么")
    assert looks_context_dependent("他去了")
    assert looks_context_dependent("这些都要")


def test_plain_sentence_does_not_look_context_dependent() -> None:
    assert not looks_context_dependent("我想买一杯咖啡")
    assert not looks_context_dependent("你好")


def test_no_history_never_retries() -> None:
    assert should_retry_with_context("那是什么", -3.0, has_history=False) is False


def test_pronoun_with_history_retries() -> None:
    assert should_retry_with_context("那是什么", -0.2, has_history=True) is True


def test_low_logprob_with_history_retries() -> None:
    assert should_retry_with_context("我想买一杯咖啡", LOW_LOGPROB - 0.1, True) is True


def test_confident_plain_sentence_does_not_retry() -> None:
    assert should_retry_with_context("我想买一杯咖啡", -0.2, True) is False


def test_english_pronouns_look_context_dependent() -> None:
    assert looks_context_dependent("What is that", keep="en")
    assert looks_context_dependent("He left", keep="en")
    assert not looks_context_dependent("I want to buy coffee", keep="en")
