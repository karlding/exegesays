from exegesays.mandarin import extract_english, extract_mandarin


def test_pure_english_is_ignored() -> None:
    assert extract_mandarin("hello there", "en") == ""


def test_pure_mandarin_is_kept() -> None:
    assert extract_mandarin("你好，世界", "zh") == "你好，世界"


def test_english_inside_a_mandarin_sentence_is_removed() -> None:
    assert extract_mandarin("我想去 the store 买咖啡", "zh") == "我想去买咖啡"


def test_digit_next_to_chinese_is_kept() -> None:
    assert extract_mandarin("3个", "zh") == "3个"


def test_japanese_kana_drops_the_utterance() -> None:
    assert extract_mandarin("今日はいい天気", "ja") == ""


def test_hangul_drops_the_utterance() -> None:
    assert extract_mandarin("안녕하세요", "ko") == ""


def test_cantonese_tag_still_keeps_chinese() -> None:
    assert extract_mandarin("开放时间早上九点至下午五点", "yue") == (
        "开放时间早上九点至下午五点"
    )
    assert extract_mandarin("我今日去街", "<|yue|>") == "我今日去街"


def test_special_tokens_are_stripped() -> None:
    assert extract_mandarin("<|zh|><|NEUTRAL|><|Speech|>你好", "") == "你好"


def test_mandarin_inside_an_english_utterance_is_kept() -> None:
    assert extract_mandarin("I met 王伟 yesterday", "en") == "王伟"


def test_pure_english_is_kept_in_english_mode() -> None:
    assert extract_english("hello there", "en") == "hello there"


def test_pure_mandarin_is_ignored_in_english_mode() -> None:
    assert extract_english("你好，世界", "zh") == ""


def test_chinese_inside_english_is_removed() -> None:
    assert extract_english("I met 王伟 yesterday", "en") == "I met yesterday"
