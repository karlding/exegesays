import io

from exegesays.output import TranslationWriter


def test_writer_streams_then_newline() -> None:
    buf = io.StringIO()
    writer = TranslationWriter(buf)
    writer.feed("Hello")
    writer.feed(" world")
    writer.finish("Hello world")
    assert buf.getvalue() == "Hello world\n"


def test_writer_reset_starts_replacement_on_new_line() -> None:
    buf = io.StringIO()
    writer = TranslationWriter(buf)
    writer.feed("wrong")
    writer.reset()
    writer.feed("right")
    writer.finish("right")
    assert buf.getvalue() == "wrong\nright\n"


def test_writer_finish_replaces_cleaned_text() -> None:
    buf = io.StringIO()
    writer = TranslationWriter(buf)
    writer.feed("raw")
    writer.finish("cleaned")
    assert buf.getvalue() == "raw\ncleaned\n"


def test_writer_finish_without_streamed_tokens() -> None:
    buf = io.StringIO()
    writer = TranslationWriter(buf)
    writer.finish("later")
    assert buf.getvalue() == "later\n"
