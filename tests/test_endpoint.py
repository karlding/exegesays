from exegesays.endpoint import CHECK_SAMPLES, TAIL_SAMPLES, SpeakerChange, SpeechCursor


def test_one_speaker_does_not_flush() -> None:
    state = SpeakerChange()
    samples = TAIL_SAMPLES
    assert state.note_tail(True, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(True, samples) is False


def test_new_speaker_then_sustained_other_voice_flushes() -> None:
    state = SpeakerChange()
    samples = TAIL_SAMPLES
    assert state.note_tail(True, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(False, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(False, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(False, samples) is True
    assert state.prefix_samples() == TAIL_SAMPLES


def test_brief_other_voice_does_not_flush() -> None:
    state = SpeakerChange()
    samples = TAIL_SAMPLES
    assert state.note_tail(True, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(False, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(True, samples) is False
    samples += CHECK_SAMPLES
    assert state.note_tail(False, samples) is False


def test_ready_for_check_waits_for_tail_and_interval() -> None:
    state = SpeakerChange()
    assert state.ready_for_check(TAIL_SAMPLES - 1) is False
    assert state.ready_for_check(TAIL_SAMPLES) is True
    state.mark_checked(TAIL_SAMPLES)
    assert state.ready_for_check(TAIL_SAMPLES + CHECK_SAMPLES - 1) is False
    assert state.ready_for_check(TAIL_SAMPLES + CHECK_SAMPLES) is True


def test_speech_cursor_counts_windows_after_the_seed() -> None:
    cursor = SpeechCursor()
    cursor.advance(True, 512, seed=1000)
    assert cursor.samples == 1000
    assert cursor.live is True
    cursor.advance(True, 512)
    assert cursor.samples == 1512
    cursor.advance(False, 512)
    assert cursor.live is False
    assert cursor.samples == 0


def test_speech_cursor_resync_replaces_a_drifted_count() -> None:
    cursor = SpeechCursor()
    cursor.advance(True, 512, seed=TAIL_SAMPLES)
    cursor.resync(TAIL_SAMPLES - 1)
    assert cursor.samples == TAIL_SAMPLES - 1
    cursor.advance(True, 512)
    assert cursor.samples == TAIL_SAMPLES - 1 + 512
