"""stt_tuning — env-driven faster-whisper decoding levers, with a strict safe-no-op default.

The contract that matters: an UNCONFIGURED install must transcribe byte-identically to before (empty kwargs),
and each knob must reach the model only when explicitly set. A bad value is skipped, never raised.
"""

import functools

import stt_tuning


def test_no_env_is_empty_noop():
    # The portability invariant: nothing set ⇒ no kwargs ⇒ pure library defaults.
    assert stt_tuning.decoding_kwargs(env={}) == {}


def test_parses_each_type():
    env = {
        "STT_BEAM_SIZE": "8",
        "STT_BEST_OF": "3",
        "STT_TEMPERATURE": "0.0",
        "STT_CONDITION_ON_PREVIOUS": "0",
        "STT_NO_SPEECH_THRESHOLD": "0.5",
        "STT_LOGPROB_THRESHOLD": "-1.0",
        "STT_VAD_FILTER": "1",
        "STT_INITIAL_PROMPT": "builder-test hello.txt",
        "STT_HOTWORDS": "builder-test",
    }
    kw = stt_tuning.decoding_kwargs(env=env)
    assert kw == {
        "beam_size": 8,
        "best_of": 3,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -1.0,
        "vad_filter": True,
        "initial_prompt": "builder-test hello.txt",
        "hotwords": "builder-test",
    }


def test_only_set_knobs_appear():
    kw = stt_tuning.decoding_kwargs(env={"STT_CONDITION_ON_PREVIOUS": "0"})
    assert kw == {"condition_on_previous_text": False}  # nothing else leaks in


def test_bool_truthy_and_falsey():
    assert stt_tuning.decoding_kwargs(env={"STT_VAD_FILTER": "true"}) == {"vad_filter": True}
    assert stt_tuning.decoding_kwargs(env={"STT_VAD_FILTER": "off"}) == {"vad_filter": False}
    assert stt_tuning.decoding_kwargs(env={"STT_VAD_FILTER": "False"}) == {"vad_filter": False}


def test_blank_value_is_skipped():
    assert stt_tuning.decoding_kwargs(env={"STT_BEAM_SIZE": "   "}) == {}


def test_invalid_value_is_skipped_not_raised():
    # A typo'd int must not crash the service — it's dropped with a warning.
    assert stt_tuning.decoding_kwargs(env={"STT_BEAM_SIZE": "five"}) == {}


def test_whispercpp_form_fields_subset():
    # whisper-server HTTP only takes temperature + prompt (its name for initial_prompt).
    env = {"STT_TEMPERATURE": "0.2", "STT_INITIAL_PROMPT": "hello.txt", "STT_BEAM_SIZE": "8"}
    assert stt_tuning.whispercpp_form_fields(env=env) == {"temperature": "0.2", "prompt": "hello.txt"}


def test_whispercpp_form_fields_empty_by_default():
    assert stt_tuning.whispercpp_form_fields(env={}) == {}


class _FakeModel:
    def transcribe(self, audio, *, language=None, **kwargs):
        return ("captured", {"language": language, **kwargs})


class _FakeSvc:
    def __init__(self):
        self._model = _FakeModel()


def test_wrap_whisper_service_noop_when_unset():
    svc = _FakeSvc()
    stt_tuning.wrap_whisper_service(svc, env={})
    # Untouched ⇒ still the plain bound method, NOT a partial; a bare call carries no extra kwargs.
    assert not isinstance(svc._model.transcribe, functools.partial)
    _tag, got = svc._model.transcribe("audio", language="en")
    assert got == {"language": "en"}


def test_wrap_whisper_service_injects_kwargs():
    svc = _FakeSvc()
    stt_tuning.wrap_whisper_service(svc, env={"STT_BEAM_SIZE": "8", "STT_CONDITION_ON_PREVIOUS": "0"})
    # pipecat-style call passes only language; our levers must ride along.
    _tag, got = svc._model.transcribe("audio", language="en")
    assert got == {"language": "en", "beam_size": 8, "condition_on_previous_text": False}
    assert isinstance(svc._model.transcribe, functools.partial)


def test_wrap_whisper_service_handles_missing_model():
    class NoModel:
        pass

    # Must not raise if the service has no faster-whisper model (e.g. a non-whisper backend).
    stt_tuning.wrap_whisper_service(NoModel(), env={"STT_BEAM_SIZE": "8"})
