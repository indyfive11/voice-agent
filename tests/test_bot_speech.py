"""bot_speech — the live 'is Aria's TTS playing' flag and its ride on the /media/state poll.

The flag lets the brain refresh its duck watchdog during a long reply (which produces no incoming user
speech to refresh it), so the watchdog can't pop the ducked bed mid-narration. Producer side only — the
brain consumes `bot_speaking` on the poll.
"""

import asyncio
import types

import bot_speech
from brains.brain_client import FakeBrainClient
from config import build_media_state_provider


def test_defaults_false():
    bot_speech.set_bot_speaking(False)
    assert bot_speech.bot_speaking() is False


def test_set_true_then_false():
    bot_speech.set_bot_speaking(True)
    assert bot_speech.bot_speaking() is True
    bot_speech.set_bot_speaking(False)
    assert bot_speech.bot_speaking() is False


def test_coerces_to_bool():
    bot_speech.set_bot_speaking(1)
    assert bot_speech.bot_speaking() is True
    bot_speech.set_bot_speaking(0)
    assert bot_speech.bot_speaking() is False


def _provider(client):
    # Fresh provider per call → no stale 1s cache between the True/False cases.
    return build_media_state_provider(
        types.SimpleNamespace(brain_client=client, session_id="sess-test")
    )


def test_provider_carries_bot_speaking_true():
    client = FakeBrainClient()
    client.media_state_value = {"playing": True, "state": "playing"}
    bot_speech.set_bot_speaking(True)
    asyncio.run(_provider(client)())
    assert client.media_state_bot_speaking[-1] is True


def test_provider_carries_bot_speaking_false():
    client = FakeBrainClient()
    client.media_state_value = {"playing": True, "state": "playing"}
    bot_speech.set_bot_speaking(False)
    asyncio.run(_provider(client)())
    assert client.media_state_bot_speaking[-1] is False
