"""Standalone TTS service for thin voice clients that offload speech synthesis.

Run on a host that can run Kokoro fast (e.g. EM); thin clients too weak for local Kokoro (a Pi-4 is
~16s/utterance) POST reply text and get back a WAV in Kokoro's good voice (~1s on x86). See
tts_service/server.py. Symmetric with stt_service/ (the STT offload).
"""
