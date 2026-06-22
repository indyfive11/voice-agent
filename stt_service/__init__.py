"""Standalone STT service for thin voice clients that offload speech-to-text.

Run on a fast host (e.g. EM); thin clients (a Pi-4 too slow for local Whisper) POST a wake-gated
utterance WAV and get text back. See stt_service/server.py.
"""
