"""Audio feedback system for the Assistant OS mood system.

Provides TTS speech, mood-based audio cues, transition sounds, and
thinking/listening cues via subprocess calls to external players.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


class AudioFeedback:
    """Plays TTS and mood-based audio cues.

    All audio playback delegates to external commands (``ffplay``,
    ``aplay``, etc.) via ``subprocess``.  Tone filenames are loaded from
    the ``mood_face_tones.v1.json`` contract.
    """

    _MOOD_CENTROIDS: dict[str, tuple[float, float]] = {
        "neutral": (0.0, 0.1),
        "curious": (0.35, 0.4),
        "happy": (0.7, 0.45),
        "excited": (0.75, 0.8),
        "calm": (0.25, 0.08),
        "annoyed": (-0.4, 0.55),
        "frustrated": (-0.5, 0.75),
        "hurt": (-0.65, 0.2),
        "sad": (-0.6, 0.1),
    }

    def __init__(
        self,
        tts_command: str = "",
        audio_cues_dir: str = "",
        tones_data: dict | None = None,
    ) -> None:
        """Initialise the audio feedback system.

        When *tones_data* is ``None`` the contract ``mood_face_tones.v1.json``
        is loaded lazily via ``melm.contracts.validation.load_mood_face_tones``.
        """
        self.tts_command = tts_command
        self.audio_cues_dir = audio_cues_dir
        if tones_data is None:
            from melm.contracts.validation import load_mood_face_tones

            self._tones: dict[str, Any] = load_mood_face_tones()
        else:
            self._tones = tones_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> bool:
        """Speak *text* via the configured TTS command.

        Returns ``True`` when the command exits with return code 0.
        Silently catches all exceptions (missing command, timeouts, etc.).
        """
        if not self.tts_command or not text:
            return False
        try:
            argv = [*shlex.split(self.tts_command), text]
            comp = subprocess.run(
                argv,
                check=False,
                timeout=10.0,
                capture_output=True,
            )
            return comp.returncode == 0
        except Exception:
            return False

    def play_mood_cue(self, mood_id: str) -> bool:
        """Play the audio cue file for *mood_id*.

        Looks up the filename in ``mood_cues`` and delegates to
        ``_play_file``.  Returns ``False`` when the mood has no cue
        or the file is missing.
        """
        mood_cues: dict[str, Any] = self._tones.get("mood_cues", {})
        filename: str | None = mood_cues.get(mood_id)
        if not filename:
            return False
        return self._play_file(filename)

    def play_transition_cue(self, prev_mood_id: str, current_mood_id: str) -> bool:
        """Play a transition cue between two mood states.

        Selects the cue type based on valence direction and magnitude:
        ``transition_positive`` when valence rises,
        ``transition_sharp_negative`` when valence drops by >0.5,
        ``transition_negative`` when valence drops,
        ``transition_default`` otherwise.
        Returns ``False`` when the moods are identical or no file exists.
        """
        if prev_mood_id == current_mood_id:
            return False

        centroids = self._MOOD_CENTROIDS
        prev_val = centroids.get(prev_mood_id, (0.0, 0.1))[0]
        curr_val = centroids.get(current_mood_id, (0.0, 0.1))[0]

        drop = prev_val - curr_val
        if curr_val > prev_val:
            cue = "transition_positive"
        elif drop > 0.5:
            cue = "transition_sharp_negative"
        else:
            cue = "transition_negative"

        tones: dict[str, Any] = self._tones.get("tones", {})
        filename: str | None = tones.get(cue)
        if not filename:
            return False
        return self._play_file(filename)

    def play_thinking(self) -> bool:
        """Play the thinking audio cue."""
        tones: dict[str, Any] = self._tones.get("tones", {})
        filename: str | None = tones.get("thinking")
        if not filename:
            return False
        return self._play_file(filename)

    def play_listening(self) -> bool:
        """Play the listening audio cue."""
        tones: dict[str, Any] = self._tones.get("tones", {})
        filename: str | None = tones.get("listening")
        if not filename:
            return False
        return self._play_file(filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_file(self, filename: str) -> bool:
        """Play an audio file via subprocess.

        Tries ``ffplay`` (cross-platform) as the default player.
        Returns ``False`` on any error or missing file.
        """
        if not filename or not self.audio_cues_dir:
            return False
        path = Path(self.audio_cues_dir) / filename
        if not path.exists():
            return False
        try:
            comp = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", str(path)],
                check=False,
                timeout=5.0,
                capture_output=True,
            )
            return comp.returncode == 0
        except Exception:
            return False
