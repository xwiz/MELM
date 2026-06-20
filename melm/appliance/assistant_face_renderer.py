"""ASCII face renderer for the Assistant OS mood system.

Renders two-line ASCII face art from ``MoodState`` data and provides
emoji equivalents and transition animations.
"""

from __future__ import annotations

from typing import Any


class FaceRenderer:
    """Renders ASCII face art from ``MoodState``-like objects.

    Consumers pass a duck-typed mood object with ``mood_id``, ``valence``,
    ``arousal``, and ``is_listening`` attributes (compatible with
    ``assistant_mood_engine.MoodState``).  Face templates are loaded from
    the ``mood_faces.v1.json`` contract.
    """

    _EMOJI_MAP: dict[str, str] = {
        "neutral": "\U0001F610",
        "calm": "\U0001F60C",
        "happy": "\U0001F60A",
        "excited": "\U0001F929",
        "curious": "\U0001F914",
        "annoyed": "\U0001F612",
        "frustrated": "\U0001F624",
        "hurt": "\U0001F622",
        "sad": "\U0001F61E",
        "listening": "\U0001F442",
    }

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

    def __init__(self, faces_data: dict | None = None) -> None:
        """Initialise the renderer.

        When *faces_data* is ``None`` the contract ``mood_faces.v1.json``
        is loaded lazily via ``melm.contracts.validation.load_mood_faces``.
        """
        if faces_data is None:
            from melm.contracts.validation import load_mood_faces

            self._faces: dict[str, Any] = load_mood_faces()
        else:
            self._faces = faces_data
        self._valence_boundaries: list[float] = list(
            self._faces.get("valence_boundaries", [0.2, 0.5, 0.8])
        )
        self._face_templates: dict[str, Any] = dict(self._faces.get("faces", {}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, mood_state: Any) -> str:
        """Build a two-line ASCII face string from *mood_state*.

        Intensity (0-3) is derived from ``|valence|`` using the contract's
        ``valence_boundaries``.  When ``is_listening`` is ``True`` the
        listening face is used regardless of intensity.
        """
        mood_id: str = getattr(mood_state, "mood_id", "neutral")
        valence: float = getattr(mood_state, "valence", 0.0)
        is_listening: bool = getattr(mood_state, "is_listening", False)

        face_key = "listening" if is_listening else mood_id
        face = self._face_templates.get(face_key)
        if face is None:
            face = self._face_templates.get(
                "neutral", {"eyes": ["·"], "mouths": ["─"]}
            )

        eyes: list[str] = face["eyes"]
        mouths: list[str] = face["mouths"]

        if is_listening or self._intensity_level(valence) == 0:
            eye = eyes[0]
        else:
            eye = eyes[1] if len(eyes) > 1 else eyes[0]

        level = self._intensity_level(valence)
        mouth = mouths[min(level, len(mouths) - 1)]

        line1 = f"( {eye} {eye} )"
        line2 = f"  {mouth}"
        return f"{line1}\n{line2}"

    def render_emoji(self, mood_state: Any) -> str:
        """Return a single emoji character for the mood."""
        mood_id: str = getattr(mood_state, "mood_id", "neutral")
        return self._EMOJI_MAP.get(mood_id, self._EMOJI_MAP["neutral"])

    def render_transition(
        self,
        prev_mood_state: Any,
        current_mood_state: Any,
    ) -> list[str]:
        """Morph between two moods, returning intermediate face strings.

        Returns an empty list when both moods are the same.  Otherwise
        generates 3 intermediate frames by interpolating valence/arousal
        and re-classifying each interpolated point to the nearest mood
        centroid via Euclidean distance.
        """
        prev_mood = getattr(prev_mood_state, "mood_id", "neutral")
        curr_mood = getattr(current_mood_state, "mood_id", "neutral")
        if prev_mood == curr_mood:
            return []

        prev_val = getattr(prev_mood_state, "valence", 0.0)
        prev_ar = getattr(prev_mood_state, "arousal", 0.1)
        curr_val = getattr(current_mood_state, "valence", 0.0)
        curr_ar = getattr(current_mood_state, "arousal", 0.1)

        frames: list[str] = []
        n_frames = 3
        for i in range(1, n_frames + 1):
            t = i / (n_frames + 1)
            v = prev_val + (curr_val - prev_val) * t
            a = prev_ar + (curr_ar - prev_ar) * t
            mid = self._classify_nearest(v, a)
            mini = _MiniMood(mid, v, a)
            frames.append(self.render(mini))
        return frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _intensity_level(self, valence: float) -> int:
        v = abs(valence)
        b = self._valence_boundaries
        if v < b[0]:
            return 0
        if v < b[1]:
            return 1
        if v < b[2]:
            return 2
        return 3

    def _classify_nearest(self, valence: float, arousal: float) -> str:
        best_id = "neutral"
        best_dist = float("inf")
        for mid, (cv, ca) in self._MOOD_CENTROIDS.items():
            d = (valence - cv) ** 2 + (arousal - ca) ** 2
            if d < best_dist:
                best_dist = d
                best_id = mid
        return best_id


class _MiniMood:
    """Minimal duck-typed mood object for transition interpolation."""

    __slots__ = ("mood_id", "valence", "arousal", "is_listening")

    def __init__(self, mood_id: str, valence: float, arousal: float) -> None:
        self.mood_id = mood_id
        self.valence = valence
        self.arousal = arousal
        self.is_listening = False
