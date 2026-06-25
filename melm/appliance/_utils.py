"""Shared utility functions for the MELM assistant."""
import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
