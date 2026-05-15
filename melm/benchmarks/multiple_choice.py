"""Multiple-choice benchmark case types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultipleChoiceCase:
    case_id: str
    category: str
    prompt: str
    options: tuple[str, ...]
    label_index: int = 0

    def option_texts(self) -> list[str]:
        return [self.prompt + option for option in self.options]
