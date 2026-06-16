"""Skill protocol and registry — radial consumer pattern for all skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SkillManifest:
    """Declarative metadata that qualifies a module as a skill.

    A skill is a radial consumer of knowledge contracts: it declares what
    contracts it consumes (knowledge_refs) and what frames it handles
    (frames).  The impl lives in free functions within the module — the
    manifest is the single registration point.
    """

    family: str
    frames: tuple[str, ...] = ()
    knowledge_refs: tuple[str, ...] = ()
    template_refs: dict[str, str] = field(default_factory=dict)

    def has_knowledge(self, contract_id: str) -> bool:
        return contract_id in self.knowledge_refs

    def has_frame(self, frame_id: str) -> bool:
        return frame_id in self.frames


class Skill(Protocol):
    """Structural protocol for a skill module.

    A *skill* is any module that exposes a module-level ``MANIFEST``
    constant of type ``SkillManifest``.  The ``MANIFEST`` is the sole
    registration contract — dispatch reads it at import time.
    """

    MANIFEST: SkillManifest


class SkillRegistry:
    """Holds all registered skills and provides lookup by family/frame."""

    def __init__(self) -> None:
        self._by_family: dict[str, SkillManifest] = {}
        self._by_frame: dict[str, str] = {}

    def register(self, manifest: SkillManifest) -> None:
        self._by_family[manifest.family] = manifest
        for frame_id in manifest.frames:
            self._by_frame[frame_id] = manifest.family

    def register_module(self, module: Any) -> None:
        manifest = getattr(module, "MANIFEST", None)
        if isinstance(manifest, SkillManifest):
            self.register(manifest)

    def by_family(self, family: str) -> SkillManifest | None:
        return self._by_family.get(family)

    def by_frame(self, frame_id: str) -> SkillManifest | None:
        family = self._by_frame.get(frame_id)
        if family is not None:
            return self._by_family.get(family)
        return None

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_family))

    @property
    def frames(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_frame))

    def __len__(self) -> int:
        return len(self._by_family)

    def __contains__(self, family: str) -> bool:
        return family in self._by_family


# Module-level singleton — populated at import time by each skill module.
_SKILL_REGISTRY = SkillRegistry()


def register_skill(manifest: SkillManifest) -> None:
    """Register a skill manifest in the global registry."""
    _SKILL_REGISTRY.register(manifest)


def get_skill_registry() -> SkillRegistry:
    """Return the global skill registry singleton."""
    return _SKILL_REGISTRY
