"""Tests for multi-pass LLM story pipeline."""
import pytest
from typing import Any
from dataclasses import dataclass, field


@dataclass
class FakeProfile:
    user_name: str = "Maya"
    age: int = 7
    location: str = "Lagos"
    culture: str = "Yoruba"
    facts: dict[str, str] = field(default_factory=lambda: {
        "favorite_color": "green",
        "school": "you go to school on weekdays",
    })


def test_pipeline_engine_importable():
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    assert StoryPipelineEngine is not None


def test_pipeline_engine_constructs():
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile())
    assert engine.profile.user_name == "Maya"
    assert engine.model_path is not None
    assert "0.5b" in engine.model_path


def test_pipeline_stage_dataclass():
    from melm.appliance.assistant_skill_story_pipeline import PipelineStage
    s = PipelineStage(name="test", phase="planning", temperature=0.3, max_tokens=256, system_prompt="test")
    assert s.name == "test"
    assert s.phase == "planning"
    assert s.temperature == 0.3


def test_pipeline_loads_stages_from_contract():
    from melm.appliance.assistant_skill_story_pipeline import load_pipeline_stages
    stages = load_pipeline_stages()
    assert len(stages) >= 10
    stage_names = [s.name for s in stages]
    for required in ("protagonist", "characters", "setting", "plot", "toc",
                     "intro", "suspense", "wow", "resolution", "end"):
        assert required in stage_names, f"missing stage: {required}"
    planning = [s for s in stages if s.phase == "planning"]
    generation = [s for s in stages if s.phase == "generation"]
    assert len(planning) == 5
    assert len(generation) == 5
