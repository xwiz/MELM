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


def test_pipeline_fallback_when_no_model():
    """When model unavailable, pipeline returns None -> synthesis can fall back."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine, is_pipeline_available
    engine = StoryPipelineEngine(FakeProfile(), model_path="/nonexistent/model.gguf")
    result = engine.generate(frozenset({"bedtime"}))
    assert result is None, "Should return None when model unavailable"
    assert not is_pipeline_available("/nonexistent/model.gguf")


def test_pipeline_builds_compact():
    """_build_compact assembles planning outputs into expected format."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile())
    compact = engine._build_compact({
        "protagonist": "NAME: Maya\nAGE: 7\nTRAITS: brave",
        "setting": "LOCATION: Lagos\nTIME: evening",
    })
    assert "[PROTAGONIST]" in compact
    assert "[SETTING]" in compact
    assert "NAME: Maya" in compact


def test_pipeline_assemble_enforces_500_words():
    """_assemble pads output to at least 500 words if generation is short."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile())
    story = engine._assemble(
        {"intro": "Once upon a time there was a brave girl.", "end": "The end."},
        {"protagonist": "Maya", "setting": "Lagos"},
    )
    assert len(story.split()) >= 500, f"Got {len(story.split())} words, need >= 500"


def test_pipeline_inject_llm():
    """Pipeline accepts injected llm via constructor."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile(), llm="fake")
    assert engine._llm == "fake"


def test_clean_traits_deduplicates():
    from melm.appliance.assistant_skill_story_pipeline import _clean_traits
    assert _clean_traits("brave, curious, curious") == "brave, curious"


def test_clean_traits_single():
    from melm.appliance.assistant_skill_story_pipeline import _clean_traits
    assert _clean_traits("brave") == "brave"


def test_clean_traits_empty():
    from melm.appliance.assistant_skill_story_pipeline import _clean_traits
    assert _clean_traits("") == ""


def test_validate_protagonist_valid():
    from melm.appliance.assistant_skill_story_pipeline import _validate_protagonist
    assert _validate_protagonist("Maya is a brave girl.", "Maya")


def test_validate_protagonist_invalid():
    from melm.appliance.assistant_skill_story_pipeline import _validate_protagonist
    assert not _validate_protagonist("Mrs. Thompson was a kind woman.", "Maya")


def test_validate_protagonist_case_insensitive():
    from melm.appliance.assistant_skill_story_pipeline import _validate_protagonist
    assert _validate_protagonist("MAYA found the drum.", "maya")


def test_pipeline_min_words_constant():
    """_MIN_STORY_WORDS constant is a positive integer."""
    from melm.appliance.assistant_skill_story_pipeline import _MIN_STORY_WORDS
    assert isinstance(_MIN_STORY_WORDS, int)
    assert _MIN_STORY_WORDS >= 100


@pytest.mark.slow
def test_full_pipeline_real_model():
    """End-to-end: generate a story via real QWEN 0.5B model. Marked slow."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, is_pipeline_available,
    )
    if not is_pipeline_available():
        pytest.skip("No QWEN model available")

    engine = StoryPipelineEngine(FakeProfile())
    story = engine.generate(frozenset({"bedtime", "rain"}))
    assert story is not None, "Pipeline should return a story"
    assert len(story.split()) >= 500, f"Story too short: {len(story.split())} words"


@pytest.mark.slow
def test_pipeline_different_topics():
    """Pipeline should produce different stories for different topics."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, is_pipeline_available,
    )
    if not is_pipeline_available():
        pytest.skip("No QWEN model available")

    engine = StoryPipelineEngine(FakeProfile())
    story1 = engine.generate(frozenset({"bedtime"}))
    story2 = engine.generate(frozenset({"tortoise", "drum"}))
    assert story1 is not None and story2 is not None
    # Stories should differ meaningfully (different topics)
    assert story1 != story2, "Different topics should yield different stories"


@pytest.mark.slow
def test_pipeline_respects_profile():
    """Pipeline should use profile name and location in the story."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, is_pipeline_available,
    )

    @dataclass
    class LagosProfile:
        user_name: str = "Kofi"
        age: int = 8
        location: str = "Accra"
        culture: str = "Ga"

    if not is_pipeline_available():
        pytest.skip("No QWEN model available")

    engine = StoryPipelineEngine(LagosProfile())
    story = engine.generate(frozenset({"adventure"}))
    assert story is not None
    assert "Kofi" in story, f"Story should mention profile name 'Kofi', got: {story[:100]}"
    assert "Accra" in story, f"Story should mention location 'Accra', got: {story[:100]}"
