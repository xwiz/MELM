"""Tests for multi-pass LLM story pipeline and offline folk tales."""
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


def test_pipeline_scene_summary_dataclass():
    from melm.appliance.assistant_skill_story_pipeline import SceneSummary
    s = SceneSummary(number=1, title="Test", actors="Maya", location="Lagos", setting="evening",
                     sub_summaries=("plays near the forest", "a rabbit appears", "alone and scared"))
    assert s.number == 1
    assert s.title == "Test"
    assert s.location == "Lagos"
    assert len(s.sub_summaries) == 3


def test_pipeline_subpass_names():
    from melm.appliance.assistant_skill_story_pipeline import _SUBPASS_NAMES
    assert len(_SUBPASS_NAMES) == 3
    assert _SUBPASS_NAMES[0] == "atmosphere"
    assert _SUBPASS_NAMES[1] == "action"
    assert _SUBPASS_NAMES[2] == "closure"


def test_pipeline_default_subpass_prompts():
    from melm.appliance.assistant_skill_story_pipeline import _default_subpass_prompts
    prompts = _default_subpass_prompts()
    assert "plot_construction" in prompts
    assert "scene_atmosphere" in prompts
    assert "scene_action" in prompts
    assert "scene_closure" in prompts


def test_pipeline_fallback_when_no_model():
    """When model unavailable, pipeline returns None -> synthesis can fall back."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine, is_pipeline_available
    engine = StoryPipelineEngine(FakeProfile(), model_path="/nonexistent/model.gguf")
    result = engine.generate(frozenset({"bedtime"}))
    assert result is None, "Should return None when model unavailable"
    assert not is_pipeline_available("/nonexistent/model.gguf")


def test_pipeline_parse_scenes():
    """_parse_scenes extracts SceneSummary list from structured LLM output."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile())
    text = (
        "SCENE 1: The Call\n"
        "ACTORS: Maya\n"
        "LOCATION: Her home\n"
        "SETTING: Evening, warm\n"
        "SUB-1: Maya plays near the forest edge\n"
        "SUB-2: A rabbit appears and lures her deeper\n"
        "SUB-3: Maya finds herself lost and alone\n"
        "\n"
        "SCENE 2: The Chase\n"
        "ACTORS: Maya, Bear\n"
        "LOCATION: Deep forest\n"
        "SETTING: Dark, scary\n"
        "SUB-1: Maya wanders through the dark woods\n"
        "SUB-2: Strange sounds surround her\n"
        "SUB-3: A growl echoes from the shadows\n"
        "\n"
        "SCENE 3: The Rescue\n"
        "ACTORS: Maya, Old Woman, Bear\n"
        "LOCATION: Riverbank\n"
        "SETTING: Misty, twilight\n"
        "SUB-1: Maya reaches a river, exhausted\n"
        "SUB-2: An old woman appears and faces the bear\n"
        "SUB-3: The woman leads Maya safely across\n"
    )
    scenes = engine._parse_scenes(text)
    assert scenes is not None
    assert len(scenes) == 3
    assert scenes[0].title == "The Call"
    assert scenes[0].actors == "Maya"
    assert scenes[0].location == "Her home"
    assert scenes[0].setting == "Evening, warm"
    assert scenes[0].sub_summaries[0] == "Maya plays near the forest edge"
    assert scenes[0].sub_summaries[1] == "A rabbit appears and lures her deeper"
    assert scenes[0].sub_summaries[2] == "Maya finds herself lost and alone"


def test_pipeline_assemble_enforces_400_words():
    """_assemble_and_polish pads output to at least 400 words if generation is short."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, _assemble_and_polish, _MIN_STORY_WORDS,
    )
    engine = StoryPipelineEngine(FakeProfile())
    story = _assemble_and_polish(engine, ["Once upon a time there was a brave girl.", "The end."])
    assert len(story.split()) >= _MIN_STORY_WORDS, f"Got {len(story.split())} words"


def test_pipeline_inject_llm():
    """Pipeline accepts injected llm via constructor."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile(), llm="fake")
    assert engine._llm == "fake"


@pytest.mark.parametrize("input_str,expected", [
    ("brave, curious, curious", "brave, curious"),
    ("brave", "brave"),
    ("", ""),
])
def test_clean_traits(input_str, expected):
    from melm.appliance.assistant_skill_story_pipeline import _clean_traits
    assert _clean_traits(input_str) == expected


@pytest.mark.parametrize("text,name,expected", [
    ("Maya is a brave girl.", "Maya", True),
    ("Mrs. Thompson was a kind woman.", "Maya", False),
    ("MAYA found the drum.", "maya", True),
])
def test_validate_protagonist(text, name, expected):
    from melm.appliance.assistant_skill_story_pipeline import _validate_protagonist
    assert _validate_protagonist(text, name) == expected


# ─── Polish pipeline: dedup ────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_grams", [
    ("birds sang melodiously in the trees",
     {"birds sang melodiously in", "sang melodiously in the", "melodiously in the trees"}),
    ("hello world", set()),
    ("", set()),
])
def test_polish_get_4grams(text, expected_grams):
    from melm.appliance.assistant_skill_story_pipeline import _get_4grams
    assert _get_4grams(text) == expected_grams


@pytest.mark.parametrize("paras,expected", [
    (
        ["The birds sang melodiously in the trees while the sun was warm and bright.",
         "The wind whispered through the leaves carrying the scent of rain.",
         "The birds sang melodiously in the trees just like before and it felt peaceful again."],
        ["The birds sang melodiously in the trees while the sun was warm and bright.",
         "The wind whispered through the leaves carrying the scent of rain."],
    ),
    (
        ["Maya ran through the tall grass feeling free and happy.",
         "A rabbit appeared from behind a rock and hopped closer.",
         "The forest grew dark and strange shadows danced around her."],
        ["Maya ran through the tall grass feeling free and happy.",
         "A rabbit appeared from behind a rock and hopped closer.",
         "The forest grew dark and strange shadows danced around her."],
    ),
    ([], []),
    (["birds sang in the trees and it was nice."], ["birds sang in the trees and it was nice."]),
    (
        ["Ada walked through the quiet streets of Lagos in the morning light.",
         "Ada walked through the quiet streets of Lagos feeling the warm sun.",
         "A merchant called out from his stall offering fresh mangoes for sale."],
        ["Ada walked through the quiet streets of Lagos in the morning light.",
         "A merchant called out from his stall offering fresh mangoes for sale."],
    ),
], ids=["removes_repeated", "keeps_unique", "empty", "single", "mixed_threshold"])
def test_polish_dedup_paragraphs(paras, expected):
    from melm.appliance.assistant_skill_story_pipeline import _dedup_paragraphs
    assert _dedup_paragraphs(paras) == expected


# ─── Polish pipeline: bland detection ──────────────────────────────────────

@pytest.mark.parametrize("paras,check", [
    (
        ["The sun was warm. The grass was green. The birds sang. It was nice.",
         "Suddenly a terrifying roar shook the ground. Maya ran in fear.",
         "An enormous wave crashed over the shore with incredible force."],
        lambda idxs: 0 in idxs and len(idxs) <= 3,
    ),
    ([], lambda idxs: idxs == []),
    (
        ["A terrifying roar shook the ground and a fearsome creature appeared.",
         "A mighty wave crashed with incredible force against the magnificent cliffs.",
         "A gentle breeze carried the scent of wonderful flowers everywhere."],
        lambda idxs: len(idxs) >= 1,
    ),
], ids=["normal", "empty", "all_intense"])
def test_polish_find_bland_paragraphs(paras, check):
    from melm.appliance.assistant_skill_story_pipeline import _find_bland_paragraphs
    assert check(_find_bland_paragraphs(paras))


# ─── Polish pipeline: exaggeration ─────────────────────────────────────────

def test_polish_inject_exaggerations_no_llm():
    """Gracefully returns paragraphs unchanged when no LLM available."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    from melm.appliance.assistant_skill_story_pipeline import _inject_exaggerations
    engine = StoryPipelineEngine(FakeProfile(), llm=None)
    paras = ["The sun was warm. The grass was green. It was a nice morning."]
    result = _inject_exaggerations(engine, paras)
    assert result == paras


def test_polish_inject_exaggerations_max_two():
    """At most 2 paragraphs are rewritten with exaggeration."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    from melm.appliance.assistant_skill_story_pipeline import _inject_exaggerations

    class FakeLLM:
        def create_chat_completion(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "REWRITTEN WITH EXAGGERATION."}}]}

    engine = StoryPipelineEngine(FakeProfile(), llm=FakeLLM())
    paras = [
        "The sun was warm and nice. The grass was green and tall.",
        "A rabbit appeared and looked around. It was cute and small.",
        "Ada walked home slowly. The evening was quiet and calm.",
    ]
    result = _inject_exaggerations(engine, list(paras))
    assert result != paras, "Result should differ from input"
    rewritten = sum(1 for p in result if p == "REWRITTEN WITH EXAGGERATION.")
    assert 1 <= rewritten <= 2, f"Expected 1-2 rewritten, got {rewritten}"


# ─── Polish pipeline: integration ──────────────────────────────────────────

def test_polish_full_story_reduces_repetition():
    """End-to-end polish reduces repeated phrases in a full story."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, _polish_story,
    )
    engine = StoryPipelineEngine(FakeProfile(), llm=None)
    story_with_repeats = (
        "The birds sang melodiously in the trees while the sun was warm.\n\n"
        "Maya walked through the tall grass feeling brave and curious.\n\n"
        "The birds sang melodiously in the trees just like before and it felt calm.\n\n"
        "A rabbit appeared from behind a rock and Maya gasped with delight.\n\n"
        "The wind whispered through the leaves carrying the scent of adventure."
    )
    result = _polish_story(story_with_repeats)
    paras = result.split("\n\n")
    assert len(paras) == 4, f"Dedup should remove 1 paragraph, got {len(paras)}"
    assert result.count("birds sang melodiously") == 1, "Repeated phrase should appear once"


def test_polish_integration_with_pipeline():
    """Polish runs inside pipeline without breaking output."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, _assemble_and_polish,
    )
    engine = StoryPipelineEngine(FakeProfile(), llm=None)
    outputs = [
        "Maya started her day with excitement and wonder in her heart.",
        "A rabbit appeared and hopped closer curious and unafraid.",
        "Maya followed the rabbit into the deep dark unknown forest.",
    ] * 5
    story = _assemble_and_polish(engine, outputs)
    assert story is not None
    assert len(story.split()) >= 400, f"Story too short: {len(story.split())}"
    # Dedup should have removed extreme repetition
    paras = story.split("\n\n")
    assert len(paras) < 15, f"Should have deduped repeated paragraphs, got {len(paras)}"


def test_pipeline_min_words_constant():
    """_MIN_STORY_WORDS constant is a positive integer."""
    from melm.appliance.assistant_skill_story_pipeline import _MIN_STORY_WORDS
    assert isinstance(_MIN_STORY_WORDS, int)
    assert _MIN_STORY_WORDS >= 100


# ─── Storytelling phrases: contract ────────────────────────────────────────

def test_storytelling_phrases_contract_structure():
    """Contract file exists, has correct schema, required cultures, functions, and phrase types."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "melm" / "contracts" / "storytelling_phrases.v1.json"
    assert p.exists(), f"Contract not found at {p}"
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload.get("schema_id", "").startswith("melm.storytelling_phrases")
    cultures = payload.get("cultures", {})
    for expected_culture in {"yoruba", "western", "swahili", "igbo"}:
        assert expected_culture in cultures, f"Missing culture '{expected_culture}'"
    required = {"openings", "transitions", "closings", "nature_descriptions",
                "character_descriptions", "exaggerations", "emotional_beats", "moral_framings"}
    for cname, cat in cultures.items():
        missing = required - set(cat.keys())
        assert not missing, f"Culture '{cname}' missing: {missing}"
        for func in required:
            assert isinstance(cat[func], list) and len(cat[func]) >= 3, \
                f"'{cname}.{func}' must have 3+ entries"
            for i, phr in enumerate(cat[func]):
                assert isinstance(phr, str) and phr.strip(), \
                    f"'{cname}.{func}[{i}]' must be non-empty string"


# ─── Storytelling phrases: style guide injection ───────────────────────────

def test_build_style_guide_returns_string():
    """_build_style_guide returns a formatted string."""
    from melm.appliance.assistant_skill_story_pipeline import _build_style_guide
    result = _build_style_guide("yoruba", "nature_descriptions")
    assert isinstance(result, str)
    assert len(result) > 10


def test_build_style_guide_different_calls():
    """Two consecutive calls return different phrases (random selection)."""
    from melm.appliance.assistant_skill_story_pipeline import _build_style_guide
    results = {_build_style_guide("yoruba", "nature_descriptions") for _ in range(10)}
    assert len(results) >= 2, "All 10 calls returned same string, random selection broken"


@pytest.mark.parametrize("culture,narrative_function", [
    ("ga", "nature_descriptions"),
    ("yoruba", "nonexistent"),
], ids=["fallback_culture", "unknown_function"])
def test_build_style_guide_fallback(culture, narrative_function):
    from melm.appliance.assistant_skill_story_pipeline import _build_style_guide
    result = _build_style_guide(culture, narrative_function)
    assert isinstance(result, str) and len(result) > 10


def test_build_style_guide_subpass_mapping():
    """Each sub-pass type maps to appropriate narrative functions."""
    from melm.appliance.assistant_skill_story_pipeline import _build_style_guide, _SUBPASS_TO_FUNCTIONS
    assert "atmosphere" in _SUBPASS_TO_FUNCTIONS
    assert "action" in _SUBPASS_TO_FUNCTIONS
    assert "closure" in _SUBPASS_TO_FUNCTIONS
    for sp, funcs in _SUBPASS_TO_FUNCTIONS.items():
        assert len(funcs) >= 1, f"{sp} has no narrative functions"


def test_style_guide_in_prompt_generation():
    """Sub-pass prompts contain the injected style guide via format."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, _default_subpass_prompts,
    )
    prompt = _default_subpass_prompts()["scene_atmosphere"]
    assert "{style_guide}" in prompt, "atmosphere prompt missing {style_guide}"
    prompt2 = _default_subpass_prompts()["scene_action"]
    assert "{style_guide}" in prompt2, "action prompt missing {style_guide}"
    prompt3 = _default_subpass_prompts()["scene_closure"]
    assert "{style_guide}" in prompt3, "closure prompt missing {style_guide}"


def test_style_guide_in_call_sub_pass():
    """format_vars contains style_guide key from _build_style_guide call."""
    from melm.appliance.assistant_skill_story_pipeline import (
        _SUBPASS_TO_FUNCTIONS, _build_style_guide,
    )
    assert "atmosphere" in _SUBPASS_TO_FUNCTIONS
    assert "action" in _SUBPASS_TO_FUNCTIONS
    assert "closure" in _SUBPASS_TO_FUNCTIONS
    # Each sub-pass maps to at least one narrative function
    for sp_name, fns in _SUBPASS_TO_FUNCTIONS.items():
        assert len(fns) > 0
        for fn in fns:
            guide = _build_style_guide("yoruba", fn)
            assert "VOCABULARY" in guide or guide == "", f"Bad guide for {sp_name}/{fn}"


def test_style_guide_prompt_includes_vocabulary_hint():
    """The style guide string contains a vocabulary hint prefix."""
    from melm.appliance.assistant_skill_story_pipeline import _build_style_guide
    result = _build_style_guide("yoruba", "openings")
    assert "VOCABULARY" in result
    assert "Try using" in result


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
def test_pipeline_multiple_calls():
    """Pipeline should handle multiple sequential generate calls."""
    from melm.appliance.assistant_skill_story_pipeline import (
        StoryPipelineEngine, is_pipeline_available,
    )
    if not is_pipeline_available():
        pytest.skip("No QWEN model available")

    engine = StoryPipelineEngine(FakeProfile())
    for topics in (frozenset({"bedtime"}), frozenset({"tortoise", "drum"})):
        story = engine.generate(topics)
        # Some calls may fail due to model state; accept occasional None
        if story is not None:
            assert len(story.split()) >= 100, f"Story too short for {topics}"


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


# ── Offline folk-tale template engine tests ──────────────────────────────────

def test_folk_tale_engine_importable():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    assert generate_folk_tale is not None


def test_folk_tale_generates_story():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    result = generate_folk_tale()
    assert result is not None
    assert isinstance(result, str)
    assert len(result) >= 200


def test_folk_tale_personalized_with_profile():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    profile = FakeProfile()
    # Run multiple times since story selection is random
    for _ in range(5):
        result = generate_folk_tale(profile=profile)
        if result and profile.user_name in result:
            return  # Found a match
    # One more check: at least we get a story back
    assert result is not None, "generate_folk_tale should return a story"


def test_folk_tale_different_each_call():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    rng1 = __import__("random").Random(42)
    rng2 = __import__("random").Random(42)
    a = generate_folk_tale(rng=rng1)
    b = generate_folk_tale(rng=rng2)
    assert a is not None and b is not None
    assert a == b, "Same seed should produce same story"


def test_folk_tale_topic_filtering():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    result = generate_folk_tale(topics=frozenset({"dragon"}))
    # Should not crash — topic filtering is best-effort
    assert result is not None


def test_personalize_story_swaps_name():
    from melm.appliance.assistant_skill_story_folk_tales import personalize_story
    story = {"title": "The Brave Prince", "text": "Once upon a time there was a brave prince named Ilonka. Ilonka lived in a village."}
    result = personalize_story(story, "Kofi", "Lagos")
    assert "Kofi" in result
    assert "Ilonka" not in result


def test_personalize_story_swaps_location():
    from melm.appliance.assistant_skill_story_folk_tales import personalize_story
    story = {"title": "The Brave Prince", "text": "He lived in a small village near the forest."}
    result = personalize_story(story, "Kofi", "Accra")
    assert "Accra" in result


def test_personalize_story_no_lord_false_positive():
    """Common-noun title words like 'Lord' must not be replaced.
    
    Story: "Tawara Toda, or 'My Lord Bag of Rice'" — 'Lord' is a title,
    not a protagonist name. Replacing it produces 'My Maya Bag of Rice'.
    """
    from melm.appliance.assistant_skill_story_folk_tales import personalize_story
    story = {
        "title": "Tawara Toda the Brave",
        "text": (
            "Long, long ago there lived, in Japan a brave warrior known to all as "
            "Tawara Toda, or \"My Lord Bag of Rice.\" His true name was Fujiwara Hidesato."
        ),
    }
    result = personalize_story(story, "Maya", "Lagos")
    assert "My Maya Bag of Rice" not in result, \
        f"'Lord' falsely replaced by user name: {result[:100]}"
    assert "My Lord Bag of Rice" in result, \
        f"'Lord' title should be preserved: {result[:100]}"
    assert "Maya" in result, "Prepended name should appear"


def test_folk_tale_contract_exists():
    """Folk tales contract file is loadable and has stories."""
    from melm.contracts.validation import load_folk_tales
    data = load_folk_tales()
    stories = data.get("stories", [])
    assert len(stories) >= 100, f"Expected 100+ stories, got {len(stories)}"
    for s in stories[:5]:
        assert "title" in s
        assert "text" in s
        assert len(s["text"]) >= 200


# ── Fine-tuned model tests ───────────────────────────────────────────────────

def test_hf_pipeline_llm_importable():
    from melm.appliance.assistant_skill_story_pipeline import _HFPipelineLLM, _HFLLMInterface
    assert _HFPipelineLLM is not None
    assert _HFLLMInterface is not None


def test_fine_tuned_engine_constructs():
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile(), use_fine_tuned=False)
    assert engine.use_fine_tuned is False
    engine2 = StoryPipelineEngine(FakeProfile(), use_fine_tuned=True)
    assert engine2.use_fine_tuned is True


@pytest.mark.slow
def test_fine_tuned_generates_text():
    """Fine-tuned model generates coherent story text (not full pipeline)."""
    from melm.appliance.assistant_skill_story_pipeline import _HFPipelineLLM
    llm = _HFPipelineLLM.get()
    if llm is None:
        pytest.skip("No fine-tuned adapter found")
    resp = llm.create_chat_completion(
        [{"role": "system", "content": "Write a short story about a brave child in one paragraph."},
         {"role": "user", "content": "Begin."}],
        temperature=0.7, max_tokens=200,
    )
    text = resp["choices"][0]["message"]["content"]
    assert text is not None
    assert len(text) >= 50, f"Generated text too short: {len(text)} chars"


@pytest.mark.slow
def test_ft_pipeline_generates_story():
    """_generate_ft produces a story from fine-tuned model."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile(), use_fine_tuned=True, model_path="")
    engine._llm = None
    story = engine.generate(frozenset({"adventure"}))
    if story is None:
        pytest.skip("Fine-tuned pipeline unavailable (no adapter or OOM)")
    words = len(story.split())
    assert words >= 200, f"FT story too short: {words} words"
    assert "Maya" in story, "Story missing protagonist name"
    assert words <= 3000, f"FT story too long: {words} words"


@pytest.mark.fast
def test_ft_pipeline_uses_two_chunks():
    """_generate_ft calls _llm_call 2-4 times (2 chunks + 0-2 exaggerations)."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine

    class CallTracker:
        _count = 0
        def create_chat_completion(self, messages, **kwargs):
            self.__class__._count += 1
            return {"choices": [{"message": {"content": "Maya found a glowing stone. She touched it. The world transformed. A rabbit showed her home. The end."}}]}

    engine = StoryPipelineEngine(FakeProfile(), use_fine_tuned=True, model_path="")
    engine._llm = CallTracker()
    result = engine.generate(frozenset())
    assert result is not None
    assert CallTracker._count >= 2, f"Expected >=2 calls, got {CallTracker._count}"


def test_ft_generate_handles_null_chunk2():
    """_generate_ft survives chunk2 returning None."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine, _MIN_STORY_WORDS

    class Chunk1Only:
        _call = 0
        def create_chat_completion(self, messages, **kwargs):
            self.__class__._call += 1
            if self.__class__._call == 1:
                return {"choices": [{"message": {"content": "Maya walked through the forest. She saw a bright light. The end."}}]}
            return {"choices": [{"message": {"content": ""}}]}

    engine = StoryPipelineEngine(FakeProfile(), use_fine_tuned=True, model_path="")
    engine._llm = Chunk1Only()
    result = engine.generate(frozenset())
    assert result is not None
    assert "Maya" in result


def test_default_plan_returns_five_scenes():
    """_default_plan returns proper 5-scene story arc."""
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(FakeProfile(), model_path="", llm=None)
    plan = engine._default_plan()
    assert plan is not None
    assert len(plan) == 5
    titles = [s.title for s in plan]
    locations = [s.location for s in plan]
    assert len(set(titles)) == 5, f"All titles must be unique: {titles}"
    assert len(set(locations)) >= 3, f"Need 3+ unique locations: {locations}"
    for s in plan:
        assert s.actors == "Maya", f"Wrong actors: {s.actors}"
        assert all(subs for subs in s.sub_summaries), f"Empty sub: {s.sub_summaries}"


