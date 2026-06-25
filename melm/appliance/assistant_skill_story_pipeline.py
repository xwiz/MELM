"""Multi-pass LLM story pipeline engine.

3-phase pipeline with INDEPENDENT sub-passes — NO text carryover between them:
  1. Plot construction (1 LLM call) — structured 5-scene plan with 3 sub-summaries each
  2. Scene expansion (15 independent LLM calls) — each sub-pass receives its own
     unique summary, never sees previous sub-pass output
  3. Assembly — stitch all 15 outputs, enforce word count

Key insight: repetition came from {text} carryover (model anchors on own output) and
shared prev_summary across sub-passes. Fix: each sub-pass is fully independent with
its own unique directive.
"""

from __future__ import annotations

import random as _random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = str(
    _REPO_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
_FALLBACK_MODEL_PATH = str(
    _REPO_ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
_MIN_STORY_WORDS = 400
_FT_ADAPTER_PATH = str(
    _REPO_ROOT / "data" / "qwen-lora-adapter"
)
_HF_BASE_MODEL = "Qwen/Qwen2.5-0.5B"


@dataclass
class SceneSummary:
    number: int
    title: str
    actors: str
    location: str
    setting: str
    sub_summaries: tuple[str, str, str]  # atmosphere, action, closure


_SUBPASS_NAMES = ("atmosphere", "action", "closure")


def _default_subpass_prompts() -> dict[str, str]:
    return {
        "plot_construction": (
            "[ROLE] Master plot-weaver of Yoruba oral tradition.\n"
            "[TASK] Write exactly 5 scenes for a bedtime adventure story.\n"
            "[CONTEXT]\n"
            "- Story for {name}, age {age}, from {location} ({culture})\n"
            "- Topics: {topics}\n"
            "[SCENE ORDER]\n"
            "1. Hook — protagonist is tempted or called to adventure\n"
            "2. Unknown — protagonist enters a strange or threatening place\n"
            "3. Chase — protagonist is pursued\n"
            "4. Rescue — a helper saves the protagonist\n"
            "5. Return — protagonist comes home changed\n"
            "[FORMAT — output SCENE 1 through SCENE 5 like this]\n"
            "SCENE 1: Unique Title\n"
            "ACTORS: Maya\n"
            "LOCATION: specific place for this scene\n"
            "SETTING: time, atmosphere, weather\n"
            "SUB-1: what happens first\n"
            "SUB-2: what happens next\n"
            "SUB-3: how scene ends\n"
            "\n"
            "Write SCENE 2, SCENE 3, SCENE 4, and SCENE 5 now. "
            "IMPORTANT: Each scene must have a DIFFERENT title, DIFFERENT location, "
            "and DIFFERENT sub-summaries. Do NOT reuse locations or plot points. "
            "Use {name} as the protagonist. Make each scene advance the story."
        ),
        "scene_atmosphere": (
            "[ROLE] Cinematic description artist.\n"
            "[TASK] Write the opening of this scene.\n"
            "[CONTEXT]\n"
            "- Location: {location}\n"
            "- Actors present: {actors}\n"
            "- Setting: {setting}\n"
            "- Previous scene ended: {prev_scene_end}\n"
            "[SCENE FOCUS]\n"
            "{sub_summary}\n"
            "{style_guide}\n"
            "[CONSTRAINTS]\n"
            "- Describe what the character sees, hears, smells, and feels\n"
            "- Use vivid sensory language a 7-year-old can picture\n"
            "- Write 3-5 fresh sentences that advance this specific moment\n"
            "- Do NOT repeat phrases from any other scene\n"
            "Write ONLY the scene text. Begin the scene fresh."
        ),
        "scene_action": (
            "[ROLE] Action storyteller.\n"
            "[TASK] Write what happens next in this scene.\n"
            "[CONTEXT]\n"
            "- Location: {location}\n"
            "- Actors present: {actors}\n"
            "- Setting: {setting}\n"
            "- Previous scene ended: {prev_scene_end}\n"
            "[SCENE FOCUS]\n"
            "{sub_summary}\n"
            "{style_guide}\n"
            "[CONSTRAINTS]\n"
            "- Write actions, events, and character movements\n"
            "- Use short sentences for tension, longer ones for wonder\n"
            "- Write 3-5 fresh sentences that advance this specific moment\n"
            "- Do NOT repeat phrases from any other scene\n"
            "Write ONLY the scene text. Start fresh from this moment."
        ),
        "scene_closure": (
            "[ROLE] Scene closer.\n"
            "[TASK] Complete this scene with its climax or emotional beat.\n"
            "[CONTEXT]\n"
            "- Location: {location}\n"
            "- Actors present: {actors}\n"
            "- Setting: {setting}\n"
            "- Previous scene ended: {prev_scene_end}\n"
            "[SCENE FOCUS]\n"
            "{sub_summary}\n"
            "{style_guide}\n"
            "[CONSTRAINTS]\n"
            "- End with a feeling: fear, relief, wonder, or determination\n"
            "- Write 3-4 fresh sentences that deliver this moment\n"
            "- Do NOT repeat phrases from any other scene\n"
            "Write ONLY the scene text. Deliver the closing moment."
        ),
    }


_SUBPASS_TO_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "atmosphere": ("nature_descriptions", "openings"),
    "action": ("character_descriptions", "exaggerations"),
    "closure": ("emotional_beats", "moral_framings"),
}

_STORYTELLING_PHRASES_CACHE: dict[str, Any] | None = None


def _load_storytelling_phrases() -> dict[str, Any] | None:
    global _STORYTELLING_PHRASES_CACHE
    if _STORYTELLING_PHRASES_CACHE is not None:
        return _STORYTELLING_PHRASES_CACHE
    try:
        from melm.contracts import load_storytelling_phrases as _load
        _STORYTELLING_PHRASES_CACHE = _load()
        return _STORYTELLING_PHRASES_CACHE
    except Exception:
        return None


def _build_style_guide(culture: str, narrative_function: str, rng: _random.Random | None = None) -> str:
    data = _load_storytelling_phrases()
    if data is None:
        return ""
    cultures = data.get("cultures", {})
    if culture not in cultures:
        culture = "western"
    cat = cultures.get(culture, {})
    phrases = cat.get(narrative_function, [])
    if not phrases:
        for func in ("openings", "nature_descriptions", "emotional_beats"):
            fallback = cat.get(func, [])
            if fallback:
                phrases = fallback
                break
    if not phrases:
        fallback_cat = cultures.get("western", {})
        for func in ("openings", "nature_descriptions"):
            fallback = fallback_cat.get(func, [])
            if fallback:
                phrases = fallback
                break
    if not phrases:
        return ""

    if rng is None:
        rng = _random.Random()
    count = min(2, len(phrases))
    selected = rng.sample(phrases, count)
    return "VOCABULARY HINTS: Try using phrases like " + " or ".join(f'"{p}"' for p in selected) + "."


class _PipelineLLM:
    _instance = None
    _model_path = None

    @classmethod
    def get(cls, model_path: str = _DEFAULT_MODEL_PATH) -> Any | None:
        if cls._instance is not None and cls._model_path == model_path:
            return cls._instance
        cls._instance = None
        if not Path(model_path).exists():
            alt = model_path.replace("0.5b", "1.5b")
            if Path(alt).exists():
                model_path = alt
            else:
                return None
        try:
            from llama_cpp import Llama
            cls._instance = Llama(model_path=model_path, n_ctx=8192, verbose=False)
            cls._model_path = model_path
        except Exception:
            return None
        return cls._instance


class _HFPipelineLLM:
    """HuggingFace Transformers + PEFT variant of the pipeline LLM.
    Loads base model with fine-tuned LoRA adapter. Drop-in replacement for _PipelineLLM.
    """
    _instance = None

    @classmethod
    def get(cls, adapter_path: str = _FT_ADAPTER_PATH) -> Any | None:
        if cls._instance is not None:
            return cls._instance
        if not Path(adapter_path).is_dir():
            return None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                _HF_BASE_MODEL, quantization_config=bnb,
                device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(_HF_BASE_MODEL, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = PeftModel.from_pretrained(base, adapter_path)
            model.eval()
            cls._instance = _HFLLMInterface(model, tokenizer)
        except Exception:
            return None
        return cls._instance


class _HFLLMInterface:
    """Wrapper making a HF+PEFT model look like llama_cpp.Llama.create_chat_completion()."""

    def __init__(self, model: Any, tokenizer: Any):
        self._model = model
        self._tokenizer = tokenizer

    def create_chat_completion(
        self, messages: list[dict], temperature: float = 0.7,
        max_tokens: int = 1024, repeat_penalty: float = 1.15,
    ) -> dict:
        import torch
        system = messages[0]["content"] if len(messages) > 0 else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        prompt = f"{system}\n\n{user}" if user else system

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=8,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=repeat_penalty,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        if not response:
            response = "(generation failed)"
        return {
            "choices": [{"message": {"content": response}}],
        }


class StoryPipelineEngine:
    """3-phase story pipeline with independent sub-passes."""

    def __init__(
        self,
        profile: Any,
        model_path: str = _DEFAULT_MODEL_PATH,
        llm: Any = None,
        use_fine_tuned: bool = False,
        rng: _random.Random | None = None,
    ):
        self.profile = profile
        self.model_path = model_path
        self._llm = llm
        self.use_fine_tuned = use_fine_tuned
        self._rng = rng or _random.Random()

    def generate(self, topics: frozenset[str] = frozenset()) -> str | None:
        """Run full pipeline. Returns story text or None (trigger fallback)."""
        if self._llm is None:
            if self.use_fine_tuned:
                self._llm = _HFPipelineLLM.get(_FT_ADAPTER_PATH)
            else:
                self._llm = _PipelineLLM.get(self.model_path)
        if self._llm is None:
            return None

        if self.use_fine_tuned:
            return self._generate_ft(topics)

        plan = self._construct_plot(topics)
        if plan is None or len(plan) < 3:
            plan = self._default_plan()
        if plan is None or len(plan) < 3:
            return None

        all_outputs: list[str] = []
        prev_scene_end = "The story begins."
        for i, scene in enumerate(plan):
            for j, sp_name in enumerate(_SUBPASS_NAMES):
                text = self._call_sub_pass(scene, sp_name, j, prev_scene_end)
                if text is None:
                    return None
                all_outputs.append(text)
            prev_scene_end = scene.sub_summaries[2]

        return _assemble_and_polish(self, all_outputs)

    def _default_plan(self) -> list[SceneSummary]:
        """Return the fallback 5-scene plan directly, skipping the broken LLM plot call."""
        name = getattr(self.profile, "user_name", "the child")
        loc = getattr(self.profile, "location", "home")
        return [
            SceneSummary(1, "The Beginning", name, loc,
                         "a peaceful morning, the world is quiet and safe",
                         (f"{name} starts the day happily",
                          f"A curious discovery draws {name} forward",
                          f"{name} crosses into the unknown")),
            SceneSummary(2, "The Unknown", name, "a strange new place",
                         "dim light, unfamiliar sounds",
                         (f"{name} looks around the strange place",
                          "Something moves in the shadows",
                          f"{name} realizes they are not alone")),
            SceneSummary(3, "The Chase", name, "a winding path through the wilderness",
                         "heart-pounding, urgent, the ground rushes beneath",
                         (f"{name} hears footsteps behind them",
                          f"{name} runs as fast as they can",
                          f"{name} sees a safe place ahead")),
            SceneSummary(4, "The Rescue", name, "a clearing with a small hut",
                         "warm light from a window, smoke rising from a chimney",
                         ("A kind face appears at the door",
                          f"The helper welcomes {name} inside",
                          f"{name} feels safe for the first time")),
            SceneSummary(5, "Home Again", name, loc,
                         "familiar and comforting, golden evening light",
                         (f"{name} finds the way home",
                          f"Family welcomes {name} back",
                          f"{name} learns the lesson of the adventure")),
        ]

    def _generate_ft(self, topics: frozenset[str] = frozenset()) -> str | None:
        """2-phase pipeline for fine-tuned model: generate 2 chunks, then polish."""
        name = getattr(self.profile, "user_name", "Maya")
        age = str(getattr(self.profile, "age", 7))
        loc = getattr(self.profile, "location", "Lagos")
        topic_str = ", ".join(sorted(topics)) if topics else "adventure"

        p1 = (
            f"Tell me a story about {name}, a {age}-year-old child from {loc}. "
            f"The story is about {topic_str}. "
            "Describe how the story begins. What does the main character see and feel? "
            "Write the first half of the story now, with vivid sensory details."
        )
        chunk1 = self._llm_call(p1, temperature=0.7, max_tokens=512)
        if chunk1 is None:
            return None

        p2 = (
            "Continue the story from where you left off. "
            f"Describe what happens next, how {name} feels in the middle of the adventure, "
            "and how the story finally ends. Write the second half now."
        )
        chunk2 = self._llm_call(p2, temperature=0.7, max_tokens=512)
        if chunk2 is None:
            chunk2 = ""

        full = chunk1 + "\n\n" + chunk2
        paragraphs = [p.strip() for p in full.replace("\n\n", "\n").split("\n") if p.strip()]
        if not paragraphs:
            return None
        deduped = _dedup_paragraphs(paragraphs)
        exaggerated = _inject_exaggerations(self, deduped)
        result = "\n\n".join(exaggerated)
        location = getattr(self.profile, "location", "Lagos")
        padding = _padding_text(location)
        while len(result.split()) < _MIN_STORY_WORDS:
            result += f"\n\n{padding}"
        return result

    def _construct_plot(self, topics: frozenset[str]) -> list[SceneSummary] | None:
        prompt = _default_subpass_prompts()["plot_construction"]
        format_vars = {
            "name": self.profile.user_name,
            "age": str(getattr(self.profile, "age", 7)),
            "location": getattr(self.profile, "location", "Lagos"),
            "culture": getattr(self.profile, "culture", "Yoruba"),
            "topics": ", ".join(sorted(topics)) if topics else "adventure",
        }
        prompt = self._safe_format(prompt, format_vars)
        if prompt is None:
            return None

        response = self._llm_call(prompt, temperature=0.7, max_tokens=1024, repeat_penalty=1.3)
        if response is None:
            return None

        scenes = self._parse_scenes(response)
        if scenes is None or len(scenes) < 3:
            return None

        # Ensure scene 1 exists by number
        if scenes[0].number != 1:
            scenes.insert(0, SceneSummary(
                number=1,
                title=f"The Beginning",
                actors=getattr(self.profile, "user_name", "the child"),
                location=getattr(self.profile, "location", "home"),
                setting="a peaceful morning, the world is quiet and safe",
                sub_summaries=(
                    f"{getattr(self.profile, 'user_name', 'The child')} starts the day happily",
                    f"A curious discovery draws {getattr(self.profile, 'user_name', 'the child')} forward",
                    f"{getattr(self.profile, 'user_name', 'The child')} crosses into the unknown",
                ),
            ))

        # Pad to 5 scenes for a proper story arc
        name = getattr(self.profile, "user_name", "the child")
        loc = getattr(self.profile, "location", "home")
        fallback_scenes = [
            ("The Unknown", "a strange new place",
             "dim light, unfamiliar sounds",
             (f"{name} looks around the strange place", f"Something moves in the shadows", f"{name} realizes they are not alone")),
            ("The Chase", "a winding path through the wilderness",
             "heart-pounding, urgent, the ground rushes beneath",
             (f"{name} hears footsteps behind them", f"{name} runs as fast as they can", f"{name} sees a safe place ahead")),
            ("The Rescue", "a clearing with a small hut",
             "warm light from a window, smoke rising from a chimney",
             (f"A kind face appears at the door", f"The helper welcomes {name} inside", f"{name} feels safe for the first time")),
            ("Home Again", loc,
             "familiar and comforting, golden evening light",
             (f"{name} finds the way home", f"Family welcomes {name} back", f"{name} learns the lesson of the adventure")),
        ]
        needed = 5 - len(scenes)
        for i in range(needed):
            title, floc, fset, subs = fallback_scenes[i % len(fallback_scenes)]
            next_num = max(s.number for s in scenes) + 1 if scenes else 1
            scenes.append(SceneSummary(next_num, title, name, floc, fset, subs))

        # Fill empty fields with defaults
        for sc in scenes:
            if not sc.actors:
                sc.actors = name
            if not sc.location:
                sc.location = loc
            if not sc.setting:
                sc.setting = "evening, the air is warm and still"
            subs = []
            for s in sc.sub_summaries:
                if not s:
                    s = f"The scene continues in {sc.location}"
                subs.append(s)
            object.__setattr__(sc, "sub_summaries", tuple(subs))

        return scenes

    def _call_sub_pass(
        self,
        scene: SceneSummary,
        sp_name: str,
        sp_index: int,
        prev_scene_end: str,
    ) -> str | None:
        prompts = _default_subpass_prompts()
        key = f"scene_{sp_name}"
        if key not in prompts:
            return None

        prompt = prompts[key]
        # Get sub-pass config for temperature/max_tokens
        configs = {
            "atmosphere": (0.6, 320),
            "action": (0.7, 256),
            "closure": (0.6, 256),
        }
        temperature, max_tokens = configs.get(sp_name, (0.6, 256))
        sub_summary = scene.sub_summaries[sp_index] if sp_index < len(scene.sub_summaries) else ""

        narrative_funcs = _SUBPASS_TO_FUNCTIONS.get(sp_name, ("openings",))
        culture = getattr(self.profile, "culture", "yoruba").lower()
        guide_parts = [_build_style_guide(culture, fn, self._rng) for fn in narrative_funcs]
        style_guide = "\n".join(g for g in guide_parts if g)

        format_vars = {
            "location": scene.location,
            "actors": scene.actors,
            "setting": scene.setting,
            "prev_scene_end": prev_scene_end,
            "sub_summary": sub_summary,
            "style_guide": style_guide,
        }
        prompt = self._safe_format(prompt, format_vars)
        if prompt is None:
            return None

        return self._llm_call(prompt, temperature=temperature, max_tokens=max_tokens)

    def _safe_format(self, text: str, vars: dict[str, str]) -> str | None:
        try:
            return text.format(**{
                k: v for k, v in vars.items()
                if f"{{{k}}}" in text or f"{{{k}:" in text
            })
        except KeyError:
            return None

    def _llm_call(self, prompt: str, temperature: float, max_tokens: int,
                   repeat_penalty: float | None = None) -> str | None:
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Begin."},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty if repeat_penalty is not None else 1.15,
            )
            text = response["choices"][0]["message"]["content"].strip()
            return text if text else None
        except Exception:
            return None

    def _parse_scenes(self, text: str) -> list[SceneSummary] | None:
        scenes: list[SceneSummary] = []
        current: dict[str, Any] = {}

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            scene_match = re.match(r"^SCENE\s+(\d+):\s*(.*)", line, re.IGNORECASE)
            if scene_match:
                if current:
                    self._finalize_scene(current, scenes)
                current = {
                    "number": int(scene_match.group(1)),
                    "title": scene_match.group(2).strip(),
                    "actors": "",
                    "location": "",
                    "setting": "",
                    "subs": ["", "", ""],
                }
                continue

            if not current:
                continue

            up = line.upper()
            if up.startswith("ACTORS:"):
                current["actors"] = line.split(":", 1)[1].strip()
            elif up.startswith("LOCATION:"):
                current["location"] = line.split(":", 1)[1].strip()
            elif up.startswith("SETTING:"):
                current["setting"] = line.split(":", 1)[1].strip()
            elif up.startswith("SUB-1:"):
                current["subs"][0] = line.split(":", 1)[1].strip()
            elif up.startswith("SUB-2:"):
                current["subs"][1] = line.split(":", 1)[1].strip()
            elif up.startswith("SUB-3:"):
                current["subs"][2] = line.split(":", 1)[1].strip()

        if current:
            self._finalize_scene(current, scenes)

        return scenes if len(scenes) >= 3 else None

    @staticmethod
    def _finalize_scene(current: dict, scenes: list) -> None:
        subs = current.get("subs", ["", "", ""])
        scenes.append(SceneSummary(
            number=current["number"],
            title=current["title"],
            actors=current.get("actors", ""),
            location=current.get("location", ""),
            setting=current.get("setting", ""),
            sub_summaries=(subs[0], subs[1], subs[2]),
        ))


def is_pipeline_available(model_path: str = _DEFAULT_MODEL_PATH) -> bool:
    if not Path(model_path).exists():
        alt = model_path.replace("0.5b", "1.5b")
        if not Path(alt).exists():
            return False
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


_INTENSE_WORDS: frozenset[str] = frozenset({
    "suddenly", "terrifying", "amazing", "incredible",
    "enormous", "magnificent", "fearsome", "wonderful",
    "breathtaking", "glorious", "mighty", "furious",
    "massive", "ancient", "mysterious", "dazzling",
})
_BLAND_THRESHOLD = 0.15


def _get_4grams(text: str) -> set[str]:
    words = text.split()
    if len(words) < 4:
        return set()
    return {" ".join(words[i:i + 4]) for i in range(len(words) - 3)}


def _dedup_paragraphs(paragraphs: list[str]) -> list[str]:
    if not paragraphs:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for para in paragraphs:
        phrases = _get_4grams(para.lower())
        if not phrases:
            result.append(para)
            continue
        repeats = sum(1 for p in phrases if p in seen)
        if repeats <= max(1, int(_BLAND_THRESHOLD * len(phrases))):
            result.append(para)
        seen |= phrases
    return result or [paragraphs[-1]]


def _find_bland_paragraphs(paragraphs: list[str]) -> list[int]:
    scored: list[tuple[int, int]] = []
    for i, para in enumerate(paragraphs):
        words = para.lower().split()
        intense_score = sum(1 for w in words if w.rstrip(".,!?;:") in _INTENSE_WORDS)
        scored.append((intense_score, i))
    scored.sort()
    return [i for _, i in scored[:3]]


def _inject_exaggerations(engine: Any, paragraphs: list[str]) -> list[str]:
    if engine._llm is None:
        return paragraphs
    candidates = _find_bland_paragraphs(paragraphs)
    if not candidates:
        return paragraphs
    for idx in candidates[:2]:
        prompt = (
            "Rewrite this scene with ONE dramatic exaggeration. "
            "Make something bigger, faster, scarier, or more wonderful than real life. "
            "Do not change the story events \u2014 just add one striking exaggeration. "
            "Write ONLY the rewritten scene text.\n\n"
            f"{paragraphs[idx]}"
        )
        rewritten = engine._llm_call(prompt, temperature=0.9, max_tokens=200)
        if rewritten:
            paragraphs[idx] = rewritten
    return paragraphs


def _polish_story(story: str) -> str:
    """Standalone: dedup repeated phrases in a full story text."""
    paragraphs = story.split("\n\n")
    deduped = _dedup_paragraphs(paragraphs)
    return "\n\n".join(deduped)


def _padding_text(location: str) -> str:
    return (
        f"And that is how the night ended in {location} \u2014 "
        "with a heart a little braver and a world a little wider. "
        "The rain had stopped, the moon had risen, and the story, "
        "like all good stories, would be told again."
    )


def _assemble_and_polish(engine: Any, all_outputs: list[str]) -> str:
    """Assemble outputs, dedup, exaggerate, then enforce min words."""
    story = "\n\n".join(all_outputs)
    paragraphs = story.split("\n\n")
    deduped = _dedup_paragraphs(paragraphs)
    exaggerated = _inject_exaggerations(engine, deduped)
    result = "\n\n".join(exaggerated)
    location = getattr(engine.profile, "location", "Lagos")
    padding = _padding_text(location)
    while len(result.split()) < _MIN_STORY_WORDS:
        result += f"\n\n{padding}"
    return result


_TRAIT_SEPARATOR_RE = re.compile(r',\s*')


def _clean_traits(traits_raw: str) -> str:
    traits = [t.strip().lower() for t in _TRAIT_SEPARATOR_RE.split(traits_raw) if t.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for t in traits:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return ", ".join(unique)


def _validate_protagonist(text: str, expected_name: str) -> bool:
    return expected_name.lower() in text.lower()
