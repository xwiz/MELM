# Story Pipeline — Multi-Pass LLM Generation

## Goal

Replace the current ~50-word template-based story generation with a multi-pass LLM pipeline producing 500+ word dramatic stories using local QWEN 1.5B GGUF via `llama_cpp`.

## Architecture

```
User says "Tell me a story about X"
    │
    ├── assistant_synthesis.py:_story_answer()
    │   ├── IF pipeline available (model loaded): run StoryPipelineEngine
    │   └── ELSE: fall back to existing template system
    │
    └── StoryPipelineEngine (assistant_skill_story_pipeline.py)
        ├── Stage 1: Protagonist     → semantic compact form
        ├── Stage 2: Characters       → semantic compact form
        ├── Stage 3: Setting          → semantic compact form
        ├── Stage 4: Plot             → semantic compact form
        ├── Stage 5: TOC              → semantic compact form
        ├── Stage 6: Intro            → creative expansion
        ├── Stage 7: Suspense scenes  → creative expansion
        ├── Stage 8: Wow moment       → creative expansion
        ├── Stage 9: Resolution       → creative expansion
        └── Stage 10: End             → creative expansion
            └── Final assembly → 500+ word story
```

### Two-phase design

**Phase A (Stages 1-5): Planning** — outputs semantic compact form (UOL-like atom notation). No creative text yet. Each stage receives ALL previous stages' compacts as context.

**Phase B (Stages 6-10): Generation** — receives the full plan (all 5 planning compacts) and generates creative prose. Each stage expands from the plan.

### Semantic Compact Form

Each planning stage outputs a structured text no longer than 200 chars:

```
PROTAGONIST: {name}, {age}, [{trait1}, {trait2}, {trait3}]
  appearance: {1-line vivid description}
  origin: {1-line background}
```

```
CHARACTERS:
  - {name}: {role}, [{trait1}, {trait2}], {1-line description}
  - {name}: {role}, [{trait1}, {trait2}], {1-line description}
```

```
SETTING: {location}, {culture}, {time}
  sights: [{detail1}, {detail2}, {detail3}]
  sounds: [{sound1}, {sound2}]
  smells: [{smell1}]
  mood: {tone}
```

```
PLOT:
  ACT1(setup): {scene1} → {scene2}
  ACT2(conflict): {event} → {event} → {crisis}
  ACT3(climax): {wow event} → {resolution}
```

```
TOC:
  1. {title} — {1-line description}
  2. {title} — {1-line description}
  ...
  N. {title} — {1-line description}
```

### Prompt Contract: `story_pipeline_prompts.v1.json`

Each stage has:
- `system_prompt`: instruction for the stage
- `temperature`: float (0.3 for planning, 0.7 for generation)
- `max_tokens`: int (256 for planning, 512 for generation)
- `input_slots`: list of context fields to inject
- `output_format`: description of expected format

### Stages Detail

| # | Stage | Phase | Temp | Max Tok | Output |
|---|-------|-------|------|---------|--------|
| 1 | protagonist | planning | 0.3 | 256 | Compact: name, age, 3 traits, 1-line appearance |
| 2 | characters | planning | 0.3 | 256 | Compact: up to 2 chars, each with role/traits/desc |
| 3 | setting | planning | 0.3 | 256 | Compact: location, time, 3 sights, 2 sounds, 1 smell, mood |
| 4 | plot | planning | 0.5 | 512 | Compact: 3-act structure, key events |
| 5 | toc | planning | 0.3 | 256 | Compact: 5-8 scene titles with 1-line descriptions |
| 6 | intro | generation | 0.7 | 512 | Creative: "Once upon a time..." opening, hook the reader |
| 7 | suspense | generation | 0.7 | 512 | Creative: 2-3 scenes building tension, obstacles |
| 8 | wow | generation | 0.8 | 512 | Creative: climax, shocking/sad/surprising moment, optional poem |
| 9 | resolution | generation | 0.6 | 512 | Creative: falling action, aftermath |
| 10 | end | generation | 0.6 | 384 | Creative: closing scene, lesson, safe return home |

### Context Injection

Phase B stages receive the entire planning context as a single compressed block:

```
[STORY PLAN]
PROTAGONIST: Maya, 7, [brave, curious, kind]
  appearance: large brown eyes that notice everything
CHARACTERS:
  - Mama Ade: [wise, patient], speaks in proverbs
SETTING: Lagos, Yoruba, evening during rainy season
  sights: [orange sunset, wet streets, market stalls closing]
  sounds: [drumming from afar, rain on tin roofs, laughter]
  mood: warm but mysterious
PLOT:
  ACT1: Maya finds a talking drum in the market
  ACT2: The drum's rhythm reveals a secret path to a river
  ACT3: Maya must cross the river by playing a new rhythm
TOC:
  1. Market Discovery
  2. The Drum's Whisper
  ...
```

Generation stages read this compact plan and expand it creatively. They do NOT see each other's output — only the plan + their stage description.

### Core module

```python
class StoryPipelineEngine:
    def __init__(self, profile, model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"):
        self.profile = profile
        self.model_path = model_path
        self._llm = None  # lazy load

    def generate(self, topics: frozenset[str]) -> str:
        plan = {}  # accumulates stage outputs
        for stage in PLANNING_STAGES:
            plan[stage.name] = self._run_stage(stage, plan)
        compact = self._build_compact(plan)
        story_parts = {}
        for stage in GENERATION_STAGES:
            story_parts[stage.name] = self._run_stage(stage, compact)
        return self._assemble(story_parts)

    def _run_stage(self, stage, context) -> str:
        prompt = stage.prompt.format(
            profile=self.profile,
            context=context,
            topics=self.topics,
        )
        response = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": stage.system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=stage.temperature,
            max_tokens=stage.max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _build_compact(self, plan: dict) -> str:
        """Compress all planning stages into a single text block."""
        return "\n".join(
            f"[{name.upper()}]\n{output}"
            for name, output in plan.items()
        )

    def _assemble(self, parts: dict) -> str:
        """Stitch generation parts into a coherent 500+ word story."""
        story = "\n\n".join(parts.values())
        # Ensure 500+ words: add transition phrases between parts
        # Fix pronoun coherence (replace generic hero name with protagonist name)
        return story
```

### Fallback

If any stage fails (model not loaded, JSON parse error, empty output), the whole pipeline falls back to existing template-based `format_story_answer()`.

### Files

| File | Purpose |
|------|---------|
| `melm/appliance/assistant_skill_story_pipeline.py` | `StoryPipelineEngine`, stages, assembly |
| `melm/contracts/story_pipeline_prompts.v1.json` | 10 prompt templates, validator, loader |
| `melm/contracts/registry.v1.json` | Update with new contract entry |
| `melm/appliance/assistant_synthesis.py` | Wire pipeline into `_story_answer()` |

### Model loading

```python
class _PipelineLLM:
    """Lazy singleton — loads once per process."""
    _instance = None
    _model_path = None

    @classmethod
    def get(cls, model_path: str) -> Llama | None:
        if cls._instance is None:
            if not Path(model_path).exists():
                alt = model_path.replace("0.5b", "1.5b")
                if Path(alt).exists():
                    model_path = alt
                else:
                    return None
            try:
                from llama_cpp import Llama
                cls._instance = Llama(model_path=model_path, n_ctx=2048, verbose=False)
                cls._model_path = model_path
            except Exception:
                return None
        return cls._instance
```

### Testing

| Test | What it checks |
|------|---------------|
| `test_stage_protagonist_compact` | Stage 1 outputs valid compact format |
| `test_stage_intro_creative_prose` | Stage 6 outputs full sentences, >50 words |
| `test_full_pipeline_500_words` | End-to-end: output is ≥500 words |
| `test_pipeline_fallback` | When model not available, falls back to template |
| `test_story_pipeline_prompts_contract` | Prompt contract loads and validates |
