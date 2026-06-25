"""3-way comparison: GGUF pipeline × offline folk tales × fine-tuned pipeline.
Runs all three on the same topic/constraints and reports quality metrics."""
import sys, os, json, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

topics = frozenset({"adventure", "bravery"})
profile_vars = {"user_name": "Maya", "age": 7, "location": "Lagos", "culture": "Yoruba"}


def make_profile():
    from dataclasses import dataclass, field
    @dataclass
    class Profile:
        user_name: str = "Maya"
        age: int = 7
        location: str = "Lagos"
        culture: str = "Yoruba"
        facts: dict[str, str] = field(default_factory=lambda: {"favorite_color": "green"})
    return Profile()


def score(text: str) -> dict:
    words = text.split()
    unique = set(w.lower().rstrip(".,!?;:\"'") for w in words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    has_name = profile_vars["user_name"].lower() in text.lower()
    has_loc = profile_vars["location"].lower() in text.lower()
    length_variation = max(len(w) for w in words) - min(len(w) for w in words) if words else 0
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    return {
        "words": len(words), "chars": len(text), "sentences": len(sentences),
        "avg_sentence_len": round(len(words) / max(len(sentences), 1), 1),
        "unique_words": len(unique),
        "vocab_ratio": round(len(unique) / max(len(words), 1), 3),
        "has_name": has_name, "has_location": has_loc,
        "avg_word_len": round(avg_word_len, 1),
        "max_word_len": max(len(w) for w in words) if words else 0,
    }


def run_gguf():
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine, is_pipeline_available
    if not is_pipeline_available():
        return None, "GGUF model not available"
    engine = StoryPipelineEngine(make_profile(), use_fine_tuned=False)
    t0 = time.time()
    text = engine.generate(topics)
    elapsed = time.time() - t0
    return text, round(elapsed, 1)


def run_folk_tales():
    from melm.appliance.assistant_skill_story_folk_tales import generate_folk_tale
    t0 = time.time()
    text = generate_folk_tale(profile=make_profile(), topics=topics)
    elapsed = time.time() - t0
    return text, round(elapsed, 1)


def run_fine_tuned():
    from melm.appliance.assistant_skill_story_pipeline import StoryPipelineEngine
    engine = StoryPipelineEngine(make_profile(), use_fine_tuned=True, model_path="")
    t0 = time.time()
    text = engine.generate(topics)
    elapsed = time.time() - t0
    return text, round(elapsed, 1)


def main():
    # Print preview of non-empty first 200 chars
    for label, func in [("GGUF Pipeline", run_gguf), ("Folk Tales", run_folk_tales), ("Fine-Tuned Pipeline", run_fine_tuned)]:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        text, elapsed = func()
        if text is None:
            print(f"  SKIPPED: {elapsed}")
            continue
        s = score(text)
        print(f"  Time: {elapsed}s | Words: {s['words']} | Chars: {s['chars']} | Sents: {s['sentences']}")
        print(f"  Vocab ratio: {s['vocab_ratio']} | Has name: {s['has_name']} | Has location: {s['has_location']}")
        print(f"  Preview:\n  {text[:200]}")
        print()


if __name__ == "__main__":
    main()
