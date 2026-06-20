"""Offline extraction script for NAMELESS verb causality database.

Reads all letter-prefix .json files from the verb_state directory and outputs
curated verb_states.v1.json entries with moral cognition priority ranking.

Usage:
    python scripts/extract_nameless_verb_states.py

Output:
    - melm/contracts/nameless_extracted_verbs.json  (all qualifying verbs)
    - stdout ranked curation guide with statistics
"""

import json
from collections import Counter
from pathlib import Path

NAMELESS_DIR = Path(r"C:\dev\nameless_vector\verb_state")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "melm" / "contracts" / "nameless_extracted_verbs.json"


def has_non_empty_state(fos: dict) -> bool:
    """Check if final_object_states has at least one non-empty string."""
    for dim in ("physical", "emotional", "mental"):
        items = fos.get(dim, [])
        if isinstance(items, list) and any(s.strip() for s in items):
            return True
        if isinstance(items, str) and items.strip():
            return True
    return False


def get_non_empty_dimensions(fos: dict) -> dict:
    """Return dict with only non-empty dimensions from final_object_states."""
    result = {}
    for dim in ("physical", "emotional", "mental"):
        items = fos.get(dim, [])
        if isinstance(items, list):
            cleaned = [s for s in items if s.strip()]
        elif isinstance(items, str):
            cleaned = [items.strip()] if items.strip() else []
        else:
            cleaned = []
        if cleaned:
            result[dim] = cleaned
    return result


def has_sentient_qualifier(states: list) -> bool:
    """Check if any state string contains _if_sentient suffix."""
    return any("_if_sentient" in s for s in states)


def extract_emotional_states(fos: dict) -> list:
    """Extract emotional dimension states (with _if_sentient suffix preserved)."""
    items = fos.get("emotional", [])
    if isinstance(items, list):
        return [s for s in items if s.strip()]
    if isinstance(items, str) and items.strip():
        return [items.strip()]
    return []


def has_physical_damage(fos: dict) -> bool:
    """Heuristic: check if physical dimension has damage-related states."""
    damage_indicators = [
        "damage", "injure", "hurt", "wound", "break", "destroy", "ruin",
        "wreck", "shatter", "crush", "cut", "puncture", "burn", "bruise",
        "scar", "tear", "rip", "fracture", "crack", "split", "harm",
        "wound", "pain", "bleed", "broken", "shredded", "irritated",
        "fatigued", "inflamed", "impacted",
    ]
    items = fos.get("physical", [])
    if isinstance(items, list):
        for s in items:
            for indicator in damage_indicators:
                if indicator in s.lower():
                    return True
    return False


def main():
    if not NAMELESS_DIR.is_dir():
        print(f"ERROR: NAMELESS directory not found: {NAMELESS_DIR}")
        return

    json_files = sorted(
        f for f in NAMELESS_DIR.iterdir()
        if f.suffix == ".json" and f.name != "global.json"
    )

    result = {}
    total_scanned = 0
    qualifying_count = 0
    sentient_count = 0
    emotional_counter: Counter = Counter()
    skipped_list_fos = 0
    skipped_bad_outcome = 0
    skipped_empty_state = 0
    bad_files = []

    for fpath in json_files:
        try:
            data = json.loads(fpath.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            bad_files.append((fpath.name, str(e)))
            continue

        outcomes = data.get("outcomes", [])
        for outcome in outcomes:
            total_scanned += 1

            if not isinstance(outcome, dict):
                skipped_bad_outcome += 1
                continue

            verb = outcome.get("verb", "")
            if not verb:
                continue

            fos = outcome.get("final_object_states")
            if not isinstance(fos, dict):
                skipped_list_fos += 1
                continue

            if not has_non_empty_state(fos):
                skipped_empty_state += 1
                continue

            qualifying_count += 1

            patient_states = get_non_empty_dimensions(fos)

            subject_mental = []
            subj_states = outcome.get("required_subject_states", {})
            if isinstance(subj_states, dict):
                mental = subj_states.get("mental", [])
                if isinstance(mental, list):
                    subject_mental = [s for s in mental if s.strip()]
                elif isinstance(mental, str) and mental.strip():
                    subject_mental = [mental.strip()]

            subject_emotional = []
            if isinstance(subj_states, dict):
                emotional = subj_states.get("emotional", [])
                if isinstance(emotional, list):
                    subject_emotional = [s for s in emotional if s.strip()]
                elif isinstance(emotional, str) and emotional.strip():
                    subject_emotional = [emotional.strip()]

            patient_types = []
            app_objects = outcome.get("applicable_objects", [])
            if isinstance(app_objects, list):
                patient_types = list(app_objects)

            # Count sentient qualifiers
            emotional_states = extract_emotional_states(fos)
            has_sentient = has_sentient_qualifier(emotional_states)
            if has_sentient:
                sentient_count += 1

            # Aggregate emotional states for frequency ranking
            for es in emotional_states:
                emotional_counter[es] += 1

            result[verb] = {
                "patient_states": patient_states,
                "subject_mental": subject_mental,
                "subject_emotional": subject_emotional,
                "patient_types": patient_types,
            }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Written {len(result)} verbs to {OUTPUT_PATH}")

    # --- Ranked Curation Guide ---

    emotional_verbs = []
    physical_damage_verbs = []
    mental_only_verbs = []

    for verb, entry in result.items():
        ps = entry["patient_states"]
        has_emo = "emotional" in ps
        has_phys = "physical" in ps
        has_ment = "mental" in ps

        if has_emo:
            emotional_verbs.append(verb)
        elif has_phys and has_ment:
            physical_damage_verbs.append(verb)
        elif has_phys and not has_ment:
            physical_damage_verbs.append(verb)
        else:
            mental_only_verbs.append(verb)

    # Sort each category alphabetically
    emotional_verbs.sort()
    physical_damage_verbs.sort()
    mental_only_verbs.sort()

    print()
    print("=" * 72)
    print("  NAMELESS VERB EXTRACTION — RANKED CURATION GUIDE")
    print("=" * 72)
    print()
    print(f"  Total verb entries scanned:       {total_scanned}")
    print(f"  Total qualifying entries:         {qualifying_count}")
    print(f"  Total unique verbs extracted:    {len(result)}")
    print(f"  Verbs with _if_sentient qualifier: {sentient_count}")
    print()
    print(f"  Skipped (bad/corrupt files):      {len(bad_files)}")
    for fname, err in bad_files:
        print(f"    - {fname}: {err}")
    print(f"  Skipped (list-format outcomes):   {skipped_bad_outcome}")
    print(f"  Skipped (list final_object_states): {skipped_list_fos}")
    print(f"  Skipped (empty/no state changes):  {skipped_empty_state}")
    print()

    # --- Emotional verbs (highest priority) ---
    print("-" * 72)
    print("  PRIORITY 1: Emotional final states (highest — moral cognition)")
    print(f"  Count: {len(emotional_verbs)} verbs")
    print("-" * 72)
    for i in range(0, len(emotional_verbs), 6):
        chunk = emotional_verbs[i : i + 6]
        print(f"    {', '.join(chunk)}")
    print()

    # --- Physical damage verbs (medium priority) ---
    print("-" * 72)
    print("  PRIORITY 2: Physical damage states (medium)")
    print(f"  Count: {len(physical_damage_verbs)} verbs")
    print("-" * 72)
    for i in range(0, len(physical_damage_verbs), 6):
        chunk = physical_damage_verbs[i : i + 6]
        print(f"    {', '.join(chunk)}")
    print()

    # --- Mental-only verbs (lowest priority) ---
    print("-" * 72)
    print("  PRIORITY 3: Mental state changes only (lowest)")
    print(f"  Count: {len(mental_only_verbs)} verbs")
    print("-" * 72)
    for i in range(0, len(mental_only_verbs), 6):
        chunk = mental_only_verbs[i : i + 6]
        print(f"    {', '.join(chunk)}")
    print()

    # --- Top 20 emotional states ---
    print("-" * 72)
    print("  TOP 20 EMOTIONAL STATES BY FREQUENCY")
    print("-" * 72)
    top20 = emotional_counter.most_common(20)
    for rank, (state, count) in enumerate(top20, 1):
        marker = "  [sentient]" if "_if_sentient" in state else ""
        print(f"  {rank:2d}. {state:40s} {count:3d}{marker}")
    print()

    # --- Summary ---
    print("=" * 72)
    print(f"  Total unique verbs extracted: {len(result)}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
