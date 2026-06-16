"""Map remaining VerbNet classes using name-based heuristics.

Appends name-based mappings for the 172 VerbNet classes that the data-driven
approach (WordNet supersense inference) could not confidently map.
"""

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_VN_MAP = _REPO / "melm" / "contracts" / "verbnet_map.v1.json"

# Name-based heuristics: VerbNet class name patterns → MELM verb class.
# Each entry is a (partial-name, melm_class) tuple checked via ``if pattern in vn_class``.
NAME_RULES: list[tuple[str, str]] = [
    # verb.body — bodily processes, sensations, grooming, gestures
    ("assuming_position", "verb.body"),
    ("body_motion", "verb.move"),      # body motion is still motion
    ("crane-40", "verb.body"),         # crane/extend body parts
    ("curtsey", "verb.body"),
    ("wink-40", "verb.body"),
    ("pain-40", "verb.body"),
    ("tingle", "verb.body"),
    ("hurt-40", "verb.body"),
    ("floss-41", "verb.body"),
    ("braid-41", "verb.body"),
    # verb.change — state changes
    ("break-45", "verb.change"),
    ("break_down", "verb.change"),
    ("calibratable_cos", "verb.change"),
    ("caused_calibratable_cos", "verb.change"),
    ("convert", "verb.change"),
    ("destroy", "verb.change"),
    ("die-42", "verb.change"),
    ("dysfunction", "verb.change"),
    ("separate", "verb.change"),
    ("split-23", "verb.change"),
    ("stop-55", "verb.change"),
    ("fill-9", "verb.change"),
    ("remove-10", "verb.change"),
    ("concealment", "verb.change"),
    ("harming", "verb.change"),
    ("render-29", "verb.change"),
    ("satisfy-55", "verb.change"),
    ("succeed", "verb.change"),
    ("disappearance", "verb.change"),
    ("terminus", "verb.change"),
    ("sustain-55", "verb.change"),
    ("subjugate", "verb.change"),
    ("begin-55", "verb.change"),
    ("become-109", "verb.change"),
    ("result-27", "verb.change"),
    ("multiply-108", "verb.change"),
    # verb.create — bringing into existence
    ("build-26", "verb.create"),
    ("prepare-26", "verb.create"),
    ("rear-26", "verb.create"),
    ("grow-26", "verb.create"),
    ("engender", "verb.create"),
    ("establish-55", "verb.create"),
    ("scribble-25", "verb.create"),
    ("illustrate-25", "verb.create"),
    ("image_impression", "verb.create"),
    ("knead", "verb.change"),           # shaping/crafting
    ("birth-28", "verb.create"),
    ("convert-26", "verb.change"),
    # verb.communicate
    ("initiate_communication", "verb.communicate"),
    ("respond", "verb.communicate"),
    ("reflexive_appearance", "verb.communicate"),
    ("respond-113", "verb.communicate"),
    ("reciprocate", "verb.communicate"),
    ("masquerade", "verb.communicate"),
    # verb.cognition
    ("assessment", "verb.cognition"),
    ("comprehend", "verb.cognition"),
    ("conjecture", "verb.cognition"),
    ("investigate", "verb.cognition"),
    ("search-35", "verb.cognition"),
    ("rummage", "verb.cognition"),
    ("ferret-35", "verb.cognition"),
    ("hunt-35", "verb.cognition"),
    ("characterize", "verb.cognition"),
    ("neglect", "verb.cognition"),
    ("adopt-93", "verb.cognition"),
    ("accept-77", "verb.cognition"),
    ("abide_by", "verb.cognition"),
    # verb.emotion
    ("marvel-31", "verb.emotion"),
    ("long-32", "verb.emotion"),
    ("care-88", "verb.emotion"),
    ("stimulate-59", "verb.emotion"),
    # verb.move — physical motion
    ("roll-51", "verb.move"),
    ("escape-51", "verb.move"),
    ("chase-51", "verb.move"),
    ("carry-11", "verb.move"),
    ("bring-11", "verb.move"),
    ("send-11", "verb.move"),
    ("slide-11", "verb.move"),
    ("leave-51", "verb.move"),
    ("reach-51", "verb.move"),
    ("meander", "verb.move"),
    ("rotate-51", "verb.move"),
    ("vehicle_path", "verb.move"),
    ("vehicle-51", "verb.move"),
    ("put-9", "verb.move"),
    ("put_direction", "verb.move"),
    ("put_spatial", "verb.move"),
    ("herd-47", "verb.move"),
    ("swarm-47", "verb.move"),
    ("body_motion", "verb.move"),
    # verb.contact — physical contact
    ("hold-15", "verb.contact"),
    ("support-15", "verb.contact"),
    ("throw-17", "verb.contact"),
    ("bump-18", "verb.contact"),
    ("coil-9", "verb.contact"),
    ("contain-15", "verb.stative"),
    ("keep-15", "verb.possess"),
    ("push-12", "verb.contact"),
    ("shake-22", "verb.contact"),
    ("amalgamate", "verb.change"),
    ("mix-22", "verb.change"),
    ("harmonize-22", "verb.change"),
    ("disassemble", "verb.contact"),
    # verb.possess — possession, finance, measurement
    ("own-100", "verb.possess"),
    ("pay-68", "verb.possess"),
    ("cost-54", "verb.possess"),
    ("earn-54", "verb.possess"),
    ("price-54", "verb.possess"),
    ("bill-54", "verb.possess"),
    ("register-54", "verb.possess"),
    ("fit-54", "verb.possess"),
    ("future_having", "verb.possess"),
    ("equip-13", "verb.possess"),
    ("fulfilling", "verb.possess"),
    # verb.social — social interaction, coercion, legal
    ("attack-60", "verb.compete"),
    ("battle-36", "verb.compete"),
    ("meet-36", "verb.social"),
    ("help-72", "verb.social"),
    ("defend-72", "verb.social"),
    ("interact-36", "verb.social"),
    ("confront", "verb.social"),
    ("forbid-64", "verb.social"),
    ("discourage-64", "verb.social"),
    ("admit-64", "verb.social"),
    ("let-64", "verb.social"),
    ("hire-13", "verb.social"),
    ("fire-10", "verb.social"),
    ("resign-10", "verb.social"),
    ("banish", "verb.social"),
    ("cheat-10", "verb.social"),
    ("free-10", "verb.social"),
    ("captain-29", "verb.social"),
    ("conduct-111", "verb.social"),
    ("confine", "verb.social"),
    ("promote-102", "verb.social"),
    ("patent", "verb.social"),
    ("volunteer", "verb.social"),
    ("employment-95", "verb.social"),
    ("acquiesce", "verb.social"),
    ("compel-59", "verb.social"),
    ("prosecute-33", "verb.social"),
    # verb.perceive — perception
    ("encounter-30", "verb.perceive"),
    ("stimulus_subject", "verb.perceive"),
    ("sound_existence", "verb.perceive"),
    # verb.consume — consumption
    ("devour", "verb.consume"),
    ("gorge", "verb.consume"),
    ("absorb-39", "verb.consume"),
    ("use-105", "verb.consume"),
    # verb.perform_media — media performance
    ("performance-26", "verb.perform_media"),
    ("rehearse", "verb.perform_media"),
    # verb.stative — states, existence, relationships
    ("comprise", "verb.stative"),
    ("seem-109", "verb.stative"),
    ("appear-48", "verb.stative"),
    ("occur-48", "verb.stative"),
    ("continue-55", "verb.stative"),
    ("involve-107", "verb.stative"),
    ("relate-86", "verb.stative"),
    ("attend-107", "verb.stative"),
    ("function-105", "verb.stative"),
    ("trifle-105", "verb.stative"),
    ("require-103", "verb.stative"),
    ("exclude-107", "verb.stative"),
    ("entity_specific_modes_being", "verb.stative"),
    ("contiguous_location", "verb.stative"),
    ("spatial_configuration", "verb.stative"),
    ("lodge-46", "verb.stative"),
    ("sound_emission", "verb.perceive"),
    ("light_emission", "verb.perceive"),
    ("substance_emission", "verb.body"),
    ("bulge-47", "verb.stative"),
    ("work-73", "verb.social"),
    ("spend_time", "verb.stative"),
    ("without-82", "verb.social"),
    ("reflect", "verb.cognition"),
    ("accept-77", "verb.cognition"),
    # Remaining 9 ambiguous classes — mapped by member verb semantics
    ("act-114", "verb.perform_media"),
    ("clear-10", "verb.change"),
    ("murder-42", "verb.contact"),
    ("orphan-29", "verb.change"),
    ("pit-10", "verb.change"),
    ("poison-42", "verb.change"),
    ("preparing-26", "verb.create"),
    ("try-61", "verb.cognition"),
    ("withdraw-82", "verb.social"),
]


def _load_current() -> dict[str, str]:
    data = json.loads(_VN_MAP.read_text(encoding="utf-8"))
    return dict(data["mappings"])


def _name_map() -> dict[str, str]:
    """Map VerbNet classes to MELM verb classes using keyword heuristics."""
    result: dict[str, str] = {}
    for pattern, melm_class in NAME_RULES:
        result[pattern] = melm_class
    return result


def _apply_name_map(
    existing: dict[str, str],
    name_map: dict[str, str],
) -> dict[str, str]:
    """Apply name-based heuristics to unmapped classes only."""
    from pathlib import Path
    import xml.etree.ElementTree as ET

    vn_dir = _REPO / ".cache-lexicon-seed" / "verbnet"
    merged = dict(existing)

    for fpath in sorted(vn_dir.rglob("*.xml")):
        vn_class = fpath.stem
        if vn_class in merged:
            continue
        tree = ET.parse(fpath)
        members = [m.get("name") for m in tree.findall(".//MEMBER") if m.get("name")]
        if not members:
            continue

        # Check each rule pattern
        for pattern, melm_class in name_map.items():
            if pattern in vn_class:
                merged[vn_class] = melm_class
                break

    return merged


def main() -> None:
    existing = _load_current()
    name_map = _name_map()
    merged = _apply_name_map(existing, name_map)

    added = len(merged) - len(existing)
    print(f"Existing mappings: {len(existing)}")
    print(f"Merged mappings:   {len(merged)}")
    print(f"Newly added:       {added}")
    print(f"Still unmapped:    {len([f for f in sorted(_REPO.glob('.cache-lexicon-seed/verbnet/*.xml')) if f.stem not in merged])}")

    # Show new mappings by MELM class
    from collections import Counter
    new_by_class: Counter[str] = Counter()
    for vn_class in sorted(set(merged.keys()) - set(existing.keys())):
        new_by_class[merged[vn_class]] += 1
    print()
    print("New mappings by MELM class:")
    for cls, cnt in new_by_class.most_common():
        print(f"  {cls:20s}: {cnt}")

    # Write
    output = {
        "schema_id": "melm.verbnet_map.v1",
        "mappings": dict(sorted(merged.items())),
    }
    _VN_MAP.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {_VN_MAP}")


if __name__ == "__main__":
    main()
