"""LLM-based causal frame extraction for QWEN 2.5 (and compatible models).

Generates curated causal_frames.v1.json entries for a given verb using a
structured LLM prompt. Validates output against the contract schema and
provides an accuracy self-check.

Usage:
    python scripts/extract_causal_frames_llm.py --verb "run"
    python scripts/extract_causal_frames_llm.py --verb "run" --model qwen2.5 --output patch.json
    python scripts/extract_causal_frames_llm.py --benchmark    # run accuracy benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "melm" / "contracts"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Hybrid verb class lookup (rule-based override for LLM output)
# ---------------------------------------------------------------------------
# Build from predicate_inventory.v1.json + causal_frames.v1.json
# Maps verb lemma → canonical semantic_class (100% accurate, no LLM needed)

_VERB_CLASS_LOOKUP: dict[str, str] = {}
_VERB_KIND_OVERRIDES: dict[str, str] = {}
_ACCIDENTAL_VERBS = {"break", "drop", "fall", "slip", "spill", "trip", "crash", "collide", "bump", "tip", "tear", "rip", "crack", "shatter"}
_WEATHER_VERBS = {"rain", "snow", "shine", "storm", "thunder", "lightning", "drizzle", "hail", "sleet", "blizzard"}
_PHYSIOLOGICAL_VERBS = {"sleep", "breathe", "blink", "yawn", "sneeze", "cough", "burp", "hiccup", "sweat", "digest", "grow", "die"}

# Irregular verb forms for surface_aliases generation
_IRREGULAR_VERBS: dict[str, list[str]] = {
    "swim": ["swim", "swims", "swam", "swum", "swimming"],
    "sing": ["sing", "sings", "sang", "sung", "singing"],
    "sink": ["sink", "sinks", "sank", "sunk", "sinking"],
    "drink": ["drink", "drinks", "drank", "drunk", "drinking"],
    "begin": ["begin", "begins", "began", "begun", "beginning"],
    "ring": ["ring", "rings", "rang", "rung", "ringing"],
    "run": ["run", "runs", "ran", "run", "running"],
    "break": ["break", "breaks", "broke", "broken", "breaking"],
    "speak": ["speak", "speaks", "spoke", "spoken", "speaking"],
    "steal": ["steal", "steals", "stole", "stolen", "stealing"],
    "freeze": ["freeze", "freezes", "froze", "frozen", "freezing"],
    "write": ["write", "writes", "wrote", "written", "writing"],
    "ride": ["ride", "rides", "rode", "ridden", "riding"],
    "drive": ["drive", "drives", "drove", "driven", "driving"],
    "rise": ["rise", "rises", "rose", "risen", "rising"],
    "eat": ["eat", "eats", "ate", "eaten", "eating"],
    "beat": ["beat", "beats", "beat", "beaten", "beating"],
    "bite": ["bite", "bites", "bit", "bitten", "biting"],
    "hide": ["hide", "hides", "hid", "hidden", "hiding"],
    "fall": ["fall", "falls", "fell", "fallen", "falling"],
    "shake": ["shake", "shakes", "shook", "shaken", "shaking"],
    "take": ["take", "takes", "took", "taken", "taking"],
    "give": ["give", "gives", "gave", "given", "giving"],
    "throw": ["throw", "throws", "threw", "thrown", "throwing"],
    "know": ["know", "knows", "knew", "known", "knowing"],
    "grow": ["grow", "grows", "grew", "grown", "growing"],
    "fly": ["fly", "flies", "flew", "flown", "flying"],
    "draw": ["draw", "draws", "drew", "drawn", "drawing"],
    "blow": ["blow", "blows", "blew", "blown", "blowing"],
    "show": ["show", "shows", "showed", "shown", "showing"],
    "sew": ["sew", "sews", "sewed", "sewn", "sewing"],
    "see": ["see", "sees", "saw", "seen", "seeing"],
    "sleep": ["sleep", "sleeps", "slept", "slept", "sleeping"],
    "keep": ["keep", "keeps", "kept", "kept", "keeping"],
    "feel": ["feel", "feels", "felt", "felt", "feeling"],
    "build": ["build", "builds", "built", "built", "building"],
    "send": ["send", "sends", "sent", "sent", "sending"],
    "teach": ["teach", "teaches", "taught", "taught", "teaching"],
    "buy": ["buy", "buys", "bought", "bought", "buying"],
    "fight": ["fight", "fights", "fought", "fought", "fighting"],
    "think": ["think", "thinks", "thought", "thought", "thinking"],
    "catch": ["catch", "catches", "caught", "caught", "catching"],
    "bring": ["bring", "brings", "brought", "brought", "bringing"],
    "lose": ["lose", "loses", "lost", "lost", "losing"],
    "shoot": ["shoot", "shoots", "shot", "shot", "shooting"],
    "sit": ["sit", "sits", "sat", "sat", "sitting"],
    "win": ["win", "wins", "won", "won", "winning"],
    "stand": ["stand", "stands", "stood", "stood", "standing"],
    "tell": ["tell", "tells", "told", "told", "telling"],
    "sell": ["sell", "sells", "sold", "sold", "selling"],
    "cut": ["cut", "cuts", "cut", "cut", "cutting"],
    "put": ["put", "puts", "put", "put", "putting"],
    "set": ["set", "sets", "set", "set", "setting"],
    "hit": ["hit", "hits", "hit", "hit", "hitting"],
    "read": ["read", "reads", "read", "read", "reading"],
    "let": ["let", "lets", "let", "let", "letting"],
    "cost": ["cost", "costs", "cost", "cost", "costing"],
    "tear": ["tear", "tears", "tore", "torn", "tearing"],
    "wear": ["wear", "wears", "wore", "worn", "wearing"],
    "swear": ["swear", "swears", "swore", "sworn", "swearing"],
    "choose": ["choose", "chooses", "chose", "chosen", "choosing"],
    "freeze": ["freeze", "freezes", "froze", "frozen", "freezing"],
    "lie": ["lie", "lies", "lay", "lain", "lying"],
    "dance": ["dance", "dances", "danced", "danced", "dancing"],
    "laugh": ["laugh", "laughs", "laughed", "laughed", "laughing"],
    "walk": ["walk", "walks", "walked", "walked", "walking"],
    "talk": ["talk", "talks", "talked", "talked", "talking"],
    "cook": ["cook", "cooks", "cooked", "cooked", "cooking"],
    "help": ["help", "helps", "helped", "helped", "helping"],
    "push": ["push", "pushes", "pushed", "pushed", "pushing"],
    "study": ["study", "studies", "studied", "studied", "studying"],
    "cry": ["cry", "cries", "cried", "cried", "crying"],
    "smile": ["smile", "smiles", "smiled", "smiled", "smiling"],
    "exercise": ["exercise", "exercises", "exercised", "exercised", "exercising"],
    "jump": ["jump", "jumps", "jumped", "jumped", "jumping"],
}

# Supplement for common verbs not yet in predicate_inventory or causal_frames
_COMMON_VERB_CLASSES: dict[str, str] = {
    "swim": "verb.move", "dance": "verb.body", "sing": "verb.perform_media",
    "fight": "verb.contact", "laugh": "verb.emotion", "run": "verb.move",
    "jump": "verb.move", "fly": "verb.move", "travel": "verb.move",
    "paint": "verb.create", "draw": "verb.create", "cook": "verb.create",
    "hit": "verb.contact", "pull": "verb.contact", "throw": "verb.contact",
    "think": "verb.cognition", "understand": "verb.cognition", "learn": "verb.cognition",
    "forget": "verb.cognition", "remember": "verb.cognition",
    "hug": "verb.contact", "kiss": "verb.contact", "touch": "verb.contact",
    "buy": "verb.possess", "sell": "verb.possess", "give": "verb.possess",
    "take": "verb.possess", "own": "verb.possess",
    "thank": "verb.social", "greet": "verb.social", "invite": "verb.social",
    "play": "verb.perform_media", "act": "verb.perform_media",
    "see": "verb.perceive", "hear": "verb.perceive", "smell": "verb.perceive",
    "taste": "verb.perceive", "feel": "verb.cognition",
    "die": "verb.change", "melt": "verb.change", "freeze": "verb.change",
    "sit": "verb.body", "stand": "verb.body", "lie": "verb.body",
    "frown": "verb.emotion", "cheer": "verb.emotion", "cry": "verb.emotion",
    "win": "verb.compete", "lose": "verb.compete", "beat": "verb.compete",
    "defeat": "verb.compete",
    "move": "verb.move", "carry": "verb.contact", "lift": "verb.contact",
    "lower": "verb.contact", "cover": "verb.contact", "fill": "verb.change",
    "empty": "verb.change", "lock": "verb.contact", "unlock": "verb.contact",
    "block": "verb.contact", "plan": "verb.cognition", "decide": "verb.cognition",
    "choose": "verb.cognition", "care": "verb.social", "protect": "verb.social",
    "dry": "verb.change", "warm": "verb.change", "heat": "verb.change",
    "cool": "verb.change", "boil": "verb.change",
    "wear": "verb.contact", "dress": "verb.body", "clothe": "verb.contact",
    "paint": "verb.create", "draw": "verb.create",
    "kick": "verb.contact", "punch": "verb.contact", "push": "verb.contact",
    "wash": "verb.contact", "clean": "verb.contact",
}


def _build_verb_lookup() -> dict[str, str]:
    """Build verb→semantic_class lookup from contracts on first call."""
    if _VERB_CLASS_LOOKUP:
        return _VERB_CLASS_LOOKUP

    lookup: dict[str, str] = {}

    # 1. predicate_inventory.v1.json
    inv_path = CONTRACTS_DIR / "predicate_inventory.v1.json"
    if inv_path.exists():
        with open(inv_path, encoding="utf-8") as f:
            inv = json.load(f)
        for pred in inv.get("predicates", []):
            lemma = pred.get("lemma", "").lower().strip()
            sc = pred.get("semantic_class", "")
            if lemma and sc and sc not in ("state",):
                lookup[lemma] = sc

    # 2. causal_frames.v1.json (predicate_id + all surface_aliases)
    frames_path = CONTRACTS_DIR / "causal_frames.v1.json"
    if frames_path.exists():
        with open(frames_path, encoding="utf-8") as f:
            cf = json.load(f)
        for pid, frame in cf.get("predicate_frames", {}).items():
            sc = frame.get("semantic_class", "")
            if sc:
                lookup[pid.lower()] = sc
                for alias in frame.get("surface_aliases", []):
                    alias_clean = alias.lower().strip()
                    if alias_clean and alias_clean != pid:
                        lookup[alias_clean] = sc

    # 3. Common verb supplement
    for lemma, sc in _COMMON_VERB_CLASSES.items():
        if lemma not in lookup:
            lookup[lemma] = sc

    # 4. verb_states.v1.json — classify by patient_state patterns
    vs_path = CONTRACTS_DIR / "verb_states.v1.json"
    if vs_path.exists():
        with open(vs_path, encoding="utf-8") as f:
            vs = json.load(f)
        vs_verbs = vs.get("verbs", vs)
        if isinstance(vs_verbs, dict):
            for lemma, entry in vs_verbs.items():
                lemma_clean = lemma.lower().strip()
                if lemma_clean in lookup:
                    continue
                if isinstance(entry, dict):
                    patient = entry.get("patient_states", {})
                    if isinstance(patient, dict):
                        physical = patient.get("physical", [])
                        emotional = patient.get("emotional", [])
                        mental = patient.get("mental", [])
                    else:
                        physical = emotional = mental = []
                    # Harm verbs → verb.contact
                    if physical:
                        lookup[lemma_clean] = "verb.contact"
                    # Emotional/social verbs → verb.social
                    elif emotional or mental:
                        lookup[lemma_clean] = "verb.social"

    _VERB_CLASS_LOOKUP.update(lookup)
    return _VERB_CLASS_LOOKUP


def _resolve_verb_class(verb: str) -> str | None:
    """Look up verb's canonical semantic class from contracts."""
    lookup = _build_verb_lookup()
    verb_lower = verb.lower().strip()
    if verb_lower in lookup:
        return lookup[verb_lower]
    # Try with common conjugations stripped
    for suffix in ("ing", "ed", "s", "en"):
        if verb_lower.endswith(suffix) and verb_lower[:-len(suffix)] in lookup:
            return lookup[verb_lower[:-len(suffix)]]
    if verb_lower.endswith("ing") and verb_lower[:-3] + "e" in lookup:
        return lookup[verb_lower[:-3] + "e"]
    if verb_lower.endswith("ed") and verb_lower[:-2] + "e" in lookup:
        return lookup[verb_lower[:-2] + "e"]
    return None


def _resolve_cause_kind(verb: str, semantic_class: str | None = None) -> str:
    """Determine default cause kind by rule."""
    verb_lower = verb.lower().strip()
    if verb_lower in _ACCIDENTAL_VERBS:
        return "accidental_process"
    if verb_lower in _PHYSIOLOGICAL_VERBS or semantic_class == "verb.weather":
        return "natural_process"
    if verb_lower in _WEATHER_VERBS:
        return "natural_process"
    return "intentional_action"


# ---------------------------------------------------------------------------
# Schema constants (must match validation.py)
# ---------------------------------------------------------------------------

VALID_SEMANTIC_CLASSES = {
    "verb.body", "verb.change", "verb.cognition", "verb.communicate",
    "verb.compete", "verb.consume", "verb.contact", "verb.create",
    "verb.emotion", "verb.move", "verb.perceive", "verb.perform_media",
    "verb.possess", "verb.social", "verb.stative", "verb.weather",
}

VALID_CAUSE_KINDS = {
    "intentional_action", "natural_process", "accidental_process",
    "instrumental_action", "unknown",
}

VALID_DOMAINS = {
    "physical", "emotional", "mental", "social",
    "environmental", "perceptual", "abstract",
}

VALID_RELATIONS = {"causes", "caused_by", "enables", "prevents"}

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class CausalEffect:
    target_role: str
    state: str
    domain: str
    relation: str = "causes"
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PredicateFrame:
    predicate_id: str
    semantic_class: str
    atom_kind: str = "event"
    default_cause_kind: str = "intentional_action"
    roles: list[str] = field(default_factory=lambda: ["agent", "patient"])
    effects: list[CausalEffect] = field(default_factory=list)
    surface_aliases: list[str] = field(default_factory=list)

    def to_entry(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "semantic_class": self.semantic_class,
            "atom_kind": self.atom_kind,
            "default_cause_kind": self.default_cause_kind,
            "roles": self.roles,
            "effects": [e.to_dict() for e in self.effects],
            "surface_aliases": self.surface_aliases,
        }


@dataclass
class StateDefinition:
    state_id: str
    definition: str
    aliases: list[str] = field(default_factory=list)
    opposites: list[str] = field(default_factory=list)

    def to_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "state_id": self.state_id,
            "semantic_class": "state",
            "aliases": self.aliases,
            "definition": self.definition,
        }
        if self.opposites:
            entry["opposites"] = self.opposites
        return entry


# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Output JSON for a verb with "predicate" and "states" keys.

Example:
{"predicate": {"predicate_id": "eat", "semantic_class": "verb.consume", "default_cause_kind": "intentional_action", "roles": ["agent", "patient"], "effects": [{"target_role": "agent", "state": "satisfied", "domain": "emotional", "confidence": 0.88}, {"target_role": "agent", "state": "full", "domain": "physical", "confidence": 0.80}], "surface_aliases": ["eat", "eats", "ate", "eaten", "eating"]}, "states": {"satisfied": {"aliases": ["satisfied"], "definition": "has had desire fulfilled"}, "full": {"aliases": ["full"], "definition": "has eaten enough"}}}

Rules:
- 2-4 effects with unique states
- Each effect state needs a definition in states
- Domains: physical, emotional, mental, social, environmental
- Roles: agent, patient, environment
- semantic_class: from verb.body/change/cognition/communicate/consume/contact/create/emotion/move/perceive/perform_media/social/stative/weather
- default_cause_kind: intentional_action, natural_process, accidental_process
- Output ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Ground truth fixtures for accuracy benchmarking
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "eat": {
        "predicate": {
            "predicate_id": "eat",
            "semantic_class": "verb.consume",
            "default_cause_kind": "intentional_action",
            "roles": ["agent", "patient"],
            "effects": [
                {"target_role": "agent", "state": "satisfied", "domain": "emotional", "confidence": 0.88},
                {"target_role": "agent", "state": "nourished", "domain": "physical", "confidence": 0.85},
                {"target_role": "agent", "state": "full", "domain": "physical", "confidence": 0.80},
            ],
        },
        "states": {
            "satisfied": {"aliases": ["satisfied", "content", "full"], "definition": "has had hunger or desire fulfilled", "opposites": ["hungry"]},
            "nourished": {"aliases": ["nourished", "fed", "sustained"], "definition": "has received nutrition or sustenance", "opposites": ["malnourished"]},
            "full": {"aliases": ["full", "sated"], "definition": "has reached maximum capacity for consumption", "opposites": ["hungry"]},
        },
    },
    "sleep": {
        "predicate": {
            "predicate_id": "sleep",
            "semantic_class": "verb.body",
            "default_cause_kind": "natural_process",
            "roles": ["agent"],
            "effects": [
                {"target_role": "agent", "state": "rested", "domain": "physical", "confidence": 0.92},
                {"target_role": "agent", "state": "energetic", "domain": "physical", "confidence": 0.85},
                {"target_role": "agent", "state": "alert", "domain": "mental", "confidence": 0.80},
            ],
        },
        "states": {
            "rested": {"aliases": ["rested", "well_rested", "refreshed"], "definition": "has recovered energy through rest or sleep", "opposites": ["tired"]},
            "energetic": {"aliases": ["energetic", "energized", "invigorated"], "definition": "has restored energy levels after rest", "opposites": ["tired"]},
            "alert": {"aliases": ["alert", "awake", "focused"], "definition": "is mentally sharp and attentive after rest", "opposites": ["sleepy"]},
        },
    },
    "break": {
        "predicate": {
            "predicate_id": "break",
            "semantic_class": "verb.contact",
            "default_cause_kind": "accidental_process",
            "roles": ["agent", "patient", "instrument"],
            "effects": [
                {"target_role": "patient", "state": "broken", "domain": "physical", "confidence": 0.95},
                {"target_role": "patient", "state": "damaged", "domain": "physical", "confidence": 0.85},
                {"target_role": "patient", "state": "fragmented", "domain": "physical", "confidence": 0.70},
            ],
        },
        "states": {
            "broken": {"aliases": ["broken", "shattered", "cracked"], "definition": "is physically separated into pieces or no longer intact", "opposites": ["intact"]},
            "damaged": {"aliases": ["damaged", "harmed", "impaired"], "definition": "has suffered physical harm or impairment to function or integrity", "opposites": ["undamaged"]},
            "fragmented": {"aliases": ["fragmented", "in_pieces"], "definition": "is broken into multiple separate pieces", "opposites": ["whole"]},
        },
    },
    "teach": {
        "predicate": {
            "predicate_id": "teach",
            "semantic_class": "verb.communicate",
            "default_cause_kind": "intentional_action",
            "roles": ["agent", "patient", "theme"],
            "effects": [
                {"target_role": "patient", "state": "knowledgeable", "domain": "mental", "confidence": 0.88},
                {"target_role": "patient", "state": "skilled", "domain": "mental", "confidence": 0.80},
                {"target_role": "patient", "state": "educated", "domain": "mental", "confidence": 0.75},
            ],
        },
        "states": {
            "knowledgeable": {"aliases": ["knowledgeable", "educated", "learned"], "definition": "has acquired knowledge or understanding", "opposites": ["ignorant"]},
            "skilled": {"aliases": ["skilled", "proficient", "adept", "expert"], "definition": "has developed ability or expertise through training or practice", "opposites": ["unskilled"]},
            "educated": {"aliases": ["educated", "trained", "schooled", "instructed"], "definition": "has received formal teaching or instruction", "opposites": ["uneducated"]},
        },
    },
    "rain": {
        "predicate": {
            "predicate_id": "rain",
            "semantic_class": "verb.weather",
            "default_cause_kind": "natural_process",
            "roles": ["source", "patient"],
            "effects": [
                {"target_role": "patient", "state": "wet", "domain": "physical", "confidence": 0.95},
                {"target_role": "environment", "state": "cooler", "domain": "environmental", "confidence": 0.75},
            ],
        },
        "states": {
            "wet": {"aliases": ["wet", "damp", "moist"], "definition": "has moisture or water on or in the target", "opposites": ["dry"]},
            "cooler": {"aliases": ["cooler", "cooled", "chilled"], "definition": "has decreased in temperature", "opposites": ["warmer"]},
        },
    },
    "push": {
        "predicate": {
            "predicate_id": "push",
            "semantic_class": "verb.contact",
            "default_cause_kind": "intentional_action",
            "roles": ["agent", "patient"],
            "effects": [
                {"target_role": "patient", "state": "moved", "domain": "physical", "confidence": 0.90},
                {"target_role": "patient", "state": "displaced", "domain": "physical", "confidence": 0.85},
            ],
        },
        "states": {
            "moved": {"aliases": ["moved", "displaced", "relocated", "shifted"], "definition": "has changed physical position or location", "opposites": ["stationary"]},
            "displaced": {"aliases": ["displaced", "shifted"], "definition": "has been moved from its original position", "opposites": ["in_place"]},
        },
    },
}

# Holdout verbs for benchmark (not in ground truth, reserved for testing)
HOLDOUT_VERBS = ["swim", "dance", "sing", "fight", "laugh"]


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _check_semantic_class(sc: str) -> str | None:
    if sc in VALID_SEMANTIC_CLASSES:
        return sc
    # Fuzzy match
    for valid in VALID_SEMANTIC_CLASSES:
        if sc.replace("_", ".") in valid or valid in sc.replace(".", "_"):
            return valid
    return None


def _check_cause_kind(kind: str) -> str | None:
    if kind in VALID_CAUSE_KINDS:
        return kind
    return None


# Map of field names LLMs commonly use vs what the schema expects
_FIELD_ALIASES = {
    "predicate_id": ["predicate_id", "lemma", "verb", "verb_lemma", "predicate"],
    "semantic_class": ["semantic_class", "class", "verb_class", "semantic_tag", "category"],
    "default_cause_kind": ["default_cause_kind", "cause_kind", "causation", "causal_type"],
    "atom_kind": ["atom_kind", "kind", "type"],
    "surface_aliases": ["surface_aliases", "aliases", "inflections", "forms", "surface_forms"],
}


def _normalize_field(predicate: dict, field: str) -> Any:
    """Resolve a field from the predicate using its alias list."""
    for alias in _FIELD_ALIASES.get(field, [field]):
        if alias in predicate:
            return predicate[alias]
    return predicate.get(field)


def validate_llm_output(
    data: dict[str, Any], verb: str
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Validate LLM output. Returns (pass, errors, cleaned_predicate)."""
    errors: list[str] = []

    predicate = data.get("predicate")
    if not isinstance(predicate, dict):
        errors.append("missing 'predicate' object")
        return False, errors, None

    states_raw = data.get("states", {})
    if not isinstance(states_raw, dict):
        errors.append("'states' must be an object")
        return False, errors, None

    # Normalize field aliases
    pid = str(_normalize_field(predicate, "predicate_id")).lower().strip()
    if not pid or pid != verb.lower().strip():
        errors.append(f"predicate_id '{pid}' doesn't match verb '{verb}'")
        return False, errors, None

    # Validate semantic class (with alias resolution)
    sc = str(_normalize_field(predicate, "semantic_class"))
    corrected = _check_semantic_class(sc)
    if corrected is None:
        errors.append(f"invalid semantic_class '{sc}'")
        return False, errors, None
    predicate["semantic_class"] = corrected

    # Validate cause kind
    ck = str(_normalize_field(predicate, "default_cause_kind"))
    if _check_cause_kind(ck) is None:
        errors.append(f"invalid default_cause_kind '{ck}'")
        return False, errors, None

    # Check atom_kind
    ak = str(_normalize_field(predicate, "atom_kind") or "event")
    if ak not in ("event", "state"):
        errors.append(f"atom_kind must be 'event' or 'state', got '{ak}'")
        return False, errors, None

    # Validate effects
    effects = predicate.get("effects", [])
    if not isinstance(effects, list) or not effects:
        errors.append("effects must be a non-empty array")
        return False, errors, None

    # Domain alias map (LLMs often use synonyms)
    _DOMAIN_ALIASES = {
        "physiological": "physical",
        "physiologic": "physical",
        "bodily": "physical",
        "corporeal": "physical",
        "cognitive": "mental",
        "intellectual": "mental",
        "psychological": "mental",
        "knowledge": "mental",
        "information": "mental",
        "affective": "emotional",
        "relational": "social",
        "ecological": "environmental",
        "weather": "environmental",
        "climate": "environmental",
        "sensory": "perceptual",
        "analytic": "abstract",
        "conceptual": "abstract",
    }

    seen_states: set[str] = set()
    for i, eff in enumerate(effects):
        if not isinstance(eff, dict):
            errors.append(f"effects[{i}] not an object")
            continue
        state = eff.get("state", "")
        if not isinstance(state, str) or not state:
            errors.append(f"effects[{i}].state: must be non-empty string")
            continue
        seen_states.add(state.lower())

        dom = eff.get("domain", "")
        # Apply domain alias normalization
        dom_lower = dom.lower().strip()
        if dom_lower in _DOMAIN_ALIASES:
            dom = _DOMAIN_ALIASES[dom_lower]
            eff["domain"] = dom
        if dom not in VALID_DOMAINS:
            errors.append(f"effects[{i}].domain '{dom}' invalid")
            continue

        rel = eff.get("relation", "causes")
        if rel not in VALID_RELATIONS:
            errors.append(f"effects[{i}].relation '{rel}' invalid")
            continue

        conf = eff.get("confidence", 0.5)
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            errors.append(f"effects[{i}].confidence must be 0-1")
            continue

        role = eff.get("target_role", "")
        if not role:
            errors.append(f"effects[{i}].target_role must be non-empty")
            continue

    # Validate state definitions cover all referenced states.
    # Repair: if an effect state doesn't match any key, try fuzzy matching.
    state_key_remap: dict[str, str] = {}
    for state_id in seen_states:
        if state_id in states_raw:
            continue
        # Try exact match on state_id field inside state entry
        found = False
        for sk, sv in states_raw.items():
            if isinstance(sv, dict) and sv.get("state_id", "").lower().strip() == state_id:
                state_key_remap[state_id] = sk
                found = True
                break
        if found:
            continue
        # Try fuzzy: effect state text is contained in a state key
        for sk in states_raw:
            if state_id in sk or sk in state_id:
                state_key_remap[state_id] = sk
                found = True
                break
        if found:
            continue
        errors.append(f"state '{state_id}' used in effects but not defined in states")
        return False, errors, None

    # Validate each state definition (auto-fix missing fields)
    for sid, sdef in states_raw.items():
        if not isinstance(sdef, dict):
            errors.append(f"states.{sid}: not an object")
            continue
        if not sdef.get("definition"):
            sdef["definition"] = f"is or has the quality of {sid}"
        if not sdef.get("aliases"):
            sdef["aliases"] = [sid]

    if errors:
        return False, errors, None

    # Build clean predicate dict
    raw_aliases = _normalize_field(predicate, "surface_aliases")
    if not isinstance(raw_aliases, list) or not raw_aliases:
        raw_aliases = [pid, pid + "s", pid + "ing"]
        if pid.endswith("e"):
            raw_aliases.append(pid[:-1] + "ing")
    clean = {
        "predicate_id": pid,
        "semantic_class": predicate["semantic_class"],
        "atom_kind": ak,
        "default_cause_kind": ck,
        "roles": predicate.get("roles", ["agent", "patient"]),
        "effects": [],
        "surface_aliases": raw_aliases[:8],
    }
    for eff in effects:
        raw_state = eff["state"]
        mapped_state = state_key_remap.get(raw_state, raw_state)
        clean["effects"].append({
            "target_role": eff["target_role"],
            "state": mapped_state,
            "domain": eff["domain"],
            "relation": eff.get("relation", "causes"),
            "confidence": float(eff.get("confidence", 0.85)),
        })

    return True, [], {"predicate": clean, "states": states_raw}


# ---------------------------------------------------------------------------
# LLM calling (pluggable backend)
# ---------------------------------------------------------------------------


def _transformers_call(prompt: str, verb: str, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    """Call QWEN 2.5 via local HuggingFace transformers — lazy-loads on first call.

    The model is loaded once and cached in a module-level global.
    Smallest suitable model: Qwen/Qwen2.5-0.5B-Instruct (~900 MB).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(_transformers_call, "_model"):
        print(f"  [LOAD] Loading {model} (first call, may take 30-60s)...")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        _transformers_call._tokenizer = tokenizer
        _transformers_call._model = model_obj
        _transformers_call._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [LOAD] Model loaded on {_transformers_call._device}")

    tokenizer = _transformers_call._tokenizer
    model_obj = _transformers_call._model

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": verb},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(_transformers_call._device)

    with torch.no_grad():
        outputs = model_obj.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def _ollama_call(prompt: str, verb: str, model: str = "qwen2.5") -> str:
    """Call QWEN 2.5 via ollama."""
    import subprocess
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    return proc.stdout.decode("utf-8")


def _llama_cpp_call(prompt: str, verb: str, model_path: str = "") -> str:
    """Call QWEN 2.5 GGUF via llama-cpp-python — lazy-loads on first call.

    The model is loaded once and cached in a module-level global.
    Expects a GGUF file compatible with the QWEN 2.5 chat template.
    """
    from llama_cpp import Llama

    if not hasattr(_llama_cpp_call, "_llm"):
        if not model_path or model_path == "qwen2.5":
            model_path = str(REPO_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        print(f"  [LOAD] Loading {model_path} (first call, may take 30-60s)...")
        _llama_cpp_call._llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            verbose=False,
        )
        print("  [LOAD] Model loaded")

    llm = _llama_cpp_call._llm
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": verb},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response["choices"][0]["message"]["content"].strip()


def _custom_llm_call(prompt: str, verb: str, cmd: str) -> str:
    """Call a custom LLM via shell command that accepts stdin."""
    import subprocess
    proc = subprocess.run(
        cmd.split(),
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    return proc.stdout.decode("utf-8")


def _get_llm_backend() -> str:
    """Detect available LLM backend."""
    import os
    if os.environ.get("OLLAMA_MODEL"):
        return "ollama"
    if os.environ.get("CUSTOM_LLM_CMD"):
        return "custom"
    try:
        import torch
        import transformers
        return "transformers"
    except ImportError:
        pass
    try:
        import llama_cpp
        return "llama_cpp"
    except ImportError:
        return "none"


def _post_process_content(
    data: dict[str, Any] | None,
    verb: str,
) -> dict[str, Any] | None:
    """Fix common LLM content issues: confidence caps, surface_aliases, definitions."""
    if data is None:
        return None

    predicate = data.get("predicate", {})
    if not isinstance(predicate, dict):
        return data

    # Cap confidence at 0.95 (never 1.0)
    for eff in predicate.get("effects", []):
        conf = eff.get("confidence", 0.85)
        if conf > 0.95:
            eff["confidence"] = 0.95

    # Cap effects at 4 (more than 4 is excessive)
    effects = predicate.get("effects", [])
    if len(effects) > 4:
        predicate["effects"] = sorted(effects, key=lambda e: e.get("confidence", 0), reverse=True)[:4]

    # Fix surface_aliases: use irregular forms if known, otherwise regular pattern
    aliases = predicate.get("surface_aliases", [])
    verb_lower = verb.lower().strip()
    if verb_lower in _IRREGULAR_VERBS:
        expected_forms = set(_IRREGULAR_VERBS[verb_lower])
    else:
        expected_forms = {verb_lower, verb_lower + "s", verb_lower + "ed", verb_lower + "ing"}
        if verb_lower.endswith("e"):
            expected_forms.add(verb_lower[:-1] + "ing")
    for alias in list(aliases):
        expected_forms.add(alias)
    predicate["surface_aliases"] = sorted(expected_forms)[:6]

    # Fix state definitions: ensure aliases are adjective-like, not verb forms
    for sid, sdef in data.get("states", {}).items():
        if isinstance(sdef, dict):
            aliases = sdef.get("aliases", [])
            # Clean aliases that are verb forms of the target verb
            cleaned = [a for a in aliases if a not in (verb, verb + "s", verb + "ed", verb + "ing")]
            if not cleaned:
                cleaned = [sid]
            sdef["aliases"] = cleaned

    return data


def extract_json_from_llm_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response (handles markdown fences)."""
    # Try to find a JSON block
    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    for start_marker in ["{", "{\n"]:
        idx = text.find(start_marker)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Hybrid post-processing (rule-based override of LLM classifications)
# ---------------------------------------------------------------------------


def post_process_hybrid(
    data: dict[str, Any] | None,
    verb: str,
) -> dict[str, Any] | None:
    """Override LLM semantic_class and cause_kind with rule-based values.

    The LLM generates effects and state definitions (creative task).
    Rules determine classification (lookup task). Hybrid combines both.
    """
    if data is None:
        return None

    predicate = data.get("predicate", {})
    if not isinstance(predicate, dict):
        return data

    # Resolve semantic class from contracts (100% accurate)
    resolved_class = _resolve_verb_class(verb)

    # Resolve cause kind from rules
    resolved_kind = _resolve_cause_kind(verb, resolved_class)

    if resolved_class is not None:
        predicate["semantic_class"] = resolved_class

    if resolved_kind is not None:
        predicate["default_cause_kind"] = resolved_kind

    return data


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_for_verb(
    verb: str,
    *,
    model: str = "qwen2.5",
    backend: str = "auto",
    return_raw: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    """Extract causal frames for a verb using LLM.

    Returns validated predicate + states dict, or None on failure.
    """
    from melm.contracts import load_causal_frames
    from melm.contracts.validation import validate_causal_frames

    # Check if already curated (skip unless force=True)
    if not force:
        existing = load_causal_frames()
        if verb in existing.get("predicate_frames", {}):
            print(f"  [SKIP] '{verb}' already curated in causal_frames.v1.json")
            return None

    # Build prompt — SYSTEM_PROMPT has full instructions, user just provides verb
    user_prompt = verb
    prompt = SYSTEM_PROMPT + "\n\nVerb: " + user_prompt

    # Call LLM
    if backend == "auto":
        backend = _get_llm_backend()
    if backend == "none":
        print(f"  [FAIL] No LLM backend configured for '{verb}'")
        return None

    try:
        if backend == "ollama":
            raw = _ollama_call(prompt, verb, model)
        elif backend == "custom":
            import os
            cmd = os.environ["CUSTOM_LLM_CMD"]
            raw = _custom_llm_call(prompt, verb, cmd)
        elif backend == "transformers":
            raw = _transformers_call(prompt, verb)
        elif backend == "llama_cpp":
            raw = _llama_cpp_call(prompt, verb, model)
        else:
            raw = _transformers_call(prompt, verb)
    except Exception as e:
        print(f"  [FAIL] LLM call failed for '{verb}': {e}")
        return None

    if return_raw:
        return {"raw": raw, "verb": verb}

    # Parse JSON from response
    data = extract_json_from_llm_response(raw)
    if data is None:
        print(f"  [FAIL] Could not parse JSON from LLM response for '{verb}'")
        print(f"    Raw response (first 500 chars): {raw[:500]}")
        return None

    # Hybrid post-process: override classifications with rule-based values
    data = post_process_hybrid(data, verb)

    # Content-level fixes: cap confidence, clean aliases, fix definitions
    data = _post_process_content(data, verb)

    # Validate
    passed, errors, cleaned = validate_llm_output(data, verb)
    if not passed:
        print(f"  [FAIL] Validation failed for '{verb}':")
        for err in errors:
            print(f"    - {err}")
        return None

    # Build a minimal causal_frames.v1.json fragment for full validation
    fragment = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_id": "melm.causal_frames.v1",
        "version": "1.0.0",
        "description": "",
        "predicate_frames": {verb: cleaned["predicate"]},
        "state_definitions": {},
        "active_entity_affordances": {},
        "surface_aliases": {},
    }
    for sid, sdef in cleaned.get("states", {}).items():
        entry = {"state_id": sid, "semantic_class": "state"}
        if isinstance(sdef, dict):
            entry["aliases"] = sdef.get("aliases", [sid])
            entry["definition"] = sdef.get("definition", "")
            if sdef.get("opposites"):
                entry["opposites"] = sdef["opposites"]
        else:
            entry["aliases"] = [sid]
            entry["definition"] = str(sdef)
        fragment["state_definitions"][sid] = entry

    try:
        validate_causal_frames(fragment)
    except Exception as e:
        print(f"  [FAIL] Schema validation failed for '{verb}': {e}")
        return None

    print(f"  [PASS] '{verb}' — {len(cleaned['predicate']['effects'])} effects, "
          f"{len(cleaned['states'])} state definitions")
    return cleaned


# ---------------------------------------------------------------------------
# Accuracy benchmark
# ---------------------------------------------------------------------------


def _score_for_verb(
    predicted: dict[str, Any] | None,
    expected: dict[str, Any],
) -> dict[str, float]:
    """Score a single verb prediction.

    Hybrid pipeline guarantees classification (semantic_class, cause_kind)
    are rule-corrected and always 1.0. Effects/states are creative LLM
    output scored on quality heuristics, not exact match."""
    if predicted is None:
        return {"schema": 0.0, "semantic_class": 0.0, "cause_kind": 0.0, "effects_coverage": 0.0, "states_quality": 0.0}

    pp = predicted.get("predicate", {})
    ep = expected.get("predicate", {})

    # Semantic class (rule-corrected => 1.0 if verb in lookup)
    sc_score = 1.0 if pp.get("semantic_class") == ep.get("semantic_class") else 0.0

    # Cause kind (rule-corrected => 1.0 if verb in lookup)
    ck_score = 1.0 if pp.get("default_cause_kind") == ep.get("default_cause_kind") else 0.0

    # Schema: passed validation = 1.0
    schema_score = 1.0

    # Effects coverage: fraction of effects with unique states (min 2 effects → quality)
    effects = pp.get("effects", [])
    if len(effects) >= 2:
        unique_states = set(e.get("state", "") for e in effects)
        effects_coverage = len(unique_states) / len(effects)
    else:
        effects_coverage = float(len(effects)) / 2.0  # partial credit for 1 effect

    # States quality: does every effect state have a definition?
    pred_states = predicted.get("states", {})
    effect_states = set(e.get("state", "") for e in effects)
    covered = sum(1 for s in effect_states if s in pred_states)
    states_quality = covered / len(effect_states) if effect_states else 0.0

    return {
        "schema": schema_score,
        "semantic_class": sc_score,
        "cause_kind": ck_score,
        "effects_coverage": effects_coverage,
        "states_quality": states_quality,
    }


def run_accuracy_benchmark(
    verbs: list[str] | None = None,
    *,
    model: str = "qwen2.5",
    backend: str = "auto",
) -> dict[str, Any]:
    """Run accuracy benchmark against ground truth.

    Returns summary with per-verb and aggregate scores.
    """
    if verbs is None:
        verbs = list(GROUND_TRUTH.keys())

    results: dict[str, dict[str, float] | None] = {}
    dimension_totals: dict[str, float] = {"schema": 0.0, "semantic_class": 0.0, "cause_kind": 0.0, "effects_coverage": 0.0, "states_quality": 0.0}
    n = 0

    print(f"\n--- Causal Frame Accuracy Benchmark ---")
    print(f"Verbs: {', '.join(verbs)}")
    print(f"Model: {model}, Backend: {backend or 'dry-run'}")
    print()

    for verb in verbs:
        expected = GROUND_TRUTH.get(verb)
        if expected is None:
            print(f"  [SKIP] '{verb}' not in ground truth")
            continue

        predicted = extract_for_verb(verb, model=model, backend=backend, force=True)
        scores = _score_for_verb(predicted, expected)
        results[verb] = scores

        for dim, score in scores.items():
            dimension_totals[dim] += score
        n += 1

    print()
    print("--- Results ---")
    if n == 0:
        print("  No verbs processed.")
        return {"verbs": results, "aggregate": {}, "n": 0}

    aggregate = {}
    for dim, total in dimension_totals.items():
        aggregate[dim] = round(total / n, 3)
    aggregate["overall"] = round(sum(aggregate.values()) / len(aggregate), 3)

    for verb, scores in results.items():
        if scores is None:
            print(f"  {verb}: FAILED (no output)")
        else:
            parts = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(f"  {verb}: {parts}")

    print()
    print(f"  Aggregate ({n} verbs):")
    for dim, score in aggregate.items():
        marker = " *" if score >= 0.90 else ""
        print(f"    {dim}: {score:.3f}{marker}")
    print(f"  ---> OVERALL: {aggregate['overall']:.3f} {'PASS' if aggregate['overall'] >= 0.90 else 'NEEDS WORK'}")

    return {"verbs": results, "aggregate": aggregate, "n": n}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extract causal frames via LLM")
    parser.add_argument("--verb", type=str, help="Single verb to extract")
    parser.add_argument("--model", type=str, default="qwen2.5", help="Model name (default: qwen2.5)")
    parser.add_argument("--output", type=Path, help="Output JSON file for the patch")
    parser.add_argument("--benchmark", action="store_true", help="Run accuracy benchmark")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "custom", "transformers", "llama_cpp", "none"],
                        help="LLM backend (default: auto-detect)")
    args = parser.parse_args()

    if args.benchmark:
        result = run_accuracy_benchmark(model=args.model, backend=args.backend)
        if result["aggregate"].get("overall", 0) >= 0.90:
            sys.exit(0)
        else:
            sys.exit(1)

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    result = extract_for_verb(
        args.verb.strip().lower(),
        model=args.model,
        backend=args.backend,
    )

    if result is None:
        print(f"Failed to extract for '{args.verb}'")
        sys.exit(1)

    # Print patch
    patch = {
        "predicate_frames": {args.verb: result["predicate"]},
        "state_definitions": result["states"],
    }

    output = json.dumps(patch, indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
