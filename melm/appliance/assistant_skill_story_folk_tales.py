"""Offline folk-tale template engine: select & personalize stories from contract."""
import random
import re

_FOLK_TALES_CACHE: dict[str, list[dict]] | None = None
_LOAD_FAILED = False
_ALT_PROTAGONIST_NAMES: tuple[str, ...] = (
    "Kofi", "Amara", "Chike", "Nneka", "Eze",
    "Adaeze", "Maya", "Theo", "Lena", "Jara",
)


def _load_folk_tales() -> list[dict] | None:
    global _FOLK_TALES_CACHE, _LOAD_FAILED
    if _FOLK_TALES_CACHE is not None:
        return _FOLK_TALES_CACHE
    if _LOAD_FAILED:
        return None
    try:
        from melm.contracts import load_folk_tales as _load
        data = _load()
        _FOLK_TALES_CACHE = data.get("stories", [])
        return _FOLK_TALES_CACHE
    except Exception:
        _LOAD_FAILED = True
        return None



_COMMON_NAMES_RX = re.compile(
    r"\b(?:"
    r"Alenoushka|Ivan|Vasilisa|Koschei|Baba\s+Yaga|Ilonka|Tiidu|Pinocchio|Geppetto|"
    r"Cinderella|Snow\s+White|Rapunzel|Jack|Tom|Hans|Gretel|Rumpelstiltskin|"
    r"Aladdin|Ali\s+Baba|Sinbad|Scheherazade|Mulan|Anansi|Brer\s+Rabbit|"
    r"Kofi|Anika|Kwame|Zuri|Amara|Chike|Nneka|Eze|Adaeze|"
    r"Lucky|Hairy|Simons|Wildrose|Paperarelloo|Magician|Stone-Cutter|Gold-Bearded|"
    r"Tritill|Litill|Robes|Fortunatus|Cottager|Shoes|Dragon|White\s+Cat|"
    r"Beasts|Immortality|Gifts|Pigeon|Dancing\s+Water|Apple\s+Tree|"
    r"Lark|Fairy|Spirit|Brave\s+Little|Envious|Neighbour|Shepherd|"
    r"Fisherman|Crow|Nightingale|Phoenix|Tortoise|Hare|Monkey|Elephant|"
    r"Swan|Wolf|Fox|Bear|Lion|Tiger|Deer|Dove|Serpent"
    r")\b", re.IGNORECASE
)


def _find_protagonist_names(text: str, title: str) -> list[str]:
    """Find the primary character name(s) in a story for replacement.
    
    Title search is case-insensitive (titles use varying case).
    Text search is case-sensitive (only capitalized proper names) to avoid
    false replacing common English words like 'bear', 'fox', 'beasts'.
    """
    names = []
    title_matches = _COMMON_NAMES_RX.findall(title)
    names.extend(title_matches)
    # Text search: case-sensitive, only match capitalized proper names
    text_rx = re.compile(_COMMON_NAMES_RX.pattern, 0)
    head = text[:2000]
    head_matches = text_rx.findall(head)
    for n in head_matches:
        if n not in names:
            names.append(n)
    return names


def _safe_replace(text: str, old: str, new: str) -> str:
    """Replace only the exact matched case. No lowercase fallback — prevents
    false replacing common English words that happen to be in the name list."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def personalize_story(story: dict, user_name: str, location: str, rng: random.Random | None = None) -> str:
    """Swap protagonist names and location in a folk tale."""
    if rng is None:
        rng = random.Random()
    text = story["text"]
    title = story.get("title", "")

    # Find names to replace
    names = _find_protagonist_names(text, title)
    has_replaced_name = False
    if names:
        target = names[0]
        text = _safe_replace(text, target, user_name)
        has_replaced_name = True
        # Swap remaining names with alternatives
        alt_names = list(_ALT_PROTAGONIST_NAMES)
        rng.shuffle(alt_names)
        for i, name in enumerate(names[1:]):
            if i < len(alt_names):
                text = _safe_replace(text, name, alt_names[i])

    # If no name replaced, prepend the user name as the protagonist
    if not has_replaced_name:
        text = f"Once there was a child named {user_name}. " + text[:1].lower() + text[1:]

    # Replace location in first paragraph setting description
    location_patterns = [
        (r"\bin\s+(?:a|the)\s+(?:small\s+)?village\b", f"in {location}"),
        (r"\bin\s+(?:a|the)\s+(?:distant\s+)?kingdom\b", f"in {location}"),
        (r"\bin\s+(?:a|the)\s+(?:faraway\s+)?land\b", f"in {location}"),
        (r"\b(?:near|outside)\s+(?:a|the)\s+(?:little\s+)?(?:village|town|city)\b", f"in {location}"),
    ]
    for pattern, replacement in location_patterns:
        text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

    return text


def _is_valid_story(story: dict) -> bool:
    """Skip fragments (start mid-sentence) and very short entries."""
    text = story.get("text", "")
    if len(text.split()) < 100:
        return False
    if text and text[0].islower():
        return False
    return True


def generate_folk_tale(profile: object | None = None, topics: frozenset[str] | None = None,
                       rng: random.Random | None = None) -> str | None:
    """Select and personalize a folk tale. Returns None if unavailable."""
    stories = _load_folk_tales()
    if not stories:
        return None
    if rng is None:
        rng = random.Random()

    # Filter out fragments (text starts mid-sentence) and very short entries
    candidates = [s for s in stories if _is_valid_story(s)]
    if topics:
        topic_text = " ".join(sorted(topics))
        scored = []
        for s in candidates:
            score = 0
            for t in topics:
                if t.lower() in s["title"].lower() or t.lower() in s["text"].lower()[:1000]:
                    score += 1
            if score > 0:
                scored.append((score, s))
        if scored:
            scored.sort(key=lambda x: -x[0])
            candidates = [s for _, s in scored]

    story = rng.choice(candidates)

    # Personalize
    if profile is not None:
        user_name = getattr(profile, "user_name", "Kofi")
        location = getattr(profile, "location", "Lagos")
        text = personalize_story(story, user_name, location, rng)
    else:
        text = story["text"]

    return text
