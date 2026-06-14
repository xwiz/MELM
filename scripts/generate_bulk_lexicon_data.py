"""Generate word→supersense and verb→verbnet-class JSONL data files.

Usage:
    python scripts/generate_bulk_lexicon_data.py

Writes to melm/contracts/:
    word_supersense_data.v1.jsonl — word→supersense entries for WordNet
    verb_data.v1.jsonl            — verb→verbnet-class entries for VerbNet

These files are consumed by the bulk lexicon seeder
(melm/appliance/assistant_lexicon_bulk.py) at bootstrap time.

The word data is drawn from:
  1. LEGACY_ROUTER_TERM_CLASSES — all 200+ routing vocabulary terms
  2. EXPANDED_WORDS per supersense — manually curated common English nouns and verbs

The verb data is drawn from the functional-grammar _VERBS list, mapped through
the verbnet_map.v1.json contract.
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from melm.contracts import load_contract_json
from melm.appliance.assistant_lexicon_legacy import LEGACY_ROUTER_TERM_CLASSES
from melm.appliance.functional_grammar import _VERBS
from melm.appliance.assistant_lexicon_legacy import LEGACY_VERB_CLASS_MAP


def main():
    # Invert the WordNet supersense map
    wn = load_contract_json("wn_supersense_map.v1.json")
    supersense_to_melm = dict(wn.get("mappings", {}))

    melm_to_supersense = {}
    for ss, mc in supersense_to_melm.items():
        melm_to_supersense.setdefault(mc, []).append(ss)

    word_supersense = []
    for term, cls in LEGACY_ROUTER_TERM_CLASSES.items():
        supersenses = melm_to_supersense.get(cls, [])
        ss = supersenses[0] if supersenses else "noun.Tops"
        from melm.appliance.assistant_lexicon_legacy import _VERBS
        pos = "verb" if term in _VERBS else "noun"
        word_supersense.append({"word": term, "supersense": ss, "pos": pos})

    EXPANDED_WORDS = {
        "noun.person": [
            "adult", "baby", "boy", "brother", "child", "dad", "daughter",
            "doctor", "driver", "engineer", "father", "friend", "girl",
            "grandma", "grandpa", "husband", "kid", "king", "lady", "lawyer",
            "leader", "man", "manager", "mom", "mother", "neighbor", "nurse",
            "officer", "owner", "parent", "partner", "patient", "person",
            "physician", "player", "police", "president", "professor", "queen",
            "relative", "resident", "scientist", "sister", "son", "stranger",
            "student", "teacher", "teen", "trainer", "volunteer", "waitress",
            "wife", "woman", "worker",
        ],
        "noun.animal": [
            "ant", "bear", "bee", "bird", "bug", "butterfly", "cat", "chicken",
            "cow", "deer", "dog", "dolphin", "duck", "eagle", "elephant", "fish",
            "fly", "fox", "frog", "goat", "goose", "hamster", "hawk", "horse",
            "insect", "lion", "lizard", "monkey", "mouse", "owl", "parrot",
            "pet", "pig", "pigeon", "puppy", "rabbit", "rat", "robin", "rooster",
            "seal", "shark", "sheep", "snake", "spider", "squirrel", "tiger",
            "turkey", "turtle", "whale", "wolf", "worm",
        ],
        "noun.artifact": [
            "airplane", "apartment", "appliance", "bag", "ball", "basket",
            "bed", "bicycle", "board", "boat", "book", "bottle", "box",
            "bridge", "building", "bus", "camera", "car", "chair", "clock",
            "computer", "container", "cup", "desk", "device", "door", "drone",
            "engine", "envelope", "factory", "fence", "flag", "floor", "fork",
            "furniture", "gadget", "garden", "gate", "glass", "glove", "gun",
            "hammer", "hat", "house", "instrument", "key", "kitchen", "knife",
            "ladder", "lamp", "laptop", "machine", "medal", "mirror", "motor",
            "network", "notebook", "paint", "panel", "pen", "pencil", "phone",
            "pillow", "pipe", "plane", "plate", "platform", "pump", "radio",
            "rail", "robot", "roof", "room", "screen", "sensor", "shelf",
            "ship", "shirt", "shoe", "signal", "speaker", "table", "tool",
            "tower", "toy", "train", "truck", "tv", "vehicle", "wall", "watch",
            "wheel", "window", "wire",
        ],
        "noun.food": [
            "apple", "bacon", "bagel", "banana", "bean", "beef", "beer",
            "berry", "bread", "breakfast", "broccoli", "cake", "candy",
            "carrot", "cheese", "cherry", "chicken", "chocolate", "coffee",
            "cookie", "corn", "cream", "cuisine", "dessert", "dinner", "dish",
            "drink", "egg", "entree", "fish", "flour", "food", "fruit",
            "garlic", "grape", "ham", "hamburger", "honey", "ice", "jam",
            "juice", "ketchup", "lamb", "lemon", "lentil", "lettuce", "lime",
            "lunch", "mango", "meal", "meat", "melon", "menu", "milk",
            "mushroom", "noodle", "nut", "oil", "olive", "onion", "orange",
            "pasta", "pastry", "pea", "peach", "peanut", "pear", "pepper",
            "pie", "pineapple", "pizza", "pork", "potato", "poultry", "pudding",
            "pumpkin", "raspberry", "recipe", "rice", "salad", "salmon", "salt",
            "sandwich", "sauce", "sausage", "snack", "soda", "soup", "spinach",
            "steak", "strawberry", "sugar", "supper", "sushi", "syrup", "tea",
            "toast", "tomato", "tuna", "turkey", "vanilla", "vegetable",
            "vinegar", "water", "watermelon", "wheat", "wine", "yogurt",
            "zucchini",
        ],
        "noun.location": [
            "address", "airport", "alley", "area", "bank", "bar", "bay",
            "beach", "block", "building", "camp", "campus", "cave", "city",
            "coast", "college", "corner", "country", "county", "court", "cove",
            "creek", "crossing", "desert", "district", "dock", "downtown",
            "drive", "farm", "field", "forest", "garage", "garden", "gate",
            "ground", "gulf", "harbor", "height", "highway", "hill", "home",
            "hospital", "hotel", "house", "island", "junction", "kitchen",
            "lab", "lake", "land", "lane", "library", "lobby", "location",
            "lodge", "lot", "mall", "market", "marsh", "meadow", "museum",
            "neighborhood", "ocean", "office", "park", "parking", "path",
            "pier", "place", "plain", "plant", "plaza", "pond", "pool", "port",
            "prairie", "prison", "property", "railroad", "ranch", "resort",
            "restaurant", "ridge", "river", "road", "roof", "room", "route",
            "school", "sea", "shore", "sidewalk", "site", "space", "square",
            "stadium", "station", "store", "stream", "street", "studio",
            "suburb", "summit", "temple", "territory", "theater", "town",
            "trail", "tunnel", "valley", "village", "warehouse", "water",
            "way", "wood", "yard", "zone",
        ],
        "noun.event": [
            "accident", "activity", "appointment", "attack", "battle", "birth",
            "ceremony", "celebration", "championship", "competition", "concert",
            "conference", "contest", "crash", "debate", "disaster", "earthquake",
            "election", "emergency", "event", "exhibition", "explosion",
            "festival", "fire", "flood", "funeral", "game", "gathering",
            "graduation", "holiday", "hurricane", "incident", "interview",
            "invasion", "journey", "landslide", "launch", "league", "lecture",
            "match", "meeting", "migration", "parade", "party", "performance",
            "protest", "race", "rally", "reception", "reunion", "riot",
            "ritual", "session", "show", "spill", "sport", "storm", "summit",
            "tournament", "trial", "vacation", "wedding", "workshop",
        ],
        "noun.cognition": [
            "assumption", "awareness", "belief", "brain", "clue", "concept",
            "conclusion", "confidence", "consciousness", "curiosity", "decision",
            "definition", "discovery", "dream", "education", "expectation",
            "experience", "expertise", "fact", "fantasy", "goal", "hint",
            "hypothesis", "idea", "illusion", "imagination", "impression",
            "information", "insight", "intellect", "intelligence", "judgment",
            "knowledge", "lesson", "logic", "memory", "metaphor", "mind",
            "notion", "opinion", "perspective", "plan", "prediction",
            "preference", "premise", "principle", "reason", "reasoning",
            "recognition", "recollection", "thought", "understanding", "view",
            "wisdom",
        ],
        "noun.act": [
            "action", "answer", "applause", "attempt", "behavior", "call",
            "campaign", "chore", "command", "complaint", "cooperation", "cry",
            "dance", "defense", "demand", "diet", "drive", "duty", "effort",
            "entertainment", "escape", "exercise", "experiment", "flight",
            "gesture", "greeting", "help", "hike", "hunt", "initiative",
            "inquiry", "invitation", "jump", "kick", "kiss", "knock", "laugh",
            "march", "measure", "mission", "movement", "nod", "operation",
            "performance", "plan", "practice", "protest", "punch", "pursuit",
            "question", "reaction", "request", "research", "response", "review",
            "run", "scream", "search", "shout", "shrug", "signal", "sleep",
            "smile", "speech", "step", "strategy", "study", "task", "test",
            "therapy", "tour", "training", "travel", "trip", "visit", "walk",
            "wave", "whisper", "work",
        ],
        "noun.feeling": [
            "affection", "anger", "anxiety", "apathy", "awe", "bitterness",
            "calm", "comfort", "compassion", "confidence", "contentment",
            "courage", "curiosity", "desire", "despair", "disappointment",
            "disgust", "doubt", "ecstasy", "embarrassment", "emotion",
            "empathy", "enthusiasm", "envy", "excitement", "fear", "feeling",
            "frustration", "gratitude", "grief", "guilt", "happiness", "hate",
            "hope", "horror", "hostility", "hunger", "hurt", "hysteria",
            "interest", "jealousy", "joy", "kindness", "liking", "loneliness",
            "love", "lust", "pain", "panic", "passion", "patience", "pity",
            "pleasure", "pride", "rage", "regret", "relief", "remorse",
            "resentment", "respect", "sadness", "satisfaction", "shame",
            "shock", "sorrow", "surprise", "sympathy", "tenderness", "terror",
            "thrill", "trust", "worry", "zeal",
        ],
        "noun.body": [
            "ankle", "arm", "back", "blood", "bone", "brain", "breast",
            "cheek", "chest", "chin", "ear", "elbow", "eye", "eyebrow",
            "eyelash", "face", "finger", "fingertip", "fist", "foot",
            "forehead", "gut", "hair", "hand", "head", "heart", "heel",
            "hip", "jaw", "joint", "kidney", "knee", "kneecap", "knuckle",
            "leg", "limb", "lip", "liver", "lung", "mouth", "muscle", "nail",
            "neck", "nerve", "nose", "organ", "palm", "pelvis", "rib",
            "shoulder", "skeleton", "skin", "skull", "spine", "stomach",
            "temple", "thigh", "throat", "thumb", "toe", "tongue", "tooth",
            "torso", "vein", "wrist",
        ],
        "noun.communication": [
            "abstract", "advertisement", "alphabet", "announcement", "article",
            "audio", "biography", "blog", "book", "broadcast", "bulletin",
            "caption", "chapter", "chat", "column", "comment", "communication",
            "conversation", "declaration", "description", "diary", "dictionary",
            "discussion", "document", "drama", "email", "essay", "explanation",
            "fiction", "file", "film", "grammar", "headline", "instruction",
            "interview", "joke", "language", "letter", "lyric", "magazine",
            "manual", "medium", "memo", "message", "monologue", "narrative",
            "news", "newspaper", "note", "notice", "novel", "paragraph",
            "phrase", "play", "poem", "post", "press", "prose", "publication",
            "quote", "radio", "recording", "remark", "report", "review",
            "rhyme", "saga", "script", "sentence", "signal", "sign", "song",
            "speech", "statement", "story", "summary", "symbol", "talk",
            "text", "thread", "title", "transcript", "translation", "video",
            "vocabulary", "voice", "web", "word", "writing",
        ],
        "noun.group": [
            "army", "audience", "band", "board", "brigade", "cast", "choir",
            "circle", "class", "club", "coalition", "committee", "community",
            "company", "congregation", "corporation", "council", "crew",
            "crowd", "delegation", "department", "division", "ensemble",
            "faculty", "family", "federation", "firm", "force", "fraction",
            "gang", "generation", "group", "guild", "institution", "league",
            "majority", "mob", "nation", "orchestra", "organization", "panel",
            "party", "population", "public", "squad", "staff", "society",
            "team", "tribe", "union",
        ],
        "noun.possession": [
            "asset", "belonging", "budget", "capital", "cash", "deposit",
            "earnings", "estate", "finance", "fortune", "fund", "goods",
            "grant", "inheritance", "investment", "loan", "money", "mortgage",
            "ownership", "payment", "pension", "property", "rent", "resource",
            "revenue", "royalty", "salary", "savings", "share", "stock",
            "tax", "treasure", "trust", "tuition", "value", "wage", "wealth",
            "welfare",
        ],
        "noun.quantity": [
            "amount", "batch", "bit", "bunch", "bundle", "capacity",
            "centimeter", "chunk", "couple", "cup", "dose", "drop", "dozen",
            "fraction", "gallon", "gram", "handful", "heap", "inch",
            "kilogram", "liter", "load", "lot", "mass", "measure", "meter",
            "mile", "million", "pair", "percent", "pile", "pinch", "portion",
            "pound", "quart", "quantity", "rate", "ratio", "score", "segment",
            "share", "size", "spoonful", "sum", "ton", "total", "unit",
            "volume", "weight", "yard",
        ],
        "noun.time": [
            "afternoon", "age", "anniversary", "autumn", "century", "date",
            "dawn", "day", "daylight", "decade", "dusk", "era", "evening",
            "future", "hour", "instant", "midnight", "minute", "moment",
            "month", "morning", "night", "noon", "past", "period", "phase",
            "present", "season", "second", "span", "spring", "summer",
            "sunrise", "sunset", "time", "today", "tomorrow", "tonight",
            "twilight", "week", "weekend", "while", "winter", "year",
            "yesterday",
        ],
        "noun.state": [
            "affair", "balance", "being", "calm", "chaos", "condition",
            "confusion", "crisis", "custom", "danger", "disorder", "emergency",
            "fashion", "form", "freedom", "habit", "health", "independence",
            "liberty", "life", "luck", "marriage", "mode", "mood", "nature",
            "need", "normal", "peace", "phase", "poverty", "power", "practice",
            "pregnancy", "quality", "readiness", "reality", "reign", "repair",
            "safety", "shape", "silence", "situation", "sleep", "state",
            "status", "style", "survival", "trend", "trouble", "truth",
            "uncertainty", "union", "void", "war", "wealth", "weather",
        ],
        "verb.cognition": [
            "analyze", "assess", "believe", "calculate", "categorize",
            "choose", "classify", "comprehend", "conceive", "conclude",
            "confirm", "consider", "decide", "deduce", "define", "determine",
            "discern", "distinguish", "doubt", "estimate", "evaluate",
            "expect", "figure", "forget", "grasp", "guess", "identify",
            "imagine", "infer", "inform", "interpret", "judge", "know",
            "learn", "mean", "memorize", "notice", "observe", "perceive",
            "plan", "predict", "prefer", "presume", "prove", "rationalize",
            "realize", "reason", "recall", "recognize", "recollect",
            "remember", "resolve", "speculate", "study", "suppose", "suspect",
            "think", "trust", "understand", "verify",
        ],
        "verb.communication": [
            "acknowledge", "address", "admit", "advise", "affirm", "allege",
            "announce", "answer", "apologize", "argue", "ask", "assert",
            "beg", "boast", "call", "cite", "claim", "comment", "communicate",
            "complain", "confess", "confirm", "contradict", "converse",
            "declare", "demand", "deny", "describe", "discuss", "exclaim",
            "explain", "express", "gossip", "greet", "imply", "inform",
            "inquire", "insist", "instruct", "interrupt", "introduce",
            "invite", "lecture", "mention", "narrate", "negotiate", "notify",
            "offer", "order", "persuade", "praise", "pray", "proclaim",
            "propose", "question", "quote", "read", "recommend", "remark",
            "reply", "report", "request", "respond", "reveal", "say", "speak",
            "state", "suggest", "summarize", "swear", "teach", "tell",
            "testify", "thank", "urge", "vow", "warn", "whisper", "write",
            "yell",
        ],
        "verb.emotion": [
            "admire", "adore", "amaze", "amuse", "anger", "annoy",
            "appreciate", "astonish", "bore", "calm", "charm", "cheer",
            "comfort", "confuse", "delight", "depress", "despise",
            "disappoint", "distress", "disturb", "embarrass", "encourage",
            "enjoy", "enthuse", "excite", "fascinate", "fear", "frighten",
            "grieve", "hate", "horrify", "humiliate", "impress", "inspire",
            "insult", "interest", "intimidate", "irritate", "jeopardize",
            "like", "love", "motivate", "move", "obsess", "offend", "panic",
            "pity", "please", "prefer", "provoke", "reassure", "regret",
            "rejoice", "relax", "relieve", "respect", "sadden", "satisfy",
            "scare", "shock", "soothe", "stimulate", "stir", "surprise",
            "terrify", "thrill", "tire", "touch", "trouble", "trust", "upset",
            "worry",
        ],
        "verb.body": [
            "breathe", "burp", "cough", "cry", "digest", "drip", "eat",
            "exercise", "faint", "feed", "freeze", "gasp", "hiccup", "hurt",
            "inhale", "itch", "perspire", "pulse", "scream", "shiver", "sigh",
            "sleep", "smell", "sneeze", "snore", "stretch", "suffer", "sweat",
            "swallow", "taste", "throb", "tickle", "tremble", "vomit", "wake",
            "weep", "yawn",
        ],
        "verb.change": [
            "accelerate", "adapt", "adjust", "alter", "amplify", "break",
            "build", "change", "compress", "convert", "cool", "correct",
            "crumble", "crush", "decrease", "deepen", "destroy", "diminish",
            "dissolve", "distort", "diversify", "double", "drop", "elevate",
            "enhance", "enlarge", "evolve", "expand", "explode", "extend",
            "fade", "fix", "flatten", "freeze", "grow", "heat", "improve",
            "increase", "inflate", "lessen", "lengthen", "loosen", "lower",
            "mature", "mend", "melt", "moderate", "modify", "multiply",
            "narrow", "normalize", "open", "raise", "reduce", "reform",
            "remodel", "repair", "reshape", "reverse", "revise", "rotate",
            "shorten", "shrink", "slow", "soften", "solidify", "stabilize",
            "strengthen", "stretch", "toughen", "transform", "triple", "vary",
            "widen", "worsen",
        ],
        "verb.motion": [
            "advance", "approach", "arrive", "ascend", "back", "bend",
            "bounce", "bow", "climb", "crawl", "creep", "cross", "dance",
            "dash", "depart", "descend", "drag", "drift", "drop", "dive",
            "emerge", "enter", "escape", "exit", "fall", "flee", "float",
            "flow", "flip", "fly", "gallop", "glide", "go", "hike", "hurry",
            "jump", "kneel", "land", "leap", "leave", "limp", "march", "move",
            "pass", "plunge", "pounce", "pull", "pump", "push", "race",
            "reach", "return", "rise", "roll", "run", "rush", "sail", "shake",
            "shift", "shrink", "sink", "skip", "slide", "slip", "slope",
            "sneak", "spin", "sprint", "squeeze", "squirm", "stand", "step",
            "stir", "stretch", "stride", "stroll", "sway", "swim", "swing",
            "turn", "twirl", "twist", "vibrate", "walk", "wander", "wave",
            "whirl", "wriggle",
        ],
        "verb.stative": [
            "appear", "awake", "be", "become", "belong", "consist",
            "constitute", "contain", "cost", "depend", "deserve", "equal",
            "exist", "fit", "have", "hold", "include", "involve", "lack",
            "last", "lie", "live", "look", "make", "match", "mean", "measure",
            "need", "owe", "own", "possess", "remain", "represent", "require",
            "resemble", "seem", "signify", "sound", "stay", "suffice", "tend",
            "weigh",
        ],
        "verb.social": [
            "adopt", "appoint", "assist", "betray", "bribe", "care",
            "cooperate", "donate", "employ", "fund", "govern", "hire", "host",
            "join", "lead", "manage", "marry", "mentor", "nominate", "nurture",
            "obey", "oppose", "organize", "participate", "protest", "recruit",
            "reject", "represent", "resign", "rule", "serve", "sponsor",
            "support", "tolerate", "volunteer",
        ],
        "verb.possess": [
            "acquire", "buy", "collect", "earn", "exchange", "gain", "get",
            "give", "keep", "lend", "loan", "obtain", "own", "possess",
            "purchase", "receive", "rent", "sell", "trade", "win",
        ],
        "verb.contact": [
            "attach", "bang", "beat", "bite", "blow", "break", "bump",
            "catch", "chop", "clap", "clean", "clip", "connect", "cover",
            "crush", "cut", "dig", "drag", "draw", "drill", "drop", "fasten",
            "fold", "grab", "grasp", "grind", "grip", "hit", "hold", "hook",
            "hug", "jam", "kick", "kiss", "knock", "lift", "link", "load",
            "lock", "merge", "mix", "pack", "pat", "peel", "pick", "pile",
            "pinch", "poke", "polish", "press", "pull", "punch", "push",
            "rub", "scratch", "screw", "scrub", "seal", "shake", "slam",
            "slap", "smash", "smooth", "snap", "squeeze", "stack", "stick",
            "stir", "strike", "stroke", "sweep", "swipe", "tie", "touch",
            "twist", "unfold", "unlock", "wash", "wipe", "wrap",
        ],
        "verb.creation": [
            "assemble", "bake", "build", "carve", "compile", "compose",
            "construct", "craft", "create", "cultivate", "design", "devise",
            "draw", "edit", "erect", "fabricate", "forge", "form", "generate",
            "grow", "invent", "knit", "make", "manufacture", "mold",
            "originate", "paint", "produce", "sculpt", "sew", "shape", "spawn",
            "weave", "write",
        ],
        "verb.perceive": [
            "detect", "discover", "distinguish", "feel", "glimpse", "hear",
            "identify", "listen", "notice", "observe", "overhear", "perceive",
            "recognize", "see", "sense", "smell", "spot", "taste", "view",
            "watch", "witness",
        ],
        "verb.consume": [
            "absorb", "bite", "brew", "chew", "consume", "cook", "devour",
            "dine", "drink", "eat", "feast", "feed", "gulp", "munch", "nibble",
            "sip", "slurp", "snack", "swallow", "taste",
        ],
        "verb.weather": [
            "drizzle", "freeze", "frost", "hail", "lighten", "pour", "rain",
            "sleet", "snow", "storm", "thunder",
        ],
    }

    added_words = set()
    for ss, words in EXPANDED_WORDS.items():
        for w in words:
            if w not in added_words:
                pos = "noun" if ss.startswith("noun.") else "verb"
                word_supersense.append({"word": w, "supersense": ss, "pos": pos})
                added_words.add(w)

    seen = set()
    unique_entries = []
    for e in word_supersense:
        key = (e["word"], e["supersense"])
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    valid_ss = set(supersense_to_melm.keys())
    valid_entries = [e for e in unique_entries if e["supersense"] in valid_ss]

    verbnet_map = load_contract_json("verbnet_map.v1.json")
    verbnet_to_melm = dict(verbnet_map.get("mappings", {}))

    melm_to_verbnet = {}
    for vn_cls, mc in verbnet_to_melm.items():
        melm_to_verbnet.setdefault(mc, []).append(vn_cls)

    verb_entries = []
    for verb_name, (_canonical, legacy_class) in sorted(_VERBS.items()):
        melm_class = LEGACY_VERB_CLASS_MAP.get(legacy_class)
        if melm_class:
            vn_classes = melm_to_verbnet.get(melm_class, [])
            if vn_classes:
                verb_entries.append({
                    "verb": verb_name,
                    "verbnet_class": vn_classes[0],
                    "pos": "verb",
                })

    data_dir = Path(__file__).resolve().parent.parent / "melm" / "contracts"
    with open(data_dir / "word_supersense_data.v1.jsonl", "w", encoding="utf-8") as f:
        for e in valid_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(data_dir / "verb_data.v1.jsonl", "w", encoding="utf-8") as f:
        for e in verb_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Generated {len(valid_entries)} word-supersense entries "
          f"({len(unique_entries) - len(valid_entries)} skipped, unknown supersense)")
    print(f"Generated {len(verb_entries)} verb-verbnet-class entries")
    print(f"Wrote to {data_dir}")


if __name__ == "__main__":
    main()
