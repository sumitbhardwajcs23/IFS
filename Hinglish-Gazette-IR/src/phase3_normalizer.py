"""Phase 3: Phonetic and dialect-aware Hinglish query normalization and bilingual expansion."""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import Any

# Hinglish + English stopwords (kept out of BM25 keyword queries, retained in semantic text)
HINGLISH_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will",
    "with", "what", "how", "when", "where", "which", "who", "whom",
    # Hindi/Hinglish postpositions, auxiliaries & question words
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "per", "tak",
    "hai", "hain", "tha", "the", "thi", "ho", "hota", "hoti", "hote", "hoga", "hogi",
    "kya", "kaise", "kaisa", "kaisi", "kitna", "kitni", "kitne", "kab", "kahan", "kaun",
    "aur", "ya", "evam", "tatha", "bhi", "hi", "to", "toh", "agar", "yadi",
    "ye", "yeh", "wo", "woh", "is", "us", "in", "un", "ek", "koi", "kuch",
    "milti", "milta", "milte", "milegi", "milega", "diya", "di", "kar", "karna",
    "karke", "liye", "liya", "wala", "wali", "wale", "baare", "bare", "sab",
}

# Bilingual & dialect lexicon mapping Hinglish / regional / informal terms
# to canonical English + Devanagari Hindi legal/gazette concepts.
HINGLISH_GAZETTE_LEXICON: dict[str, list[str]] = {
    # Employment, Service, Leave, Promotion, Transfer
    "mahila": ["women", "female", "woman", "महिला", "स्त्री"],
    "aurat": ["women", "female", "महिला"],
    "stri": ["women", "female", "महिला"],
    "nari": ["women", "female", "महिला"],
    "karmchari": ["employee", "staff", "officer", "कर्मचारी", "कार्मिक"],
    "karmachari": ["employee", "staff", "कर्मचारी"],
    "mulazim": ["employee", "servant", "कर्मचारी"],
    "naukar": ["employee", "servant", "कर्मचारी"],
    "adhikari": ["officer", "authority", "अधिकारी"],
    "afsar": ["officer", "अधिकारी"],
    "sarkari": ["government", "state", "public", "सरकारी", "राजकीय", "शासकीय"],
    "shaskiya": ["government", "official", "शासकीय", "सरकारी"],
    "rajkiya": ["government", "state", "राजकीय"],
    "naukri": ["recruitment", "appointment", "service", "employment", "नियुक्ति", "भर्ती", "सेवा"],
    "nokri": ["recruitment", "appointment", "service", "नियुक्ति", "भर्ती"],
    "rojgar": ["employment", "service", "रोजगार", "सेवा"],
    "bharti": ["recruitment", "selection", "appointment", "भर्ती", "चयन", "नियुक्ति"],
    "chayan": ["selection", "recruitment", "चयन"],
    "niyukti": ["appointment", "posting", "नियुक्ति", "तैनाती"],
    "tainati": ["posting", "deployment", "तैनाती"],
    "padonnati": ["promotion", "पदोन्नति"],
    "taraqqi": ["promotion", "पदोन्नति"],
    "tarakki": ["promotion", "पदोन्नति"],
    "tabadla": ["transfer", "posting", "स्थानान्तरण", "तैनाती"],
    "badli": ["transfer", "स्थानान्तरण"],
    "sthanantaran": ["transfer", "स्थानान्तरण"],
    "chhutti": ["leave", "holiday", "अवकाश", "छुट्टी"],
    "chutti": ["leave", "holiday", "अवकाश"],
    "avkash": ["leave", "अवकाश"],
    "rukhsat": ["leave", "अवकाश"],
    "vetan": ["salary", "pay", "pay-matrix", "scale", "वेतन", "पे-मैट्रिक्स"],
    "tankhwa": ["salary", "pay", "वेतन"],
    "tankhwah": ["salary", "pay", "वेतन"],
    "pagar": ["salary", "pay", "वेतन"],
    "bhatta": ["allowance", "भत्ता"],
    "pension": ["pension", "retirement", "पेंशन", "सेवानिवृत्ति"],
    "sevanivritti": ["retirement", "retired", "सेवानिवृत्ति"],
    "retire": ["retirement", "retired", "सेवानिवृत्ति"],

    # Maternity, Family, Health, Welfare
    "prasuti": ["maternity", "childbirth", "प्रसूति", "मातृत्व"],
    "matritva": ["maternity", "मातृत्व", "प्रसूति"],
    "baccha": ["child", "infant", "बच्चा", "शिशु"],
    "bachcha": ["child", "infant", "बच्चा"],
    "janm": ["birth", "जन्म"],
    "garbhavati": ["pregnant", "maternity", "गर्भवती", "प्रसूति"],
    "kalyan": ["welfare", "कल्याण"],
    "swasthya": ["health", "medical", "स्वास्थ्य", "चिकित्सा"],
    "chikitsa": ["medical", "hospital", "चिकित्सा", "चिकित्सालय"],
    "aspatal": ["hospital", "चिकित्सालय", "अस्पताल"],
    "ayurved": ["ayurvedic", "ayurveda", "आयुर्वेद", "आयुर्वेदिक"],

    # Land, Revenue, Acquisition, Agriculture (major UP Gazette topics)
    "jameen": ["land", "area", "hectare", "भूमि", "जमीन", "क्षेत्रफल"],
    "zameen": ["land", "area", "hectare", "भूमि", "क्षेत्रफल"],
    "bhumi": ["land", "revenue", "भूमि", "राजस्व"],
    "bhoomi": ["land", "भूमि"],
    "khet": ["land", "agricultural", "कृषि", "खेत"],
    "adhigrahan": ["acquisition", "acquired", "अधिग्रहण", "अर्जन"],
    "arjan": ["acquisition", "अर्जन", "अधिग्रहण"],
    "kabza": ["possession", "occupation", "कब्जा"],
    "muavja": ["compensation", "मुआवजा", "प्रतिकर"],
    "muawaza": ["compensation", "मुआवजा", "प्रतिकर"],
    "pratikar": ["compensation", "प्रतिकर"],
    "rajaswa": ["revenue", "rajaswa", "राजस्व"],
    "malguzari": ["revenue", "land revenue", "राजस्व"],
    "chakbandi": ["consolidation", "chakbandi", "चकबन्दी"],
    "kisan": ["farmer", "cultivator", "कृषक", "किसान"],
    "krishi": ["agriculture", "agricultural", "कृषि"],
    "gram": ["village", "gram", "ग्राम", "गांव"],
    "gaon": ["village", "gram", "ग्राम", "गाँव"],
    "panchayat": ["panchayat", "gram panchayat", "पंचायत"],
    "tehsil": ["tehsil", "subdivision", "तहसील"],
    "janpad": ["district", "janpad", "जनपद", "जिला"],
    "jila": ["district", "जनपद", "जिला"],
    "zilla": ["district", "जनपद", "जिला"],
    "sinchai": ["irrigation", "water", "सिंचाई"],
    "pani": ["water", "जल", "सिंचाई"],
    "nagar": ["municipal", "municipality", "corporation", "नगर", "निगम", "निकाय"],
    "shehar": ["urban", "municipal", "नगर"],
    "nikay": ["ulb", "municipality", "body", "निकाय"],

    # Legal, Judicial, Rules, Orders, Notifications
    "niyam": ["rule", "rules", "regulation", "नियम", "नियमावली"],
    "niyamavali": ["rules", "regulations", "नियमावली"],
    "kanun": ["act", "law", "statute", "अधिनियम", "विधि", "कानून"],
    "kanoon": ["act", "law", "statute", "अधिनियम", "विधि"],
    "vidhi": ["law", "act", "procedure", "विधि"],
    "adhiniyam": ["act", "statute", "अधिनियम"],
    "dhara": ["section", "clause", "धारा"],
    "aadesh": ["order", "direction", "आदेश", "शासनादेश"],
    "adesh": ["order", "आदेश"],
    "shasanadesh": ["government order", "order", "शासनादेश"],
    "hukm": ["order", "आदेश"],
    "adhisochna": ["notification", "अधिसूचना"],
    "adhisuchna": ["notification", "अधिसूचना"],
    "vigyapti": ["notification", "notice", "विज्ञप्ति"],
    "paripatra": ["circular", "परिपत्र"],
    "sanshodhan": ["amendment", "संशोधन"],
    "badlav": ["amendment", "change", "संशोधन"],
    "nyayalaya": ["court", "judicial", "न्यायालय"],
    "adalat": ["court", "judge", "न्यायालय", "न्यायाधीश"],
    "kachahari": ["court", "न्यायालय"],
    "judge": ["judge", "magistrate", "judicial", "न्यायाधीश"],
    "nyayadhish": ["judge", "magistrate", "न्यायाधीश"],
    "vibhag": ["department", "ministry", "विभाग"],
    "mantralaya": ["ministry", "department", "मंत्रालय", "विभाग"],
    "mahakma": ["department", "विभाग"],
    "shulk": ["fee", "duty", "charge", "शुल्क"],
    "kar": ["tax", "duty", "कर"],
    "vitta": ["finance", "वित्त"],
    "paisa": ["finance", "amount", "rupees", "वित्त", "रु०"],
    "naam": ["name", "नाम"],
    "name": ["name", "नाम"],
    "shart": ["condition", "conditions", "eligibility", "शर्त", "शर्तें"],
    "yogyata": ["eligibility", "qualification", "योग्यता", "अर्हता"],
    "umr": ["age", "आयु"],
    "aayu": ["age", "आयु"],

    # Common English spelling variations found in Hinglish typing
    "benifit": ["benefit", "benefits"],
    "benifits": ["benefits", "benefit"],
    "materniti": ["maternity"],
    "matrnity": ["maternity"],
    "notificaton": ["notification"],
    "notifikeshan": ["notification"],
    "recrutment": ["recruitment"],
    "rekruitment": ["recruitment"],
    "goverment": ["government"],
    "govrnment": ["government"],
    "employe": ["employee", "employees"],
    "employes": ["employees"],
    "salery": ["salary"],
    "penshan": ["pension"],
    "kompensation": ["compensation"],
    "akuisition": ["acquisition"],
}

_TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[a-zA-Z0-9]+(?:[\-/\.][a-zA-Z0-9]+)*")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def hinglish_phonetic_key(token: str) -> str:
    """Compute a phonetic signature tailored to Romanized Hindi + English spelling variants."""
    word = re.sub(r"[^a-z0-9]", "", token.lower())
    if not word:
        return ""
    if word.isdigit():
        return word

    # Normalize aspirated & consonant clusters common in Hinglish transliteration
    replacements = [
        ("chh", "c"),
        ("ch", "c"),
        ("sh", "s"),
        ("bh", "b"),
        ("ph", "f"),
        ("kh", "k"),
        ("gh", "g"),
        ("jh", "j"),
        ("th", "t"),
        ("dh", "d"),
        ("ksh", "x"),
        ("gya", "g"),
        ("ee", "i"),
        ("ii", "i"),
        ("oo", "u"),
        ("uu", "u"),
        ("aa", "a"),
        ("ai", "e"),
        ("ay", "e"),
        ("au", "o"),
        ("aw", "o"),
        ("w", "v"),
        ("z", "j"),
        ("q", "k"),
        ("x", "ks"),
    ]
    for src, dst in replacements:
        word = word.replace(src, dst)

    # Soft c -> s before front vowels, else k
    word = re.sub(r"c(?=[eiy])", "s", word)
    word = word.replace("c", "k")

    # Collapse consecutive duplicate characters (e.g. chhutti -> cuti)
    word = re.sub(r"(.)\1+", r"\1", word)

    # Keep leading vowel if present, drop medial/trailing vowels for consonant skeleton
    if len(word) > 2:
        head, tail = word[0], re.sub(r"[aeiou]", "", word[1:])
        skeleton = head + tail
    else:
        skeleton = word

    return skeleton or word


# Precompute phonetic lookup index over HINGLISH_GAZETTE_LEXICON keys
_LEXICON_KEYS = sorted(HINGLISH_GAZETTE_LEXICON.keys())
_PHONETIC_TO_LEXICON: dict[str, list[str]] = {}
for _k in _LEXICON_KEYS:
    _pk = hinglish_phonetic_key(_k)
    if _pk:
        _PHONETIC_TO_LEXICON.setdefault(_pk, []).append(_k)


def _match_lexicon_term(token: str) -> tuple[str | None, str]:
    """Match a Latin token to the lexicon via exact, phonetic, or fuzzy similarity."""
    lower = token.lower()
    if lower in HINGLISH_GAZETTE_LEXICON:
        return lower, "exact"

    # Exact phonetic skeleton match
    pk = hinglish_phonetic_key(lower)
    if pk and pk in _PHONETIC_TO_LEXICON and len(lower) >= 3:
        candidates = _PHONETIC_TO_LEXICON[pk]
        best = max(
            candidates,
            key=lambda c: difflib.SequenceMatcher(None, lower, c).ratio(),
        )
        if difflib.SequenceMatcher(None, lower, best).ratio() >= 0.62:
            return best, "phonetic"

    # Character-level fuzzy fallback for typos (e.g. 'matrnity', 'notificaton')
    if len(lower) >= 5:
        close = difflib.get_close_matches(lower, _LEXICON_KEYS, n=1, cutoff=0.80)
        if close:
            return close[0], "fuzzy"

    return None, ""


@dataclass
class NormalizedQuery:
    """Structured output from Phase 3 Hinglish query normalization."""

    original_query: str
    cleaned_query: str
    tokens: list[str]
    corrected_tokens: list[str]
    dialect_expansions: dict[str, list[str]]
    phonetic_corrections: dict[str, str]
    bm25_tokens: list[str]
    semantic_query: str
    explanation: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HinglishNormalizer:
    """Phonetic and dialect-aware normalizer for code-mixed Hinglish Gazette queries."""

    def normalize(self, query: str) -> NormalizedQuery:
        cleaned = re.sub(r"\s+", " ", (query or "").strip())
        raw_tokens = _TOKEN_RE.findall(cleaned)

        corrected_tokens: list[str] = []
        dialect_expansions: dict[str, list[str]] = {}
        phonetic_corrections: dict[str, str] = {}
        bm25_tokens: list[str] = []
        semantic_hints: list[str] = []
        explanation: list[str] = []

        seen_bm25: set[str] = set()
        seen_hints: set[str] = set()

        def add_bm25(tok: str) -> None:
            t = tok.lower()
            if t and t not in seen_bm25:
                seen_bm25.add(t)
                bm25_tokens.append(t)

        for tok in raw_tokens:
            if _DEVANAGARI_RE.search(tok):
                corrected_tokens.append(tok)
                add_bm25(tok)
                continue

            lower = tok.lower()
            if lower in HINGLISH_STOPWORDS:
                continue

            matched_key, match_type = _match_lexicon_term(lower)
            if matched_key:
                if match_type in {"phonetic", "fuzzy"} and matched_key != lower:
                    phonetic_corrections[lower] = matched_key
                    explanation.append(f"Phonetic/spelling match: '{lower}' -> '{matched_key}'")
                corrected_tokens.append(matched_key)
                add_bm25(lower)
                add_bm25(matched_key)

                expansions = HINGLISH_GAZETTE_LEXICON[matched_key]
                dialect_expansions[matched_key] = expansions
                explanation.append(f"Dialect/code-mix map: '{matched_key}' -> {', '.join(expansions)}")
                for exp in expansions:
                    for sub_tok in _TOKEN_RE.findall(exp):
                        add_bm25(sub_tok)
                    if exp not in seen_hints:
                        seen_hints.add(exp)
                        semantic_hints.append(exp)
            else:
                corrected_tokens.append(lower)
                add_bm25(lower)

            # Include phonetic skeleton for Latin words >= 4 chars for fuzzy BM25 matching
            if len(lower) >= 4 and not lower.isdigit():
                pk = hinglish_phonetic_key(lower)
                if pk and len(pk) >= 3:
                    add_bm25(f"ph_{pk}")

        if semantic_hints:
            semantic_query = f"{cleaned} ({', '.join(semantic_hints[:14])})"
        else:
            semantic_query = cleaned

        return NormalizedQuery(
            original_query=query,
            cleaned_query=cleaned,
            tokens=raw_tokens,
            corrected_tokens=corrected_tokens,
            dialect_expansions=dialect_expansions,
            phonetic_corrections=phonetic_corrections,
            bm25_tokens=bm25_tokens,
            semantic_query=semantic_query,
            explanation=explanation,
        )


def tokenize_document_for_bm25(text: str) -> list[str]:
    """Tokenize Gazette passage text into normalized words + phonetic signatures."""
    tokens = _TOKEN_RE.findall(text or "")
    out: list[str] = []
    for tok in tokens:
        if _DEVANAGARI_RE.search(tok):
            out.append(tok)
            continue
        lower = tok.lower()
        if lower in HINGLISH_STOPWORDS or len(lower) <= 1:
            continue
        out.append(lower)
        if len(lower) >= 4 and not lower.isdigit():
            pk = hinglish_phonetic_key(lower)
            if pk and len(pk) >= 3:
                out.append(f"ph_{pk}")
    return out
