"""Deterministic Kruti Dev 010 (legacy ASCII-mapped Hindi) to Unicode Devanagari converter."""

from __future__ import annotations

import re

# Ordered multi-character to single-character replacements (longest / specific patterns first).
# Based on standard Kruti Dev 010 -> Unicode Devanagari mapping.
_KRUTIDEV_REPLACEMENTS: list[tuple[str, str]] = [
    # Punctuation & special symbols
    ("ñ", "॰"),
    ("Q+Z", "फ़र्"),
    ("sas", "से"),  # rare artifact guard? No: 's' is े, 'a' is ं, keep atomic below
]

# Clean atomic replacement list (exact Kruti Dev 010 table)
_SUBS: list[tuple[str, str]] = [
    ("'", "श्"),
    ('"', "ष्"),
    ("¼", "("),
    ("½", ")"),
    ("¿", "{"),
    ("À", "}"),
    ("¾", "="),
    ("A", "।"),
    ("B", "ठ"),
    ("C", "ब"),
    ("D", "क"),
    ("E", "म"),
    ("F", "थ्"),
    ("G", "ळ"),
    ("H", "भ्"),
    ("I", "प्"),
    ("J", "श्र"),
    ("K", "ज्ञ"),
    ("L", "स्"),
    ("M", "ड"),
    ("N", "छ"),
    ("O", "व्"),
    ("P", "च्"),
    ("Q", "फ"),
    ("R", "त्"),
    ("S", "ए"),
    ("T", "ज्"),
    ("U", "न्"),
    ("V", "ट"),
    ("W", "ॅ"),
    ("X", "ग्"),
    ("Y", "ल्"),
    ("Z", "र्"),  # placeholder reph handled in post-pass; keep marker first
]

# Actually let's use the exact, conflict-free Kruti Dev 010 ordered mapping table
# where multi-char sequences come strictly before single-char sequences!
_KD_CHARS_FROM = [
    "ñ", "Q+Z", "sas", "aa", ")Z", "ZZ", "‘", "’", "“", "”",
    "å", "ƒ", "„", "…", "†", "‡", "ˆ", "‰", "Š", "‹",
    "¶+", "d+", "[+k", "[+", "x+", "T+", "t+", "M+", "<+", "Q+", ";+", "j+", "u+",
    "Ùk", "Ù", "Dr", "–", "—", "é", "™", "=kk", "f=k",
    "à", "á", "â", "ã", "ºz", "º", "í", "{k", "{", "=", "«",
    "Nî", "Vî", "Bî", "Mî", "<î", "|", "K", "}",
    "J", "Vª", "Mª", "<ªª", "Nª", "Ø", "Ý", "nzZ", "æ", "ç", "Á", "xz", "#", ":",
    "v‚", "vks", "vkS", "vk", "v", "b±", "Ã", "bZ", "b", "m", "Å", ",s", ",", "_",
    "ô", "d", "Dk", "D", "[k", "[", "x", "Xk", "X", "Ä", "?k", "?", "³",
    "pkS", "p", "Pk", "P", "N", "t", "Tk", "T", ">", "÷", "¥",
    "ê", "ë", "V", "B", "ì", "ï", "M+", "<+", "M", "<", ".k", ".",
    "r", "Rk", "R", "Fk", "F", ")", "n", "/k", "èk", "/", "Ë", "è", "u", "Uk", "U",
    "i", "Ik", "I", "Q", "¶", "c", "Ck", "C", "Hk", "H", "e", "Ek", "E",
    ";", "¸", "j", "y", "Yk", "Y", "G", "o", "Ok", "O",
    "'k", "'", "\"k", "\"", "l", "Lk", "L", "g",
    "È", "z",
    "Ì", "Í", "Î", "Ï", "Ñ", "Ò", "Ó", "Ô", "Ö", "Ø", "Ù", "Ük", "Ü",
    "‚", "ks", "kS", "k", "h", "q", "w", "`", "s", "S",
    "a", "¡", "%", "W", "•", "·", "∙", "·", "~j", "~", "\\", "+", " ः",
    "^", "*", "Þ", "ß", "(", "¼", "½", "¿", "À", "¾", "A", "-", "&", "&", "Œ", "]", "~ ", "@",
]

_KD_CHARS_TO = [
    "॰", "QZ+", "sa", "a", "र्द्ध", "Z", "\"", "\"", "'", "'",
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
    "फ़्", "क़", "ख़", "ख़्", "ग़", "ज़्", "ज़", "ड़", "ढ़", "फ़", "य़", "ऱ", "ऩ",
    "त्त", "त्त्", "क्त", "दृ", "कृ", "न्न", "न्न्", "=k", "f=",
    "ह्न", "ह्य", "हृ", "ह्म", "ह्र", "ह्", "द्द", "क्ष", "क्ष्", "त्र", "त्र्",
    "छ्य", "ट्य", "ठ्य", "ड्य", "ढ्य", "द्य", "ज्ञ", "द्व",
    "श्र", "ट्र", "ड्र", "ढ्र", "छ्र", "क्र", "फ्र", "र्द्र", "द्र", "प्र", "प्र", "ग्र", "रु", "रू",
    "ऑ", "ओ", "औ", "आ", "अ", "ईं", "ई", "ई", "इ", "उ", "ऊ", "ऐ", "ए", "ऋ",
    "क्क", "क", "क", "क्", "ख", "ख्", "ग", "ग", "ग्", "घ", "घ", "घ्", "ङ",
    "चै", "च", "च", "च्", "छ", "ज", "ज", "ज्", "झ", "झ्", "ञ",
    "ट्ट", "ट्ठ", "ट", "ठ", "ड्ड", "ड्ढ", "ड़", "ढ़", "ड", "ढ", "ण", "ण्",
    "त", "त", "त्", "थ", "थ्", "द्ध", "द", "ध", "ध", "ध्", "ध्", "ध्", "न", "न", "न्",
    "प", "प", "प्", "फ", "फ्", "ब", "ब", "ब्", "भ", "भ्", "म", "म", "म्",
    "य", "य्", "र", "ल", "ल", "ल्", "ळ", "व", "व", "व्",
    "श", "श्", "ष", "ष्", "स", "स", "स्", "ह",
    "ीं", "्र",
    "द्द", "ट्ट", "ट्ठ", "ड्ड", "कृ", "भ", "्य", "ड्ढ", "झ्", "क्र", "त्त्", "श", "श्",
    "ॉ", "ो", "ौ", "ा", "ी", "ु", "ू", "ृ", "े", "ै",
    "ं", "ँ", "ः", "ॅ", "ऽ", "ऽ", "ऽ", "ऽ", "्र", "्", "?", "़", ":",
    "‘", "’", "“", "”", ";", "(", ")", "{", "}", "=", "।", "-", "-", "?", "ॐ", ",", "् ", "/",
]

# Wait: notice index for "Œ" above had a typo ("Invoices" -> should be "॰" or "ॐ" / "।" — let's fix "Œ" and verify!)
# Also notice "Ÿ" (U+0178) in `mŸkj`! Let's check exact mapping for `Ÿ`, `Œ`, `š`, `œ`, `ž` (Windows-1252 C1 range)!
# In Windows-1252:
#   0x8C = Œ, 0x9A = š, 0x9C = œ, 0x9E = ž, 0x9F = Ÿ (same as Ù in some Mac/PDF encodings: `Ù` is 0xD9, `Ÿ` is 0x9F!)
# Look at our PDF output earlier:
#   `mŸkj` -> `उत्तर` (`Ÿ` = `त्त्` -> `Ÿk` = `त्त`!)
#   `mÙkj` -> `उत्तर` (`Ù` = `त्त्` -> `Ùk` = `त्त`!)
#   `'kfuokj` -> `शनिवार` (`'` = `श्`, `'k` = `श`)
#   `¼ykblsULM Vw iksLV fcnkmV izhisesUV½` -> `(लाइसेन्स्ड टू पोस्ट विदाउट प्रीपेमेन्ट)` (wait: `fcnkmV` = `विदाउट`, `izhisesUV` = `प्रीपेमेन्ट`)
#   `vDVwcj` -> `अक्टूबर` (`v` = `अ`, `D` = `क्`, `V` = `ट`, `w` = `ू`, `c` = `ब`, `j` = `र`)
#   `pUnk` -> `चन्दा` (`p` = `च`, `U` = `न्`, `n` = `द`, `k` = `ा`)
#   `i`"B la[;k` -> `पृष्ठ संख्या` (`i` = `प`, `` ` `` = `ृ`, `"` = `ष्`, `B` = `ठ` -> `पृष्ठ`; `l` = `स`, `a` = `ं`, `[` = `ख्`, `;` = `य`, `k` = `ा` -> `संख्या`!)
#   `>k¡lh` -> `झाँसी` (`>k` or `>` + `k`: wait in table above `>` -> `झ`, so `>k` would give `झा` unless `>k` -> `झ`! Let's check!)
# Look closely at `>k¡lh` in `C_202611181059330770.pdf`:
#   >k¡lh = झाँसी (Jhansi), NOT झाँासी! That means `>` in Kruti Dev is `झ्` (half jha) or `>k` = `झ`!
#   Wait: let's check `>` in Kruti Dev 010: Shift+. (`>`) IS `झ` in some versions or `झ्`? Wait: in our PDF, `>k¡lh` is literally `>k` + `¡` + `l` + `h` = `झ` + `ा`? NO, Jhansi has NO aa-matra on झ — it's `झाँसी`! Wait: Jhansi IS `झाँसी` (झ + ा + ँ + स + ी)!! HAHA! Yes! `झ` + `ा` + `ँ` + `स` + `ी` = `झाँसी`!
#   What about `dk;Z-fof/k;ka`? `d` (`क`) + `k` (`ा`) + `;` (`य`) + `Z` (`र्`) = `कार्य`! `f` + `o` (`व`) = `वि`, `f` + `/k` (`ध्` + `ा` = `ध`) = `धि`, `;` (`य`) + `k` (`ा`) + `a` (`ं`) = `यां` -> `कार्य-विधियां`!

_EXTRA_SUBS: list[tuple[str, str]] = [
    ("Ÿk", "त्त"),
    ("Ÿ", "त्त्"),
    ("Œ", "ॐ"),
    ("œ", "त्त्"),
    ("š", "श्"),
]

# Build combined ordered replacement pairs
assert len(_KD_CHARS_FROM) == len(_KD_CHARS_TO)
_ORDERED_PAIRS: list[tuple[str, str]] = []
for src, dst in zip(_KD_CHARS_FROM, _KD_CHARS_TO):
    if src == "Œ":
        _ORDERED_PAIRS.append(("Œ", "ॐ"))
    elif src == "sas":
        continue  # handled naturally by 'sa' + 's' or keep? Wait: in Kruti Dev, 'sas' is e.g. 'मंे'? No, 'esa' = में ('e'='म', 's'='े', 'a'='ं'). Do NOT replace 'sas' globally!
    elif src == "aa":
        _ORDERED_PAIRS.append(("aa", "ं"))
    else:
        _ORDERED_PAIRS.append((src, dst))

_ORDERED_PAIRS = _EXTRA_SUBS + _ORDERED_PAIRS

# Build single-pass regex for longest match among keys (preserving order for equal lengths)
# Wait: sorting keys by descending length guarantees multi-char tokens like 'vkS', 'pkS', 'f=k', 'Ÿk', '>k' match before single chars!
_UNIQUE_MAP: dict[str, str] = {}
for src, dst in _ORDERED_PAIRS:
    if src not in _UNIQUE_MAP:
        _UNIQUE_MAP[src] = dst

_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_UNIQUE_MAP.keys(), key=len, reverse=True))
)

# Matra & Reph reordering regexes
# 1. In Kruti Dev, 'f' (छोटी इ की मात्रा) appears BEFORE the consonant/conjunct.
#    Note: 'f' was NOT in _KD_CHARS_FROM, so it stays literal 'f' until reordering!
_CHHOTI_I_RE = re.compile(
    r"f((?:[\u0915-\u0939\u0958-\u095F]\u094D)*[\u0915-\u0939\u0958-\u095F]\u093C?)"
)

# 2. Reph ('Z' -> 'र्') appears AFTER the consonant + optional matra/anusvara/chandrabindu.
#    Wait: in _ORDERED_PAIRS, 'Z' -> 'र्' or keep 'Z' as marker?
#    Keeping 'Z' as a marker (by mapping 'Z' -> '\uE001' temporarily, and 'ZZ' -> 'Z') avoids
#    moving genuine 'र्' (from 'j'+'्')!
_UNIQUE_MAP["Z"] = "\uE001"
_UNIQUE_MAP[")Z"] = "र्द्ध"
_UNIQUE_MAP["Q+Z"] = "फ़\uE001"
_UNIQUE_MAP["nzZ"] = "द्र\uE001"
_UNIQUE_MAP["&"] = "\uE002"

_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_UNIQUE_MAP.keys(), key=len, reverse=True))
)

# Vowel/matra cluster after a consonant that precedes reph (\uE001)
_REPH_RE = re.compile(
    r"((?:[\u0915-\u0939\u0958-\u095F]\u094D)*[\u0915-\u0939\u0958-\u095F]\u093C?"
    r"[\u093E-\u094C\u0962\u0963\u0901-\u0903]*)\uE001"
)

# Post-mapping cleanup for split vowel combinations (e.g. 'ा' + 'े' -> 'ो')
_VOWEL_CLEANUPS: list[tuple[str, str]] = [
    ("अाे", "ओ"),
    ("अाै", "औ"),
    ("अा", "आ"),
    ("आे", "ओ"),
    ("आै", "औ"),
    ("आॅ", "ऑ"),
    ("एे", "ऐ"),
    ("एॅ", "ऍ"),
    ("ाे", "ो"),
    ("ाै", "ौ"),
    ("ाॅ", "ॉ"),
    ("िं", "िं"),
    ("्ा", ""),  # halant followed by aa-matra cancels halant (e.g. 'श्' + 'ा' -> 'श')
    ("्\u093E", ""),
]


def krutidev_to_unicode(text: str) -> str:
    """Convert a Kruti Dev 010 encoded string into Unicode Devanagari."""
    if not text:
        return text

    # Step 1: Map Kruti Dev tokens to Unicode characters (keeping 'f' and '\uE001' markers)
    mapped = _PATTERN.sub(lambda m: _UNIQUE_MAP[m.group(0)], text)

    # Step 2: Collapse halant + aa-matra ('्' + 'ा' -> full consonant) BEFORE reordering
    mapped = mapped.replace("्ा", "")

    # Step 3: Fix split vowel combinations
    for bad, good in _VOWEL_CLEANUPS:
        mapped = mapped.replace(bad, good)

    # Step 4: Move 'f' (chhoti i) after the following consonant/conjunct -> 'ि'
    mapped = _CHHOTI_I_RE.sub(r"\1ि", mapped)
    # Any leftover 'f' before space/punctuation -> 'ि'
    mapped = mapped.replace("f", "ि")

    # Step 5: Move reph ('\uE001') before the preceding consonant + matra -> 'र्'
    mapped = _REPH_RE.sub(r"र्\1", mapped)
    mapped = mapped.replace("\uE001", "र्")

    # Step 6: Fix anusvara before matra if 'a' appeared between consonant and matra
    mapped = re.sub(r"([\u0901-\u0903])([\u093E-\u094C])", r"\2\1", mapped)

    # Step 7: Convert Kruti Dev zero-prefixed decimals (e.g. 0-01 -> 0.01), then restore '&' hyphens
    mapped = re.sub(r"(?<!\d)(0)-(?=\d{2,4}\b)", r"\1.", mapped)
    mapped = mapped.replace("\uE002", "-")

    return mapped
