"""
Text processing and regex utilities shared across Term_Mapping pipelines.
Promoted according to CODING_STANDARDS.md (Search, Reuse & Promote).
"""

import re

PUNCT_STRIP = "'\"“”‘’`.,;:()[]{}<>!? \t\n"

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
ASCII_ALPHA_RE = re.compile(r"[a-zA-Z]")
SYMBOL_ONLY_RE = re.compile(r"^[\d\s.,~%+-]+$")
ACRONYM_TOKEN_RE = re.compile(r"\b[A-Za-z0-9-]+\b", re.ASCII)


def clean_phrase(text: str) -> str:
    """Strips leading and trailing quotes, apostrophes, and punctuation marks."""
    return text.strip(PUNCT_STRIP)


def has_hangul(text: str) -> bool:
    """Checks whether text contains at least one Korean Hangul syllable."""
    return bool(HANGUL_RE.search(text))


def has_ascii_alpha(text: str) -> bool:
    """Checks whether text contains at least one ASCII alphabet letter."""
    return bool(ASCII_ALPHA_RE.search(text))


def is_symbol_only(text: str) -> bool:
    """Checks whether text consists entirely of digits, whitespace, and punctuation symbols."""
    return bool(SYMBOL_ONLY_RE.match(text))
