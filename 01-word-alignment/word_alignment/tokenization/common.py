"""Small backend registry; a backend returns original-text token dictionaries."""
import re

BACKENDS = {}


def register(language, name, packages=()):
    def decorate(function):
        BACKENDS[language, name] = {"function": function, "packages": packages}
        return function
    return decorate


def original_tokens(text, spans):
    """Merge overlapping surface spans and split whitespace, as the old C did."""
    merged = []
    for start, end, tags in sorted(spans, key=lambda x: (x[0], x[1])):
        if start == end:
            continue
        if not 0 <= start < end <= len(text):
            raise ValueError("토큰 원문 위치가 범위를 벗어났습니다")
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
            merged[-1][2].extend(tags)
        else:
            merged.append([start, end, list(tags)])
    result = []
    for start, end, tags in merged:
        for match in re.finditer(r"\S+", text[start:end]):
            a, b = start + match.start(), start + match.end()
            result.append({"text": text[a:b], "start": a, "end": b, "pos": tags[:]})
    return result


def validate_tokens(text, tokens):
    covered, previous = set(), 0
    for token in tokens:
        a, b = token["start"], token["end"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError("잘못된 토큰 원문 범위")
        if a < previous or token["text"] != text[a:b] or not token["text"].strip():
            raise ValueError("토큰이 겹치거나 원문과 일치하지 않습니다")
        previous = b
        covered.update(range(a, b))
    if not {i for i, c in enumerate(text) if not c.isspace()} <= covered:
        raise ValueError("토큰화 과정에서 원문 문자가 누락되었습니다")


def subwords(suite, text):
    tokenizer = suite.bge_tokenizer()
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    return original_tokens(text, [(a, b, ["SUBWORD"]) for a, b in encoded["offset_mapping"]])
