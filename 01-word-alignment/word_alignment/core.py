import json
import re
from itertools import zip_longest


def parallel_lines(en_path, ko_path):
    with en_path.open(encoding="utf-8-sig") as en, ko_path.open(encoding="utf-8-sig") as ko:
        for index, (left, right) in enumerate(zip_longest(en, ko), 1):
            if left is None or right is None:
                raise ValueError(f"입력 줄 수가 다릅니다: {index}번째 줄. 파일을 자동으로 자르지 않습니다.")
            yield index, left.rstrip("\r\n"), right.rstrip("\r\n")


class Tokenizer:
    def __init__(self):
        try:
            from kiwipiepy import Kiwi
            import spacy
        except ImportError as exc:
            raise ValueError("lib/requirements-align.txt의 Kiwi와 spaCy를 설치하세요.") from exc
        self.kiwi = Kiwi(num_workers=2)
        # Tokenization only: POS filtering and terminology selection belong downstream.
        self.english = spacy.blank("en")

    def __call__(self, sentence, language):
        if language == "en":
            spans = [(t.idx, t.idx + len(t.text), None) for t in self.english(sentence) if not t.is_space]
        else:
            spans = [(t.start, t.start + t.len, t.tag) for t in self.kiwi.tokenize(sentence)]
        # Kiwi can assign overlapping original spans to contracted morphemes.
        # Merge these rather than inventing offsets for a normalized surface.
        merged = []
        for start, end, pos in sorted(spans):
            if merged and start < merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
                if pos:
                    merged[-1][2].append(pos)
            else:
                merged.append([start, end, [pos] if pos else []])
        tokens = []
        for start, end, tags in merged:
            value = sentence[start:end]
            if not value.strip():
                continue
            # Kiwi may return a multiword proper noun as one token. All backends
            # need whitespace-free units; split only its original surface while
            # retaining exact character offsets. Phrase extraction can reunite it.
            for match in re.finditer(r"\S+", value):
                part = match.group()
                if "|||" in part:
                    raise ValueError(f"백엔드 입력으로 표현할 수 없는 토큰: {part!r}")
                tokens.append({"text": part, "start": start + match.start(),
                               "end": start + match.end(), "pos": list(tags)})
        return tokens


def prepare(en_path, ko_path, spool, max_sentences=0, tokenizer=None, dry_run=False):
    total = selected = empty = 0
    writer = None if dry_run else spool.open("w", encoding="utf-8")
    try:
        for total, en, ko in parallel_lines(en_path, ko_path):
            if max_sentences and total > max_sentences:
                continue  # Still validate the complete files have equal line counts.
            selected += 1
            row = {"line_id": total, "en": en, "ko": ko}
            if not en.strip() or not ko.strip():
                empty += 1
                row.update(status="empty_text", tokens={"en": [], "ko": []})
            elif not dry_run:
                if tokenizer is None:
                    tokenizer = Tokenizer()
                row["tokens"] = {lang: tokenizer(row[lang], lang) for lang in ("en", "ko")}
                row["status"] = "ready" if all(row["tokens"].values()) else "empty_tokens"
            if writer:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if writer:
            writer.close()
    return {"total_lines": total, "selected_lines": selected, "empty_lines": empty}


def records(spool):
    with spool.open(encoding="utf-8") as reader:
        for line in reader:
            yield json.loads(line)


def validate_links(links, row):
    result = set()
    for i, j in links:
        if not (0 <= i < len(row["tokens"]["en"]) and 0 <= j < len(row["tokens"]["ko"])):
            raise ValueError(f"line {row['line_id']}: 잘못된 정렬 인덱스 {i}-{j}")
        result.add((int(i), int(j)))
    return sorted(result)


def candidate(row, a, b, c, d, kind):
    result = {"kind": kind, "score": None}
    for lang, start, end in (("en", a, b), ("ko", c, d)):
        tokens = row["tokens"][lang]
        left, right = tokens[start]["start"], tokens[end - 1]["end"]
        result[lang] = row[lang][left:right]
        result[lang + "_span"] = [left, right]
        result[lang + "_token_ids"] = list(range(start, end))
    return result


def extract_candidates(row, links, max_phrase_tokens):
    """Conservative phrase extraction: aligned boundaries, no external links.

    Retain every token link; add consistent contiguous phrases, without padding
    boundaries with unaligned tokens. No POS/term-frequency filter or top-K cap.
    """
    pairs = [candidate(row, i, i + 1, j, j + 1, "word") for i, j in links]
    neighbors = {}
    for i, j in links:
        neighbors.setdefault(i, set()).add(j)
    for a in sorted(neighbors):
        targets = set()
        for end in range(a, min(len(row["tokens"]["en"]), a + max_phrase_tokens)):
            targets.update(neighbors.get(end, ()))
            if end not in neighbors:
                continue
            c, d = min(targets), max(targets) + 1
            if d - c > max_phrase_tokens or (end == a and d - c == 1):
                continue
            if any(c <= j < d and not a <= i <= end for i, j in links):
                continue
            pairs.append(candidate(row, a, end + 1, c, d, "phrase"))
    return pairs


def parse_pharaoh(line):
    result = []
    for value in line.split():
        parts = value.split("-")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError(f"잘못된 Pharaoh 정렬: {value!r}")
        result.append(tuple(map(int, parts)))
    return result
