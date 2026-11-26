"""Tokenization ablations preserving original Unicode character spans."""
import re



def load_lexicon(path):
    if path is None:
        return set()
    with path.open(encoding="utf-8-sig") as reader:
        return {line.strip() for line in reader if line.strip() and not line.lstrip().startswith("#")}




def merge_compounds(sentence, tokens, lexicon=(), max_lexicon_length=None):
    """Join contiguous nominal pieces inside an eojeol; lexical exceptions longest first.

    Whitespace is crossed only by an explicit lexicon entry. Its model spelling
    uses U+2581; the output text and offsets always refer to the untouched input.
    """
    nominal = {"NNG", "NNP"}
    longest = max_lexicon_length if max_lexicon_length is not None else max(map(len, lexicon), default=0)
    result, index = [], 0
    while index < len(tokens):
        end = index + 1
        first = tokens[index]
        if set(first["pos"]) and set(first["pos"]) <= nominal:
            while end < len(tokens):
                previous, current = tokens[end - 1], tokens[end]
                if previous["end"] != current["start"] or not current["pos"]:
                    break
                if not set(current["pos"]) <= nominal | {"XSN"}:
                    break
                end += 1
        # Exact lexicon entries must start/end on existing tokenizer boundaries.
        if lexicon:
            for candidate_end in range(index + 1, len(tokens) + 1):
                if tokens[candidate_end - 1]["end"] - first["start"] > longest:
                    break
                surface = sentence[first["start"]:tokens[candidate_end - 1]["end"]]
                if surface in lexicon:
                    end = max(end, candidate_end)
        selected = tokens[index:end]
        start, stop = first["start"], selected[-1]["end"]
        surface = sentence[start:stop]
        result.append({"text": surface, "start": start, "end": stop,
                       "pos": [p for token in selected for p in token["pos"]],
                       "model_text": re.sub(r"\s+", "▁", surface),
                       "component_token_ids": list(range(index, end))})
        index = end
    return result
