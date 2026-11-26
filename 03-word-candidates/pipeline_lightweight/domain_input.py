"""Export locally observed, reviewed candidates for word-domain-check."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def domain_pairs(rich_path, polysemy_path):
    pairs = {}

    def add(record):
        key = (record["surface_a"], record["surface_b"])
        pair = pairs.setdefault(key, {**record, "sample_sources": []})
        # Both anchor directions can describe the same occurrence.
        pair["frequency"] = max(pair["frequency"], record["frequency"])
        pair["sample_sources"] = list(dict.fromkeys(pair["sample_sources"] + record["sample_sources"]))
        return pair

    for record in read_rows(rich_path):
        add(record)
    for cluster in read_rows(polysemy_path):
        anchor = cluster["anchor_term"]
        language = cluster["anchor_lang"]
        if language not in {"ko", "en"}:
            raise ValueError(f"지원하지 않는 다의어 앵커 언어: {language}")
        for translation in cluster["translations"]:
            term = translation["term"]
            ko, en = (anchor, term) if language == "ko" else (term, anchor)
            canonical = cluster.get("canonical_term") or anchor
            pair = add({
                "surface_a": ko, "surface_b": en,
                "canonical_a": canonical if language == "ko" else ko,
                "canonical_b": canonical if language == "en" else en,
                "frequency": translation["frequency"], "sample_sources": translation["sample_sources"],
                "category": "polysemy_candidate",
            })
            pair["canonical_a" if language == "ko" else "canonical_b"] = canonical
            pair.setdefault("polysemy_anchors", []).append({
                "anchor_term": anchor, "anchor_lang": language,
                "total_frequency": cluster["total_frequency"], "translation": translation,
            })
    return list(pairs.values())


def export_domain_input(rich_path: Path, polysemy_path: Path, bundles_path: Path, output: Path) -> int:
    pairs = domain_pairs(rich_path, polysemy_path)
    needed = {source for pair in pairs for source in pair["sample_sources"]}
    sentences = {}
    for bundle in read_rows(bundles_path):
        for candidate in bundle["candidates"]:
            source = f"{bundle['a_sent_id']}:{candidate['b_sent_id']}"
            if source not in needed:
                continue
            example = {"source_pair": source, "ko_text": bundle["a_text"], "en_text": candidate["b_text"]}
            if not example["ko_text"].strip() or not example["en_text"].strip():
                raise ValueError(f"한영 예문이 비어 있습니다: {source}")
            if sentences.setdefault(source, example) != example:
                raise ValueError(f"같은 출처 ID에 서로 다른 한영 예문이 있습니다: {source}")

    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        with staged.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                sources = pair["sample_sources"]
                if not sources or any(source not in sentences for source in sources):
                    raise ValueError(f"한영 예문을 연결할 수 없습니다: {pair['surface_a']} / {pair['surface_b']} ({sources})")
                pair["sample_sentences"] = [sentences[source] for source in sources]
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        staged.replace(output)
    return len(pairs)
