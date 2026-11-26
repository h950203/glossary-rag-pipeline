#!/usr/bin/env python3
"""Group analyzed nominal/verbal term pairs; no local NLP or re-alignment."""
import argparse
from collections import Counter, defaultdict
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import groups as g
from aligned_input import read_input, input_summary, build_pair_inventory, export_pair_memberships
from observed_statistics import observed_statistics, finalize_observed_quality, attach_bilingual_evidence
from quality import initialize_quality
from nlp_tokens import load_rules

ROOT = Path(__file__).resolve().parent
KINDS = ("nominal", "verbal")


def build_terms(rows, max_tokens, sample_count):
    nodes = {lang: {} for lang in g.LANGUAGES}
    for row in rows:
        for pair in row["alignment_pairs"]:
            for lang in g.LANGUAGES:
                unit = pair[lang+"_unit"]
                parts = tuple(unit["tokens"])
                key = (unit["type"], parts)
                node = nodes[lang].setdefault(key, {"rows": set(), "variants": defaultdict(set),
                                                   "units": [], "unit": unit})
                node["rows"].add(row["source_line"])
                node["variants"][pair[lang]].add(row["source_line"])
                if len(node["units"]) < sample_count:
                    node["units"].append({"source_line": row["source_line"], "line_id": row["line_id"], **unit})
    all_terms = {lang: {} for lang in g.LANGUAGES}
    for lang, local in nodes.items():
        for (kind, parts), node in sorted(local.items()):
            unit = node["unit"]
            term_id = lang + "_" + hashlib.sha256(json.dumps([kind,parts],ensure_ascii=False).encode()).hexdigest()[:20]
            canonical = unit["canonical"]
            term = {"term_id": term_id, "language": lang, "unit_type": kind,
                    "term": canonical, "normalized": unit.get("normalized", " ".join(parts)), "tokens": list(parts),
                    "token_count": len(parts), "word_count": len(canonical.split()),
                    "character_count": len(canonical.replace(" ","")),
                    "pos": unit["pos"], "candidate_flags": [], "shape_reasons": [],
                    "surface_variants": [{"surface": s, "candidate_sentence_frequency": len(lines)}
                                         for s,lines in sorted(node["variants"].items())],
                    "candidate_sentence_frequency": len(node["rows"]),
                    "sample_candidate_lines": sorted(node["rows"])[:sample_count],
                    "sample_upstream_units": node["units"], "shape_eligible": len(parts)<=max_tokens,
                    "over_token_limit": len(parts)>max_tokens, "_candidate_rows": node["rows"],
                    "occurrence_frequency": 0, "sentence_frequency": 0, "sample_occurrences": [],
                    "_left": Counter(), "_right": Counter(), "_containing": []}
            initialize_quality(term)
            all_terms[lang][(kind,parts)] = term
    return all_terms


def typed_rows(rows, kind):
    return [dict(row, alignment_pairs=[p for p in row["alignment_pairs"] if p["unit_type"]==kind]) for row in rows]


def parser(managed=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=None if managed else ROOT.parent/"word-alignment_v1.2"/"result"/"term-pairs.jsonl")
    if not managed:
        p.add_argument("--output-dir",type=Path,default=ROOT/"output")
    else:
        p.add_argument("--alignment-method",default="C_nltk_treebank",help="자동 입력 선택 시 사용할 정렬 방식")
        p.add_argument("--include-occurrences",action="store_true",help="3_complete에 원문 표기와 좌표 추가")
        p.add_argument("--candidate-form",choices=("representative","surface"),default="representative")
        p.add_argument("--require-both",action="store_true",help="3_complete에 양쪽 후보가 있는 문장만 저장")
    p.add_argument("--methods",nargs="+",choices=g.METHODS,default=["3B_evidence"])
    p.add_argument("--unit-types",nargs="+",choices=KINDS,default=list(KINDS))
    p.add_argument("--max-records",type=int,default=0)
    p.add_argument("--max-candidate-tokens",type=int,default=12)
    p.add_argument("--sample-sources",type=int,default=5)
    p.add_argument("--min-atomic-sentences",type=int,default=3)
    p.add_argument("--min-compound-sentences",type=int,default=5)
    p.add_argument("--min-verb-sentences",type=int,default=3)
    p.add_argument("--min-cvalue",type=float,default=5.)
    p.add_argument("--min-npmi",type=float,default=.1)
    p.add_argument("--min-context-types",type=int,default=2)
    p.add_argument("--min-sense-purity",type=float,default=.9)
    p.add_argument("--rules-file",type=Path,default=ROOT/"rules.json")
    if not managed:
        p.add_argument("--overwrite",action="store_true")
    else:
        p.set_defaults(overwrite=False)
    return p


def run(args, *, prepared_input=False):
    import fcntl
    import math
    started=time.monotonic()
    if args.max_records<0 or min(args.max_candidate_tokens,args.sample_sources,args.min_atomic_sentences,args.min_compound_sentences,args.min_verb_sentences)<1:
        raise ValueError("문장 제한은 0 이상, 개수 조건은 1 이상이어야 합니다")
    if not math.isfinite(args.min_cvalue) or args.min_cvalue<0 or not math.isfinite(args.min_npmi) or not -1<=args.min_npmi<=1 or not 0<args.min_sense_purity<=1 or args.min_context_types<0:
        raise ValueError("점수·비율 조건을 확인하세요")
    args.methods=list(dict.fromkeys(args.methods)); args.unit_types=list(dict.fromkeys(args.unit_types))
    args.active_rules=load_rules(args.rules_file)
    out=args.output_dir.expanduser().resolve()
    source = args.input.resolve()
    is_managed_snapshot = prepared_input and source == out / "sample_alignments.jsonl"
    if (source.is_relative_to(out) and not is_managed_snapshot) or args.rules_file.resolve().is_relative_to(out):
        raise ValueError("입력·규칙 파일을 출력 폴더 안에 둘 수 없습니다")
    out.mkdir(parents=True,exist_ok=True)
    with (out/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("같은 출력 폴더에서 실행 중입니다") from exc
        if (out/"manifest.json").exists() and not args.overwrite:
            raise ValueError("기존 결과가 있습니다. --overwrite 또는 새 출력 폴더를 사용하세요")
        rows,raw,input_hash=read_input(args.input.expanduser(),args.max_records)
        raw = {lang: defaultdict(set) for lang in g.LANGUAGES}
        for row in rows:
            row["alignment_pairs"]=[p for p in row["alignment_pairs"] if p["unit_type"] in args.unit_types]
            for pair in row["alignment_pairs"]:
                for lang in g.LANGUAGES:
                    unit=pair[lang+"_unit"]
                    raw[lang][(unit["type"],pair[lang],tuple(unit["tokens"]))].add(row["source_line"])
        if not any(r["alignment_pairs"] for r in rows):
            raise ValueError("선택한 유형의 용어쌍이 없습니다")
        counts=input_summary(rows,raw)
        manifest={"schema":"word-similar-v1.2","status":"running","started_at":datetime.now(timezone.utc).isoformat(),
                  "input":str(args.input.resolve()),"input_sha256":input_hash,"input_summary":counts,
                  "settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!="active_rules"},
                  "methods":args.methods,"unit_types":args.unit_types,"uses_llm":False,"uses_local_nlp":False,
                  "frequency_basis":"exact_observed_pairs","phase_seconds":{},
                  "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                   ("build_glossary.py","groups.py","aligned_input.py","observed_statistics.py","quality.py")}}
        g.write_json(out/"manifest.json",manifest)
        try:
            print(f"입력 {len(rows):,}문장 / {counts['pair_occurrences']:,}대응; 상위 분석 재사용; 로컬 NLP 없음",flush=True)
            phase=time.monotonic()
            all_terms=build_terms(rows,args.max_candidate_tokens,args.sample_sources)
            pairs=build_pair_inventory(rows,all_terms,args.sample_sources)
            for kind in args.unit_types:
                local_rows=typed_rows(rows,kind)
                for lang in g.LANGUAGES:
                    local={parts:t for (category,parts),t in all_terms[lang].items() if category==kind}
                    frequencies,total=observed_statistics(local,local_rows,lang,args.max_candidate_tokens,args.sample_sources,args.active_rules)
                    g.compute_scores(local,frequencies,total,frequency_scope="observed")
                    finalize_observed_quality(local,lang,args,args.active_rules)
            attach_bilingual_evidence(all_terms)
            manifest["phase_seconds"]["provided_analysis_and_statistics"]=time.monotonic()-phase
            by_id={t["term_id"]:t for local in all_terms.values() for t in local.values()}
            summaries={}
            with tempfile.TemporaryDirectory(prefix=".build-",dir=out) as temp:
                staging=Path(temp)
                g.write_jsonl(staging/"terms.jsonl",(g.public(t) for t in by_id.values()))
                g.write_jsonl(staging/"aligned-pairs.jsonl",pairs)
                g.write_jsonl(staging/"sources.jsonl",rows)
                g.write_json(staging/"rules.json",args.active_rules)
                g.write_json(staging/"input-summary.json",{**counts,"normalized_pairs":len(pairs),"uses_local_nlp":False})
                columns=("language","unit_type","term_id","term","sentence_frequency","occurrence_frequency",
                         "aligned_containment_sentence_frequency","c_value","min_npmi","context_types","quality_status")
                g.write_csv(staging/"candidate-scores.csv",columns,({k:t[k] for k in columns} for t in by_id.values()))
                g.write_csv(staging/"normalization-links.csv",("language","unit_type","term_id","canonical","surface"),
                    ({"language":t["language"],"unit_type":t["unit_type"],"term_id":t["term_id"],
                      "canonical":t["term"],"surface":v["surface"]} for t in by_id.values() for v in t["surface_variants"]))
                for method in args.methods:
                    phase=time.monotonic(); records=[]; families=[]; hierarchy=[]
                    for kind in args.unit_types:
                        local_args=copy(args)
                        if kind=="verbal": local_args.min_atomic_sentences=args.min_verb_sentences
                        for lang in g.LANGUAGES:
                            local={parts:t for (category,parts),t in all_terms[lang].items() if category==kind}
                            result=g.build_method(local,lang,method,local_args,set())
                            if kind=="verbal":
                                for record in result[0]:
                                    if record["is_core_candidate"]: record["selection_reason"]="verbal_lemma_core"
                            records.extend(result[0]); families.extend(result[1]); hierarchy.extend(result[2])
                    summary=g.export_method(staging/method,by_id,records,families,hierarchy,method)
                    summary["unit_types"]={kind:{"terms":sum(r["unit_type"]==kind for r in records),
                                                "families":sum(f["unit_type"]==kind for f in families)} for kind in args.unit_types}
                    summary["alignment_pairs"]=export_pair_memberships(staging/method,pairs,{r["term_id"]:r for r in records},by_id)
                    g.write_json(staging/method/"summary.json",summary); summaries[method]=summary
                    manifest["phase_seconds"][method+"_grouping"]=time.monotonic()-phase
                g.write_json(staging/"summary.json",{"methods":summaries,"automatic_deletions":0,"normalized_pairs":len(pairs)})
                for entry in staging.iterdir():
                    if entry.is_dir():
                        (out/entry.name).mkdir(exist_ok=True)
                        for child in entry.iterdir(): os.replace(child,out/entry.name/child.name)
                    else: os.replace(entry,out/entry.name)
            manifest["status"]="complete"
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
            raise
        finally:
            manifest["elapsed_seconds"]=time.monotonic()-started
            g.write_json(out/"manifest.json",manifest)
            print(f"종료: {manifest['status']}; {manifest['elapsed_seconds']:.2f}초; 저장: {out}",flush=True)
    return 0


if __name__=="__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except (ValueError,OSError,ImportError,KeyError,TypeError) as exc:
        print(f"오류: {exc}",file=sys.stderr); sys.exit(1)
    except KeyboardInterrupt:
        print("중단됨. 새 폴더 또는 --overwrite로 재실행하세요.",file=sys.stderr); sys.exit(130)
