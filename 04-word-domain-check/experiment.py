"""Controlled context-first experiments using the v1.6 local Gemma runtime."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from assessment import evaluate, in_target
from runtime.contracts import REASON_LIMIT, normalize, model_item, text, validate_decision
from runtime.gemma_backend import MODEL, Generator, append, decode_object, digest, load_checkpoint, request
from runtime.storage import logged_run, new_run, resolve_run, write_json
from runtime.taxonomy import Taxonomy
from variants import (PROFILES, apply_child_review, assignment_catalog, classify_payload, review_fields,
                      validate_child_review, validate_coded_decision, validate_major_recheck)

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        for index, line in enumerate(source, 1):
            if line.strip():
                yield index, json.loads(line)


def write_jsonl(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_items(args):
    items = []
    for line, original in read_jsonl(args.input_file):
        try:
            items.append(normalize(original, line))
        except ValueError as exc:
            raise ValueError(f"입력 {line}행: {exc}") from exc
        if args.sample_seed is None and args.max_records and len(items) == args.max_records:
            break
    if not items:
        raise ValueError("입력에 용어가 없습니다")
    if args.sample_seed is not None and 0 < args.max_records < len(items):
        items = sorted(random.Random(args.sample_seed).sample(items, args.max_records),
                       key=lambda item: item["input_line"])
    return items


def load_reference(path, items, taxonomy, kind):
    selected = {item["id"]: item for item in items}
    records = {}
    for _, row in read_jsonl(path):
        ident = row["id"]
        if ident not in selected:
            continue
        if ident in records:
            raise ValueError(f"평가 ID 중복: {ident}")
        item = selected[ident]
        if kind == "codex_reference" and "original" not in row:
            raise ValueError("Codex 평가 파일은 원래 입력을 포함한 all_results.jsonl이어야 합니다")
        if "original" in row:
            previous = normalize(row["original"], item["input_line"])
            if (previous["terms"], previous["contexts"]) != (item["terms"], item["contexts"]):
                raise ValueError(f"{ident} 평가 데이터와 현재 입력의 용어 또는 문맥이 다릅니다")
        status, domains = row["status"], row["domains"]
        if status not in {"accept", "reject", "pending"} or not isinstance(domains, list):
            raise ValueError(f"{ident}: 평가 status 또는 domains가 잘못됐습니다")
        if any(taxonomy.resolve(domain) != domain for domain in domains) or bool(domains) != (status == "accept"):
            raise ValueError(f"{ident}: 평가 도메인과 상태를 확인하세요")
        records[ident] = row
    if not records or (kind == "codex_reference" and records.keys() != selected.keys()):
        raise ValueError("평가 파일에 선택한 입력 ID가 없습니다. Codex 기준 파일은 선택 항목 전체가 필요합니다")
    return records


def criteria_for_model(taxonomy):
    rules = {major: {"scope": branch["scope"]} for major, branch in taxonomy.domains.items()}
    for major, branch in taxonomy.domains.items():
        for child in branch["subdomains"]:
            path = major + "/" + child
            rules[path] = taxonomy.criterion(path)
    return {"policy": taxonomy.data.get("policy", {}), "criteria": rules}


def validate_interpretation(raw, item):
    value, _ = decode_object(raw)
    if set(value) != {"id", "meaning", "alignment", "technicality", "reason"} or value["id"] != item["id"]:
        raise ValueError("id, meaning, alignment, technicality, reason 및 실제 입력 ID를 사용하세요")
    allowed = {"match", "mismatch", "uncertain"} if "ko" in item["terms"] else {"not_applicable"}
    if value["alignment"] not in allowed or value["technicality"] not in {"technical", "nontechnical", "uncertain"}:
        raise ValueError("입력 언어에 맞는 alignment와 technicality 값이 필요합니다")
    meaning = value["meaning"]
    if "ko" in item["terms"] and isinstance(meaning, dict) and set(meaning) == {"ko", "en"}:
        ko = text(meaning["ko"], "meaning.ko")
        en = text(meaning["en"], "meaning.en")
        value["meaning"] = f"{ko} (EN: {en})"
    text(value["meaning"], "meaning", 240, empty=True)
    text(value["reason"], "reason", REASON_LIMIT)
    return value


def classify(generator, item, taxonomy, rules, prompts, args, profile):
    trace = []
    catalog = assignment_catalog(rules) if profile == "assignment_codes" else None
    preliminary = None
    shown = None

    def ask(phase, payload, validator):
        generator.last_request = {}
        try:
            value, error = request(generator, prompts[phase], payload, validator, retries=args.response_retries)
        except RuntimeError as exc:
            if "입력" not in str(exc) or "한도 초과" not in str(exc):
                raise
            value, error = None, str(exc)
        trace.append({"phase": phase, "shown_contexts": [c["id"] for c in payload["contexts"]],
                      "decision": value, "error": error, "diagnostics": deepcopy(generator.last_request)})
        return value, error

    def decide(limit):
        nonlocal preliminary, shown
        packet = model_item(item, limit)
        scope = {**item, "contexts": packet["contexts"]}
        shown = scope
        payload = classify_payload(packet, rules, catalog)
        preliminary, error = ask("interpret", packet, lambda raw: validate_interpretation(raw, item))
        if error:
            return None, error
        payload["preliminary_analysis"] = preliminary
        if catalog is not None:
            return ask("decision", payload, lambda raw: validate_coded_decision(raw, scope, taxonomy, catalog))
        return ask("decision", payload, lambda raw: validate_decision(decode_object(raw)[0], scope, taxonomy))

    decision, error = decide(args.initial_examples)
    available = len(item["contexts"])
    expanded_limit = min(args.max_examples or available, available)
    retry_reason = decision and decision["status"] == "pending"
    if retry_reason and expanded_limit > args.initial_examples:
        decision, error = decide(expanded_limit)
    if (profile == "major_recheck" and decision and decision["status"] == "reject"
            and preliminary["alignment"] != "mismatch" and expanded_limit > len(shown["contexts"])):
        packet = model_item(item, expanded_limit)
        scope = {**item, "contexts": packet["contexts"]}
        majors = {path: criteria for path, criteria in rules["criteria"].items() if "/" not in path}
        additional_ids = [c["id"] for c in scope["contexts"][len(shown["contexts"]):]]
        recheck, _ = ask("major_recheck", {**packet, "major_criteria": majors,
                                          "additional_context_ids": additional_ids},
                         lambda raw: validate_major_recheck(raw, scope, taxonomy, additional_ids))
        if recheck and recheck["status"] == "accept":
            decision = recheck
    if profile == "child_review" and decision and decision["status"] == "accept":
        fields = review_fields(decision, rules)
        if fields:
            packet = model_item(item, len(shown["contexts"]))
            reviewed, error = ask("child_review", {**packet, "review_fields": fields},
                                  lambda raw: validate_child_review(raw, shown, fields))
            decision = apply_child_review(decision, reviewed, shown, taxonomy) if reviewed else None
    return {"id": item["id"], "decision": decision, "error": error, "trace": trace}


def export(path, items, outcomes, target, benchmarks, revision):
    rows, metrics, phases = [], Counter(), {}
    for item in items:
        ident = item["id"]
        outcome = outcomes.get(ident)
        decision = outcome.get("decision") if outcome else None
        if decision is None:
            decision = {"status": "error" if outcome else "unprocessed", "meaning": "",
                        "reason": outcome["error"] if outcome else "미분류", "assignments": [],
                        "evidence": [], "proposals": [], "domains": [], "major_domains": []}
        row = {key: item[key] for key in ("id", "input_line", "terms", "original")}
        row.update(decision, review_basis="context_and_model", externally_verified=False)
        row["domain_category"] = (1 if in_target(row["domains"], target) else
                                  2 if target["major"] in row["major_domains"] else 3
                                  ) if row["status"] == "accept" else None
        rows.append(row)
        for stage in (outcome or {}).get("trace", []):
            diagnostics = stage["diagnostics"]
            metrics["logical_requests"] += 1
            metrics["attempts"] += diagnostics.get("attempt_count", 0)
            metrics["generated_tokens"] += sum(g.get("generated_tokens", 0) for g in diagnostics.get("generations", []))
            phase = phases.setdefault(stage["phase"], Counter())
            phase["requests"] += 1
            phase["errors"] += bool(stage["error"])
            phase["attempts"] += diagnostics.get("attempt_count", 0)
    status_counts = Counter(row["status"] for row in rows)
    write_jsonl(path / "all_results.jsonl", rows)
    for name, status in [("accepted", "accept"), ("rejected", "reject"), ("pending", "pending"),
                         ("errors", "error"), ("unprocessed", "unprocessed")]:
        write_jsonl(path / f"{name}.jsonl", (row for row in rows if row["status"] == status))
    for category, name in [(1, "target_domain"), (2, "major_domain"), (3, "other_domain")]:
        write_jsonl(path / f"{name}.jsonl", (row for row in rows if row["domain_category"] == category))
    write_jsonl(path / "proposals.jsonl", ({"id": row["id"], **p} for row in rows for p in row["proposals"]))
    indexed = {row["id"]: row for row in rows}
    for kind, reference in benchmarks.items():
        scores, differences = evaluate(items, indexed, reference, target, kind)
        write_json(path / f"{kind}_summary.json", scores)
        write_jsonl(path / f"{kind}_differences.jsonl", differences)
    if status_counts["unprocessed"]:
        status = "incomplete"
    else:
        status = "complete_with_errors" if status_counts["error"] else "complete"
    summary = {"status": status, "target": target, "records": len(items),
               "counts": {key: status_counts[key] for key in ("accept", "reject", "pending", "error", "unprocessed")},
               "target_accepted": sum(row["domain_category"] == 1 for row in rows),
               "model_commit": revision, "latest_attempt_metrics": dict(metrics), "phases": phases}
    write_json(path / "summary.json", summary)
    return summary


def prompts_for(root, version):
    profile = PROFILES[version]
    phases = ["interpret", "decision"]
    if profile in {"child_review", "major_recheck"}:
        phases.append(profile)
    return {phase: (root / "prompts" / f"{phase}.md").read_text(encoding="utf-8") for phase in phases}


def execute(args, root, version, factory=Generator):
    if (args.initial_examples < 1 or min(args.max_records, args.max_examples, args.response_retries) < 0 or
            (args.max_examples and args.max_examples < args.initial_examples) or
            min(args.max_input_tokens, args.max_new_tokens, args.max_retry_tokens) < 1 or
            args.max_retry_tokens < args.max_new_tokens):
        raise ValueError("개수·토큰 설정을 확인하세요. max_examples=0은 전체 예문입니다")
    items = load_items(args)
    taxonomy = Taxonomy(args.taxonomy_file)
    target = taxonomy.target(args.domain)
    rules, prompts = criteria_for_model(taxonomy), prompts_for(root, version)
    benchmarks = {kind: load_reference(file, items, taxonomy, kind)
                  for kind, file in (("codex_reference", args.reference_file), ("gold", args.gold_file)) if file}
    option_names = ("model", "revision", "load_in_4bit", "max_input_tokens", "max_new_tokens", "max_retry_tokens",
                    "initial_examples", "max_examples", "response_retries")
    profile = PROFILES[version]
    settings = {"version": version, "profile": profile, "inference": {key: getattr(args, key) for key in option_names},
                "input_sha256": file_hash(args.input_file), "selected_ids": [item["id"] for item in items],
                "taxonomy_sha256": file_hash(args.taxonomy_file), "target": target, "prompt_sha256": digest(prompts),
                "code_sha256": {f.name: file_hash(f) for f in (Path(__file__), ROOT / "assessment.py", ROOT / "variants.py",
                                RUNTIME / "gemma_backend.py", RUNTIME / "taxonomy.py", RUNTIME / "contracts.py",
                                RUNTIME / "storage.py")}}
    if args.dry_run:
        preview = {"version": version, "records": len(items), "target": target,
                   "benchmarks": {key: len(value) for key, value in benchmarks.items()}, "model_loaded": False,
                   "profile": profile, "phases": list(prompts)}
        if args.show_prompt:
            first = model_item(items[0], args.initial_examples)
            preview["prompts"] = prompts
            catalog = assignment_catalog(rules) if profile == "assignment_codes" else None
            preview["decision_input"] = classify_payload(first, rules, catalog)
            preview["interpretation_input"] = first
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    path = resolve_run(args.resume) if args.resume else new_run(Path(os.environ.get("PP_STAGE_RESULT_ROOT", root / "result")))
    checkpoint = path / "details/checkpoint.jsonl"
    signature = digest(settings)
    if args.resume and not checkpoint.is_file():
        raise ValueError("재개할 체크포인트가 없습니다")
    with logged_run(path, "word-domain-check-v" + version, resume=bool(args.resume)) as run:
        if args.resume:
            _, entries = load_checkpoint(checkpoint, signature)
        else:
            append(checkpoint, {"signature": signature, "type": "header"})
            entries = []
        outcomes, revision, generator = {}, None, None
        for entry in entries:
            if entry["type"] == "model":
                revision = entry["revision"]
            if entry["type"] == "outcome":
                outcomes[entry["id"]] = entry
        write_json(path / "manifest.json", {**settings, "input": str(args.input_file.resolve()),
                   "benchmarks": {key: len(value) for key, value in benchmarks.items()},
                   "allow_record_errors": args.allow_record_errors,
                   "review_basis": "context_and_model", "externally_verified": False})
        write_json(path / "taxonomy.json", taxonomy.data)
        write_json(path / "details/prompts.json", prompts)
        began = time.monotonic()
        try:
            for index, item in enumerate(items, 1):
                previous = outcomes.get(item["id"], {})
                if previous.get("decision") is not None:
                    continue
                if generator is None:
                    generator = factory(args, revision or args.revision)
                    if revision is not None and generator.revision != revision:
                        raise ValueError("재개 모델 commit이 다릅니다")
                    revision = generator.revision
                    append(checkpoint, {"type": "model", "revision": revision})
                outcome = classify(generator, item, taxonomy, rules, prompts, args, profile)
                append(checkpoint, {"type": "outcome", **outcome})
                outcomes[item["id"]] = outcome
                if index % 25 == 0 or index == len(items):
                    print(f"{index}/{len(items)} · {item['id']} · {time.monotonic() - began:.1f}초", flush=True)
        finally:
            summary = export(path, items, outcomes, target, benchmarks, revision)
            run["status"] = summary["status"]
        errors = summary["counts"]["error"]
        if errors and args.allow_record_errors and summary["target_accepted"]:
            print(f"처리 오류 {errors:,}건은 errors.jsonl에 기록했습니다. "
                  f"검증을 통과한 목표 분야 용어 {summary['target_accepted']:,}건으로 계속합니다.", flush=True)
            return 0
        return int(errors > 0)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", "--input", type=Path, required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--taxonomy-file", type=Path, default=ROOT / "config/domains.json")
    p.add_argument("--reference-file", type=Path, help="Codex 비교 기준 all_results.jsonl")
    p.add_argument("--gold-file", type=Path, help="사람이 검수한 id/status/domains JSONL")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--sample-seed", type=int, help="전체 입력에서 선택할 때 재현 가능한 표본 번호")
    p.add_argument("--initial-examples", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=3)
    p.add_argument("--response-retries", type=int, default=1)
    p.add_argument("--allow-record-errors", action="store_true",
                   help="목표 분야에 검증된 용어가 있으면 개별 판정 오류를 기록하고 종료 코드 0 반환")
    p.add_argument("--resume", type=Path)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-retry-tokens", type=int, default=1536)
    p.add_argument("--profile-timing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show-prompt", action="store_true", help="dry-run에서 첫 요청 표시")
    p.set_defaults(enable_thinking=False, max_thinking_tokens=0)
    return p


def main(root, version):
    args = parser().parse_args()
    try:
        return execute(args, root, version)
    except KeyboardInterrupt:
        print("중단됨. 같은 설정과 --resume 실행폴더로 재개할 수 있습니다.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
