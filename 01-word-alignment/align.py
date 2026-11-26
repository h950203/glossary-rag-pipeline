#!/usr/bin/env python3
"""v1.4: align bilingual input, then let Gemma select term-pair changes."""
from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import engine
from word_alignment.lab_runner import source_plan
from word_alignment.candidate_review import export_bundle
from word_alignment.gemma_selection import add_options, settings
from result_layout import PROJECT, latest_run, logged_run, new_run, promote, resolve_run, write_json

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(os.environ.get("PP_STAGE_RESULT_ROOT", ROOT / "result"))
PLAN = "details/gemma-plan.json"


def selection_sources(path, meta, methods):
    sources = {}
    for method in methods:
        info = meta["public_files"].get(method, {})
        filename = info.get("review_candidates.jsonl")
        if not filename or Path(filename).name != filename or not (path / filename).is_file():
            raise ValueError(f"{method}: 완료된 정렬 후보 파일이 필요합니다")
        sources[method] = str(path / filename)
    return sources


def gemma(path, meta):
    interpreter = os.environ.get("ALIGNMENT_GEMMA_PYTHON", sys.executable)
    command = [interpreter, "-m", "word_alignment.gemma_selection", str(path)]
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            raise
    if status:
        if status in (130, -2):
            raise KeyboardInterrupt
        raise RuntimeError(f"Gemma 선별이 중단됐습니다. run.sh select --resume {path}")
    summary = json.loads((path / "details/gemma-summary.json").read_text())
    for method, row in summary["methods"].items():
        meta.setdefault("public_files", {}).setdefault(method, {})["term-pairs.jsonl"] = row["term_pairs"]
    meta["gemma_selection"] = "details/gemma-summary.json"


def select_only(argv):
    parser = ArgumentParser(prog="run.sh select", description="완료된 v1.3 정렬 결과를 Gemma로 선별")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="완료된 v1.3 실행 폴더")
    source.add_argument("--resume", type=Path, help="v1.4 정렬·선별 실행 폴더에서 Gemma 단계만 재개")
    parser.add_argument("--method", help="선별할 v1.3 정렬 방식. 기본값은 그 실행의 기본 방식")
    parser.add_argument("--select-sentences", type=int, default=0, help="처음 N문장만 선별. 0이면 전체")
    parser.add_argument("--local-files-only", action="store_true")
    add_options(parser)
    args = parser.parse_args(argv)
    if args.resume:
        path = resolve_run(args.resume)
        if not (path / PLAN).is_file():
            raise ValueError("재개할 Gemma 선별 계획이 없습니다. 원래 run 명령에 --resume을 사용하세요")
        if any(getattr(args, option) != parser.get_default(option) for option in vars(args) if option != "resume"):
            raise ValueError("--resume에는 실행 폴더만 지정하세요. 저장된 선별 설정을 사용합니다")
        with logged_run(path, PROJECT, resume=True) as meta:
            gemma(path, meta)
        return 0

    config = settings(args)
    if args.select_sentences < 0:
        raise ValueError("--select-sentences는 0 이상이어야 합니다")
    original = args.run_dir.expanduser().resolve()
    if original.name == "details":
        original = original.parent
    existing = json.loads((original / "details/run.json").read_text())
    if existing.get("project") != "word-alignment-v1.3" or existing.get("status") != "complete" or existing.get("tokenize_only"):
        raise ValueError("완료된 v1.3 정렬 실행 폴더를 지정하세요")
    method = args.method or existing["primary_method"]
    sources = selection_sources(original, existing, [method])
    path = new_run(RESULT_ROOT)
    with logged_run(path, PROJECT) as meta:
        meta.update(command="select", source_run=str(original), primary_method=method, methods=[method], tokenize_only=False)
        write_json(path / PLAN, {"settings": config, "sources": sources, "primary": method, "limit": args.select_sentences})
        gemma(path, meta)
    return 0


def publish(path, methods, primary):
    details = path / "details"
    manifest_path = details / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for method in methods:
        prefix = "" if method == primary else method + "-"
        folder = details / "methods" / method
        files[method] = {}
        for name in ("alignments-readable.txt",):
            target = path / (prefix + name)
            promote(folder / name, target)
            files[method][name] = target.name
        files[method].update(export_bundle(path, details / manifest["stages"][method]["path"], prefix))
    write_json(manifest_path, manifest)
    return files


def execute(args):
    config = settings(args) if args.command == "run" else None
    args.resume = bool(args.resume_run)
    if args.resume:
        path = latest_run(RESULT_ROOT) if args.resume_run == "latest" else resolve_run(args.resume_run)
    elif args.dry_run:
        path = RESULT_ROOT / datetime.now().strftime("%y%m%d_%H%M%S")
    else:
        # Check input consistency before reserving a run folder.
        source_plan(args)
        path = new_run(RESULT_ROOT)
    args.output_dir = path / "details"
    args.managed_layout = True
    if args.dry_run:
        return engine.run(args)
    with logged_run(path, PROJECT, args.resume) as meta:
        plan_path = path / PLAN
        previous_plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
        if meta.get("command") == "select":
            raise ValueError("v1.3 결과의 Gemma 선별은 run.sh select --resume으로 재개하세요")
        if previous_plan and (args.command != "run" or previous_plan["settings"] != config):
            raise ValueError("재개 시 기존 Gemma 설정과 동일해야 합니다")
        engine.run(args)
        if args.command == "run" and args.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        if args.command == "run":
            manifest = json.loads((args.output_dir / "manifest.json").read_text())
            completed = [method for method, item in manifest["stages"].items() if item["status"] == "complete"]
            # On resume only requested methods are re-exported; preserve prior views.
            primary = meta.setdefault("primary_method", completed[0])
            generated = publish(path, args.methods, primary)
            meta.setdefault("public_files", {}).update(generated)
            meta["methods"] = completed
            meta["tokenize_only"] = False
            meta["command"] = args.command
            sources = selection_sources(path, meta, completed)
            write_json(plan_path, {"settings": config, "sources": sources, "primary": primary, "limit": 0})
            gemma(path, meta)
        else:
            meta["tokenize_only"] = True
        meta["command"] = args.command
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv and argv[0] == "select":
            return select_only(argv[1:])
        parser = engine.parser(managed=True, selection_options=add_options)
        parser.epilog = "완료된 v1.3 결과 재사용 또는 Gemma 단계 재개: run.sh select --help"
        args = parser.parse_args(argv)
        engine.validate(args)
        if args.command in {"methods", "list-methods"}:
            return engine.main(argv)
        if args.command in {"run", "tokenize"}:
            return execute(args)
        if args.command == "compare":
            path = resolve_run(args.lab_dir) if args.lab_dir else latest_run(RESULT_ROOT, complete=True)
            args.lab_dir = path / "details"
            if args.output is None:
                args.output = args.lab_dir / ("comparison-export." + args.format)
            with logged_run(path, PROJECT, resume=True):
                engine.export_comparisons(args)
            return 0
        if args.input:
            path = new_run(RESULT_ROOT)
            args.output_dir = path / "details"
        else:
            path = resolve_run(args.result_dir) if args.result_dir else latest_run(RESULT_ROOT, complete=True)
            args.result_dir = path / "details"
        with logged_run(path, PROJECT, resume=not bool(args.input)) as meta:
            from word_alignment.readable import export_cli
            export_cli(args)
            if args.input:
                promote(args.output_dir / "alignments-readable.txt", path / "alignments-readable.txt")
            else:
                from word_alignment.term_export import export_terms
                from word_alignment.lab_compare import select_stages
                manifest = json.loads((args.result_dir / "manifest.json").read_text())
                methods = select_stages(manifest, args.methods)
                export_terms(args.result_dir, manifest, methods)
                write_json(args.result_dir / "manifest.json", manifest)
                primary = meta.setdefault("primary_method", methods[0])
                for method, files in publish(path, methods, primary).items():
                    meta.setdefault("public_files", {}).setdefault(method, {}).update(files)
        return 0
    except KeyboardInterrupt:
        print("중단됨. 같은 명령에 --resume을 추가해 재개하세요.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, sqlite3.Error) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
