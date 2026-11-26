#!/usr/bin/env python3
"""한국어·영어 토큰화와 BGE-M3 단어 정렬을 독립적으로 실행합니다."""
import argparse
from pathlib import Path
import sqlite3
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from word_alignment.lab_runner import run
from word_alignment.lab_compare import export_comparisons
from word_alignment.tokenization import load_methods

ROOT = Path(__file__).resolve().parent
SHARED_CACHE = ROOT.parent / ".cache-alignment-lab" / "models"
MODEL_CACHE = SHARED_CACHE if SHARED_CACHE.is_dir() else ROOT / ".cache" / "models"


def parser(managed=False, selection_options=None):
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "토큰화 후 독립 BGE 정렬"), ("tokenize", "토큰화만 저장; encoder 추론 없음")):
        execute = sub.add_parser(name, help=help_text)
        execute.add_argument("--en", type=Path, required=True)
        execute.add_argument("--ko", type=Path, required=True)
        if not managed:
            execute.add_argument("--output-dir", type=Path, default=ROOT / "result")
        selection = execute.add_mutually_exclusive_group()
        selection.add_argument("--methods", "--profiles", nargs="+", help="기본: C_nltk_treebank")
        selection.add_argument("--all-methods", action="store_true")
        execute.add_argument("--methods-file", type=Path, help="추가 토큰화 조합 JSON")
        if managed:
            execute.add_argument("--resume", nargs="?", const="latest", dest="resume_run",
                                 help="최근 실행 재개 또는 기존 YYMMDD_HHMMSS 폴더 지정")
            execute.set_defaults(resume=False)
        else:
            execute.add_argument("--resume", action="store_true", help="같은 입력·설정 재개; 새 방식 추가도 지원")
        execute.add_argument("--dry-run", action="store_true")
        execute.add_argument("--max-sentences", type=int, default=0)
        execute.add_argument("--compound-lexicon", type=Path)
        execute.add_argument("--bge-model", default="BAAI/bge-m3")
        execute.add_argument("--bge-revision", default="main")
        execute.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
        execute.add_argument("--local-files-only", action="store_true")
        execute.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        execute.add_argument("--fp16", action="store_true")
        execute.add_argument("--threads", type=int, default=2)
        execute.add_argument("--max-model-tokens", type=int, default=512)
        execute.add_argument("--bge-span-tokens", type=int, default=3)
        execute.add_argument("--max-term-tokens", type=int, default=12)
        execute.add_argument("--term-types", nargs="+", choices=["nominal","verbal"], default=["nominal","verbal"])
        execute.add_argument("--max-span-candidates", type=int, default=2048)
        execute.add_argument("--min-similarity", type=float, default=0.50)
        execute.add_argument("--margin", type=float, default=0.03)
        execute.add_argument("--checkpoint-every", type=int, default=25)
        execute.add_argument("--progress-every", type=int, default=100)
        execute.add_argument("--include-punctuation", action="store_true", help="읽기용 파일에도 문장부호 대응 포함; 원본은 항상 보존")
        if name == "run" and selection_options:
            selection_options(execute)

    listing = sub.add_parser("methods", aliases=["list-methods"], help="분리 방식·라이브러리·기본 비교 조합")
    listing.add_argument("--methods-file", type=Path)
    listing.add_argument("--json", action="store_true")

    compare = sub.add_parser("compare", help="동일 문장에서 방식별 공통·추가·제거 대응 비교")
    compare.add_argument("--lab-dir", type=Path, default=None if managed else ROOT / "result")
    compare.add_argument("--methods", "--stages", dest="stages", nargs="+")
    compare.add_argument("--reference")
    selection = compare.add_mutually_exclusive_group()
    selection.add_argument("--line-ids", nargs="+")
    selection.add_argument("--sample-size", type=int, default=0)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--en-text")
    compare.add_argument("--ko-text")
    compare.add_argument("--exact", action="store_true")
    compare.add_argument("--sentence-contains")
    compare.add_argument("--kind", choices=["all", "word", "phrase"], default="all")
    compare.add_argument("--mode", choices=["all", "common", "changed"], default="all")
    compare.add_argument("--format", choices=["jsonl", "json", "csv"], default="jsonl")
    compare.add_argument("--layout", choices=["sentences", "pairs"], default="sentences")
    compare.add_argument("--fields", nargs="+")
    compare.add_argument("--limit", type=int, default=0)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--overwrite", action="store_true")

    export = sub.add_parser("export", help="기존 전체 JSONL을 읽기용 파일로 변환; 추론 없음")
    source = export.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="기존 alignments.jsonl 직접 지정")
    source.add_argument("--result-dir", type=Path, default=None if managed else ROOT / "result")
    export.add_argument("--methods", nargs="+", help="result-dir에서 선택할 완료 방식; 기본 전체")
    if not managed:
        export.add_argument("--output-dir", type=Path, help="직접 입력 파일의 읽기용 저장 폴더; 기본 result/readable")
    else:
        export.set_defaults(output_dir=None)
    export.add_argument("--include-punctuation", action="store_true")
    return command


def validate(args):
    import math
    if args.command in {"methods", "list-methods"}:
        return
    if args.command in {"run", "tokenize"}:
        args.term_types = list(dict.fromkeys(args.term_types))
        if args.max_sentences < 0 or args.max_model_tokens < 3:
            raise ValueError("max-sentences는 0 이상, max-model-tokens는 3 이상이어야 합니다")
        for key in ("bge_span_tokens", "max_term_tokens", "max_span_candidates", "threads", "checkpoint_every", "progress_every"):
            if getattr(args, key) < 1:
                raise ValueError(f"--{key.replace('_', '-')}는 1 이상이어야 합니다")
        if not math.isfinite(args.min_similarity) or not -1 <= args.min_similarity <= 1:
            raise ValueError("min-similarity는 [-1,1]이어야 합니다")
        if not math.isfinite(args.margin) or not 0 <= args.margin <= 2:
            raise ValueError("margin은 [0,2]이어야 합니다")
        if args.fp16 and args.device != "cuda":
            raise ValueError("--fp16은 --device cuda에서 사용하세요")
        available = load_methods(args.methods_file)
        args.methods = list(available) if args.all_methods else list(dict.fromkeys(args.methods or ("C_nltk_treebank",)))
        if set(args.methods) - set(available):
            raise ValueError(f"알 수 없는 방식: {set(args.methods)-set(available)}; methods 명령을 확인하세요")
    elif args.command == "export":
        if args.input and args.methods:
            raise ValueError("--input과 --methods를 함께 사용할 수 없습니다")
        if not args.input and args.output_dir:
            raise ValueError("--output-dir는 --input과 함께 사용하세요")
    elif args.sample_size < 0 or args.limit < 0:
        raise ValueError("sample-size와 limit는 0 이상이어야 합니다")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate(args)
        if args.command in {"methods", "list-methods"}:
            import json
            methods = load_methods(args.methods_file)
            if args.json:
                print(json.dumps(methods, ensure_ascii=False, indent=2))
            else:
                for name, spec in methods.items():
                    print(f"{name}: en={spec['en']}, ko={spec['ko']} — {spec.get('description','')}")
            return 0
        if args.command == "export":
            from word_alignment.readable import export_cli
            result = export_cli(args)
        else:
            result = run(args) if args.command in {"run", "tokenize"} else export_comparisons(args)
        return 0 if result is None else result
    except KeyboardInterrupt:
        print("중단됨. 같은 명령에 --resume을 추가해 재개하세요.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, sqlite3.Error) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
