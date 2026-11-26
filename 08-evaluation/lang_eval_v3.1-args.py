#!/usr/bin/env python3
"""Evaluate one translation TXT using the COMET/LaBSE functions from v3.1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="번역 TXT 파일을 직접 COMET/LaBSE로 평가합니다.",
        epilog=(
            "기본 원문: 스크립트 폴더의 data/benchmarkGE.en. "
            "기본 정답: 원문 파일의 확장자를 target-lang으로 바꾼 경로. "
            "상대 경로는 현재 작업 폴더 기준입니다."
        ),
    )
    parser.add_argument("hypothesis", nargs="?", help="평가할 번역 TXT 경로")
    parser.add_argument("--hypothesis-file", "--input-file", "--txt", dest="hypothesis_file",
                        help="평가할 번역 TXT 경로 (위치 인자 대신 사용 가능)")
    parser.add_argument("--source-file", "--src-file", "--source", dest="source_file",
                        help="원문 TXT 경로 (기본: data/benchmarkGE.en)")
    parser.add_argument("--reference-file", "--ref-file", "--reference", dest="reference_file",
                        help="정답 번역 TXT 경로 (기본: 원문과 같은 이름의 .ko 파일)")
    parser.add_argument("--source-lang", "--src-lang", dest="source_lang", default="en")
    parser.add_argument("--target-lang", "--tgt-lang", dest="target_lang", default="ko")
    parser.add_argument("--output-log", "--output-log-file", dest="output_log",
                        default=str(SCRIPT_DIR / "log/evaluation_log_args.json"),
                        help="누적 평가 JSON 경로 (기본: log/evaluation_log_args.json)")
    parser.add_argument("--name", help="평가 이름 (기본: 번역 파일명)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--check-only", action="store_true",
                        help="파일 경로·행 수·빈 번역 수만 확인하고 종료 (모델 로딩 없음)")
    return parser


def load_aligned_text(path: Path) -> list[str]:
    # Keep blank hypothesis lines: removing them would shift later alignments.
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]


def resolve_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    if bool(args.hypothesis) == bool(args.hypothesis_file):
        parser.error("번역 TXT는 위치 인자 또는 --hypothesis-file 중 하나로 입력하세요.")
    args.source_lang = args.source_lang.strip().lower()
    args.target_lang = args.target_lang.strip().lower()
    if not args.source_lang or not args.target_lang:
        parser.error("source-lang와 target-lang을 지정하세요.")
    source_path = Path(args.source_file or SCRIPT_DIR / f"data/benchmarkGE.{args.source_lang}").resolve()
    reference_path = (
        Path(args.reference_file).resolve()
        if args.reference_file else source_path.with_suffix(f".{args.target_lang}")
    )
    hypothesis_path = Path(args.hypothesis or args.hypothesis_file).resolve()
    output_log = Path(args.output_log).resolve()
    if output_log in {source_path, reference_path, hypothesis_path}:
        parser.error("output-log는 평가 입력 파일과 다른 경로여야 합니다.")
    try:
        sources = load_aligned_text(source_path)
        references = load_aligned_text(reference_path)
        candidates = load_aligned_text(hypothesis_path)
    except (OSError, UnicodeError) as error:
        parser.error(f"평가 파일을 읽을 수 없습니다: {error}")
    if len({len(sources), len(references), len(candidates)}) != 1:
        parser.error(
            f"행 수가 다릅니다: source={len(sources)}, reference={len(references)}, "
            f"translation={len(candidates)}. 같은 문장 순서의 완성된 파일을 사용하세요."
        )
    if not sources:
        parser.error("평가할 문장이 없습니다.")
    for label, lines in (("source", sources), ("reference", references)):
        empty_lines = [index + 1 for index, line in enumerate(lines) if not line]
        if empty_lines:
            parser.error(f"{label}에 빈 줄이 있습니다: {empty_lines[:10]}")
    return {
        "source_path": source_path,
        "reference_path": reference_path,
        "hypothesis_path": hypothesis_path,
        "output_log": output_log,
        "sources": sources,
        "references": references,
        "candidates": candidates,
    }


def load_engine():
    spec = importlib.util.spec_from_file_location("lang_eval_v3_1_args_engine", SCRIPT_DIR / "lang_eval_v3.1.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("lang_eval_v3.1.py를 불러올 수 없습니다.")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return engine


def evaluate(args: argparse.Namespace, inputs: dict) -> None:
    engine = load_engine()
    output_log = engine.load_output_log(inputs["output_log"])
    device = args.device
    if device == "auto":
        device = "cuda" if engine.torch.cuda.is_available() else "cpu"
    if device == "cuda" and not engine.torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행할 수 있습니다.")

    print(f"[DEVICE] {device}", flush=True)
    print(f"[COMET] 모델 로딩: {engine.COMET_MODEL_NAME}", flush=True)
    comet_model = engine.load_from_checkpoint(engine.download_model(engine.COMET_MODEL_NAME))
    comet_score = engine.calculate_comet(
        inputs["sources"], inputs["references"], inputs["candidates"], comet_model, device
    )
    # Release COMET before loading LaBSE so both models need not share VRAM.
    del comet_model
    if device == "cuda":
        engine.torch.cuda.empty_cache()

    print(f"[LaBSE] 모델 로딩: {engine.LABSE_MODEL_NAME}", flush=True)
    tokenizer = engine.AutoTokenizer.from_pretrained(engine.LABSE_MODEL_NAME)
    labse_model = engine.AutoModel.from_pretrained(engine.LABSE_MODEL_NAME).to(device)
    labse_score = engine.calculate_labse(
        inputs["candidates"], inputs["references"], tokenizer, labse_model, device
    )
    now = datetime.now(timezone.utc)
    record = {
        "id": f"eval-args-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "name": args.name or inputs["hypothesis_path"].stem,
        "script_version": "lang_eval_v3.1-args",
        "status": "completed",
        "input": {
            "file": str(inputs["source_path"]),
            "reference_file": str(inputs["reference_path"]),
        },
        "output": {"txt": str(inputs["hypothesis_path"])},
        "evaluation": {
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "status": "completed",
            "evaluated_at": now.isoformat(),
            "sentence_count": len(inputs["sources"]),
            "empty_hypotheses": sum(not line for line in inputs["candidates"]),
            "reference_file": str(inputs["reference_path"]),
            "comet_model": engine.COMET_MODEL_NAME,
            "labse_model": engine.LABSE_MODEL_NAME,
            "comet": float(comet_score),
            "labse": float(labse_score),
        },
    }
    output_log["experiments"].append(record)
    engine.rebuild_best_evaluations(output_log)
    engine.save_json(inputs["output_log"], output_log)
    print(f"[RESULT] COMET={comet_score:.6f}, LaBSE={labse_score:.6f}")
    print(f"[DONE] {inputs['output_log']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    inputs = resolve_inputs(parser, args)
    print(json.dumps({
        "source": str(inputs["source_path"]),
        "reference": str(inputs["reference_path"]),
        "translation": str(inputs["hypothesis_path"]),
        "sentence_count": len(inputs["sources"]),
        "empty_hypotheses": sum(not line for line in inputs["candidates"]),
        "output_log": str(inputs["output_log"]),
    }, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    evaluate(args, inputs)


if __name__ == "__main__":
    main()
