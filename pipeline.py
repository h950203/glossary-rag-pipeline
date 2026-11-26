#!/usr/bin/env python3
"""Run the terminology and translation pipeline with a shared result directory."""

import argparse
from datetime import datetime, timezone
import fcntl
from itertools import islice, zip_longest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, mkdtemp


ROOT = Path(__file__).resolve().parent
STAGES = (
    "01-word-alignment", "02-word-similar", "03-word-candidates", "04-word-domain-check",
    "05-glossary", "06-sentence-selection", "07-translation", "08-evaluation",
)
PLAIN_TRANSLATION = "07-no-rag-plain.ko.txt"
PLAIN_LOG = "07-no-rag-plain.ko.csv"
COT_TRANSLATIONS = ("07-no-rag.ko.txt", "07-rag-terms.ko.txt", "07-rag-examples.ko.txt")
ARTEFACTS = (
    ("01-aligned-terms.jsonl",),
    ("02-representative-candidates.jsonl", "02-group-pairs.csv", "02-representative-groups.csv"),
    ("03-domain-review-input.jsonl",),
    ("04-target-domain.jsonl", "04-major-domain.jsonl", "04-other-domain.jsonl", "04-domain-decisions.jsonl"),
    ("05-glossary.jsonl", "05-glossary-with-examples.jsonl"),
    ("06-eval-source.en", "06-eval-reference.ko"),
    (*COT_TRANSLATIONS, PLAIN_TRANSLATION),
    ("08-terms-vs-no-rag.csv", "08-examples-vs-no-rag.csv", "08-four-conditions.csv"),
)
START_INPUTS = {
    3: {"02-representative-candidates.jsonl": ("3_complete.jsonl",)},
    4: {"03-domain-review-input.jsonl": ("combined_for_domain_check.jsonl", "combined_with_polysemy.jsonl",
                                       "combined.jsonl")},
    5: {"04-target-domain.jsonl": ("target_domain.jsonl",)},
    6: {"05-glossary.jsonl": ("glossary.jsonl",),
        "05-glossary-with-examples.jsonl": ("glossary-with-examples.jsonl",)},
    7: {"06-eval-source.en": ("eval-source.en",), "06-eval-reference.ko": ("eval-reference.ko",)},
    8: {"07-no-rag.ko.txt": ("no-rag.ko.txt",), "07-rag-terms.ko.txt": ("rag-terms.ko.txt",),
        "07-rag-examples.ko.txt": ("rag-examples.ko.txt",)},
}
PATH_OPTIONS = (
    "alignment_result", "train_en", "train_ko", "train_json", "eval_en", "eval_ko", "eval_json", "filter_glossary",
    "eval_glossary", "taxonomy_file", "reference_file", "gold_file", "comet_checkpoint",
)
TEXT_OPTIONS = ("domain", "python", "eval_python", "bge_model", "gemma_model", "gemma_revision")
BOOL_OPTIONS = ("load_in_4bit", "local_files_only", "allow_record_errors")
INT_OPTIONS = ("max_sentences", "eval_limit", "batch_size")
BGE_CONFIG_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_SAFETENSORS_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"
BGE_CONFIG_FILES = (
    "config.json", "colbert_linear.pt", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
)


def resolve_bge_model(model, run_dir):
    if model != "BAAI/bge-m3":
        return model

    # Use the cached safetensors form of the same checkpoint used in the smoke test.
    # Keep the Hub snapshots intact and share the local view across stages 01 and 03.
    hub_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache_dirs = (
        ROOT.parent / ".cache-alignment-lab/models",
        Path(os.environ.get("HF_HUB_CACHE", hub_home / "hub")),
        ROOT / "01-word-alignment/.cache/models",
    )
    model_dir = run_dir / "details/model-assets/bge-m3"
    required = (*BGE_CONFIG_FILES, "model.safetensors")
    if all((model_dir / name).is_file() for name in required):
        return str(model_dir)

    for cache in cache_dirs:
        snapshots = cache / "models--BAAI--bge-m3/snapshots"
        files = {name: snapshots / BGE_CONFIG_REVISION / name for name in BGE_CONFIG_FILES}
        files["model.safetensors"] = snapshots / BGE_SAFETENSORS_REVISION / "model.safetensors"
        if not all(path.is_file() for path in files.values()):
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, path in files.items():
            link = model_dir / name
            if not link.is_file():
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(path.resolve(strict=True))
        print(f"[BGE] 캐시된 safetensors 사용: {model_dir}", flush=True)
        return str(model_dir)
    return model


def load_config(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    fields = {**dict.fromkeys((*PATH_OPTIONS, *TEXT_OPTIONS), str),
              **dict.fromkeys(BOOL_OPTIONS, bool), **dict.fromkeys(INT_OPTIONS, int)}
    if not isinstance(data, dict):
        raise ValueError("설정 파일은 JSON 객체여야 합니다.")
    for name, value in data.items():
        if name not in fields:
            raise ValueError(f"알 수 없는 설정 항목: {name}")
        if type(value) is not fields[name] or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"설정 항목 {name}에는 비어 있지 않은 {fields[name].__name__} 값이 필요합니다.")
        if name in PATH_OPTIONS or (name in {"bge_model", "gemma_model", "python", "eval_python"}
                                    and value.startswith((".", "~"))):
            data[name] = str((path.parent / Path(value).expanduser()).absolute())
    return data


def validate_inputs(config, start=1):
    for scope in ("train", "eval"):
        if (scope == "train" and start > 1) or (scope == "eval" and start > 6):
            continue
        source = config.get(scope + "_en")
        target = config.get(scope + "_ko")
        bundle = config.get(scope + "_json")
        if scope == "train" and config.get("alignment_result"):
            if source or target or bundle:
                raise ValueError("--alignment-result와 용어 구축 원문 입력을 함께 사용할 수 없습니다.")
            alignment_input(config["alignment_result"])
            continue
        if (bool(bundle) and (source or target)) or (not bundle and not (source and target)):
            raise ValueError(f"--{scope}-json 또는 --{scope}-en과 --{scope}-ko를 지정하세요.")
    for name in PATH_OPTIONS:
        if (name in {"alignment_result", "train_en", "train_ko", "train_json"} and start > 1
                or name in {"eval_en", "eval_ko", "eval_json"} and start > 6):
            continue
        value = config.get(name)
        if name != "alignment_result" and value and not Path(value).is_file():
            raise ValueError(f"{name}: 파일을 찾을 수 없습니다: {value}")


def alignment_input(value):
    path = Path(value)
    if path.is_dir():
        manifest = path / "details/run.json"
        if not manifest.is_file():
            raise ValueError(f"정렬 실행 정보가 없습니다. 완료된 term-pairs.jsonl을 직접 지정하세요: {path}")
        run = json.loads(manifest.read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get("tokenize_only"):
            raise ValueError(f"완료된 한영 정렬 실행이 필요합니다: {path}")
        filename = run.get("public_files", {}).get("C_nltk_treebank", {}).get("term-pairs.jsonl")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"C_nltk_treebank의 최종 term-pairs.jsonl을 찾을 수 없습니다: {path}")
        path /= filename
    if path.suffix != ".jsonl" or not path.is_file():
        raise ValueError(f"최종 정렬 JSONL을 찾을 수 없습니다: {path}")
    row = first_record(path)
    if (not isinstance(row, dict) or row.get("schema") != "term-pairs-v1.1"
            or row.get("method") != "C_nltk_treebank" or not isinstance(row.get("pairs"), list)):
        raise ValueError(f"C_nltk_treebank의 term-pairs-v1.1 JSONL이 필요합니다: {path}")
    return path


def first_record(path):
    with path.open(encoding="utf-8-sig") as stream:
        first = next((line for line in stream if line.strip()), None)
    return json.loads(first) if first else None


def start_input_files(start, path, config):
    if start == 1:
        if config.get("alignment_result"):
            raise ValueError("--stage 1 --input과 --alignment-result를 함께 사용할 수 없습니다.")
        for name in ("train_json", "train_en", "train_ko"):
            config[name] = None
        if path.is_file():
            config["train_json"] = str(path)
        elif path.is_dir() and (path / "train.json").is_file():
            config["train_json"] = str(path / "train.json")
        elif path.is_dir():
            config.update(train_en=str(path / "train.en"), train_ko=str(path / "train.ko"))
        else:
            raise ValueError(f"구축용 입력을 찾을 수 없습니다: {path}")
        return {}
    if start == 2:
        if path.is_dir() and (path / ARTEFACTS[0][0]).is_file():
            path /= ARTEFACTS[0][0]
        return {ARTEFACTS[0][0]: alignment_input(path)}

    required = START_INPUTS[start].copy()
    if start >= 7:
        required.update(START_INPUTS[6])
    if start == 8:
        required.update(START_INPUTS[7])
        del required[ARTEFACTS[4][1]]
        if config.get("eval_glossary"):
            del required[ARTEFACTS[4][0]]
    if not path.is_dir() and len(required) > 1:
        raise ValueError(f"{start:02d}단계 --input에는 필요한 파일을 담은 폴더를 지정하세요: {', '.join(required)}")
    files = {}
    for name, aliases in required.items():
        candidate = path
        if path.is_dir():
            candidate = next((path / item for item in (name, *aliases) if (path / item).is_file()), None)
        if candidate is None or not candidate.is_file() or not candidate.stat().st_size:
            raise ValueError(f"{start:02d}단계 입력 파일이 없거나 비었습니다: {path} ({name})")
        files[name] = candidate
    if start == 8:
        plain = next((path / name for name in (PLAIN_TRANSLATION, "no-rag-plain.ko.txt", "no-rag-plain.txt")
                      if (path / name).is_file()), None)
        if plain is not None:
            if not plain.stat().st_size:
                raise ValueError(f"08단계 Plain 번역이 비었습니다: {plain}")
            files[PLAIN_TRANSLATION] = plain
            log = plain.resolve().with_suffix(".csv")
            if log.is_file():
                files[PLAIN_LOG] = log

    for name, keys in (
        (ARTEFACTS[1][0], {"ko_sentence", "en_sentence", "ko_candidates", "en_candidates"}),
        (ARTEFACTS[2][0], {"surface_a", "surface_b", "sample_sentences"}),
        (ARTEFACTS[3][0], {"status", "terms"}),
        (ARTEFACTS[4][0], {"en", "ko"}),
        (ARTEFACTS[4][1], {"en", "ko"}),
    ):
        if name in files:
            row = first_record(files[name])
            if not isinstance(row, dict) or not keys <= row.keys():
                raise ValueError(f"{name}에 필요한 필드가 없습니다 ({', '.join(sorted(keys))}): {files[name]}")
    return files


def reuse_file(source, output):
    try:
        os.link(source.resolve(), output)
    except OSError:
        shutil.copyfile(source, output)


def register_inputs(files, run_dir):
    if not files:
        return {}
    for name in files:
        if (run_dir / name).exists() or (run_dir / name).is_symlink():
            raise FileExistsError(f"기존 파일이 있습니다. 새 --run-dir을 지정하세요: {run_dir / name}")
    destination = Path(mkdtemp(prefix="imported-", dir=run_dir / "details"))
    for name, source in files.items():
        reuse_file(source, destination / name)
    for name in files:
        (run_dir / name).symlink_to((destination / name).relative_to(run_dir))
    return {name: str(source) for name, source in files.items()}


def save_state(path, state):
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage(index, attempt, cfg, run_dir, *, reuse_translations=False):
    env = {**os.environ, "PP_STAGE_RESULT_ROOT": str(attempt), "PYTHONUNBUFFERED": "1"}
    if cfg["local_files_only"]:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def call(script, *arguments, script_stage=index):
        interpreter = (cfg.get("eval_python") if script_stage == 7 else None) or cfg["python"]
        command = [interpreter, str(ROOT / STAGES[script_stage] / script), *map(str, arguments)]
        print(f"[{STAGES[index]}] {attempt}", flush=True)
        log = attempt / "run.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                status = process.wait()
        if status:
            raise RuntimeError(f"{STAGES[index]} 종료 코드 {status}. 로그 {log}")

    def native():
        folders = [path for path in attempt.iterdir() if path.is_dir() and (path / "details/run.json").is_file()]
        if len(folders) != 1:
            raise RuntimeError(f"단계의 실행 폴더를 확인할 수 없습니다: {attempt}")
        return folders[0]

    gemma = cfg["gemma_model"]
    revision = cfg["gemma_revision"]
    offline = ["--local-files-only"] if cfg["local_files_only"] else []
    quant = ["--load-in-4bit"] if cfg["load_in_4bit"] else []
    s = ARTEFACTS

    def translate_plain(output):
        call("translate_no_rag_plain.py", "--input-file", run_dir / s[5][0], "--output-file", output.with_suffix(""),
             "--model", gemma, "--revision", revision, "--batch-size", cfg["batch_size"], *quant, *offline,
             script_stage=6)

    if index == 0:
        if cfg.get("alignment_result"):
            original = alignment_input(cfg["alignment_result"])
            output = attempt / "term-pairs.jsonl"
            reuse_file(original, output)
            print(f"[01-word-alignment] 완료된 정렬 재사용: {original}", flush=True)
            return [output]
        bge_model = resolve_bge_model(cfg["bge_model"], run_dir)
        train_en, train_ko = cfg["train_en"], cfg["train_ko"]
        if cfg.get("train_json"):
            inputs = attempt / "inputs"
            inputs.mkdir()
            train_en, train_ko = inputs / "train.en", inputs / "train.ko"
            prepare_json(cfg["train_json"], (train_en, train_ko), cfg["max_sentences"])
        call("align.py", "run", "--en", train_en, "--ko", train_ko, "--methods", "C_nltk_treebank",
             "--device", "cuda", "--fp16", "--bge-model", bge_model, "--gemma-model", gemma,
             "--gemma-revision", revision, "--max-sentences", cfg["max_sentences"], *offline,
             *(["--gemma-4bit"] if quant else []))
        return [native() / "term-pairs.jsonl"]
    if index == 1:
        limit = ["--sentences", cfg["max_sentences"]] if cfg["max_sentences"] else []
        call("experiment.py", "--input", run_dir / s[0][0], "--alignment-method", "C_nltk_treebank", *limit)
        return [native() / n for n in ("3_complete.jsonl", "group-pairs.csv", "groups-simple.csv")]
    if index == 2:
        bge_model = resolve_bge_model(cfg["bge_model"], run_dir)
        call("co-work/run_mapping.py", "--candidates_file", run_dir / s[1][0], "--output_dir", attempt,
             "--device", "cuda", "--bge_model", bge_model, "--batch_size", cfg["batch_size"])
        return [attempt / "combined_for_domain_check.jsonl"]
    if index == 3:
        options = ["--allow-record-errors"] if cfg.get("allow_record_errors", True) else []
        for key in ("taxonomy_file", "gold_file", "reference_file"):
            if cfg[key]:
                options.extend(("--" + key.replace("_", "-"), cfg[key]))
        call("review.py", "--input-file", run_dir / s[2][0], "--domain", cfg["domain"],
             "--model", gemma, "--revision", revision, *quant, *offline, *options)
        return [native() / n for n in ("target_domain.jsonl", "major_domain.jsonl",
                                      "other_domain.jsonl", "all_results.jsonl")]
    if index == 4:
        outputs = [attempt / n for n in s[index]]
        for script, output in zip(("convert_domain_glossary.py", "convert_domain_glossary_with_examples.py"), outputs):
            call(script, "--input-file", run_dir / s[3][0], "--output-file", output)
            if not output.stat().st_size:
                raise ValueError("목표 분야에 평가 가능한 용어집 항목이 없습니다. 도메인 판정과 충돌 집계를 확인하세요.")
        return outputs
    if index == 5:
        outputs = [attempt / n for n in s[index]]
        if cfg.get("eval_json"):
            prepare_json(cfg["eval_json"], outputs, cfg["eval_limit"])
        else:
            prepare_parallel(cfg["eval_en"], cfg["eval_ko"], outputs, cfg["eval_limit"])
        if cfg["filter_glossary"]:
            filtered = [attempt / f"filtered-{output.name}" for output in outputs]
            call("filter_term_sentences.py", "--glossary", cfg["filter_glossary"], "--input-en", outputs[0],
                 "--input-ko", outputs[1], "--output-en", filtered[0], "--output-ko", filtered[1])
            outputs = filtered
        return outputs
    if index == 6:
        outputs = [attempt / n for n in s[index]]
        for variant, output in enumerate(outputs[:-1]):
            if reuse_translations:
                old_output = (run_dir / output.name).resolve()
                reuse_file(old_output, output)
                old_log = old_output.with_suffix(".csv")
                if old_log.is_file():
                    reuse_file(old_log, output.with_suffix(".csv"))
                continue
            glossary = run_dir / s[4][max(0, variant - 1)]
            mode = ["--no-rag"] if variant == 0 else ["--glossary", str(glossary)]
            call("translate_rag_v1.1.py", "--input-file", run_dir / s[5][0], "--output-file", output.with_suffix(""),
                 "--fixed-prompt", "--model", gemma, "--revision", revision, "--batch-size", cfg["batch_size"],
                 "--max-examples-per-term", int(variant == 2), *quant, *offline, *mode)
        if reuse_translations:
            print(f"[07-translation] 기존 번역 세 조건 재사용: {attempt}", flush=True)
        translate_plain(outputs[-1])
        return outputs

    outputs = [attempt / n for n in s[index]]
    checkpoint = ["--comet-checkpoint", cfg["comet_checkpoint"]] if cfg["comet_checkpoint"] else []
    plain = run_dir / PLAIN_TRANSLATION
    if not plain.is_file():
        plain = attempt / PLAIN_TRANSLATION
        print("[08-evaluation] 이전 세 조건 실행에 Plain 번역을 추가합니다.", flush=True)
        translate_plain(plain)
    call("lang_eval_rag_with_plain.py", "--source-file", run_dir / s[5][0], "--reference-file", run_dir / s[5][1],
         "--no-rag-file", run_dir / s[6][0], "--rag-terms-file", run_dir / s[6][1],
         "--rag-examples-file", run_dir / s[6][2], "--plain-file", plain.resolve(),
         "--glossary", cfg["eval_glossary"] or run_dir / s[4][0], "--terms-output-file", outputs[0],
         "--examples-output-file", outputs[1], "--output-file", outputs[2],
         "--device", "cuda", "--batch-size", cfg["batch_size"], *checkpoint)
    return outputs


def prepare_parallel(en, ko, outputs, limit):
    if not en or not ko:
        raise ValueError("번역 평가용 --eval-en과 --eval-ko가 필요합니다.")
    if len({en, ko}) != 2:
        raise ValueError("평가 영어 원문과 한국어 정답본에 서로 다른 파일을 지정하세요.")
    with (Path(en).open(encoding="utf-8-sig") as english, Path(ko).open(encoding="utf-8-sig") as korean,
          outputs[0].open("x", encoding="utf-8") as out_en, outputs[1].open("x", encoding="utf-8") as out_ko):
        rows = zip_longest(english, korean)
        count = 0
        for count, (src, ref) in enumerate(islice(rows, limit or None), 1):
            if not src or not ref or not src.strip() or not ref.strip():
                raise ValueError(f"평가 파일의 {count}행이 비어 있거나 행 수가 다릅니다.")
            out_en.write(src.strip() + "\n")
            out_ko.write(ref.strip() + "\n")
        if not count:
            raise ValueError("평가할 문장이 없습니다.")


def prepare_json(path, outputs, limit):
    bundle = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = bundle.get("data") if isinstance(bundle, dict) else bundle
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: en·ko 객체 배열 또는 data 배열이 필요합니다.")
    with (outputs[0].open("x", encoding="utf-8") as out_en,
          outputs[1].open("x", encoding="utf-8") as out_ko):
        for index, row in enumerate(islice(rows, limit or None), 1):
            if not isinstance(row, dict) or not all(
                isinstance(row.get(lang), str) and row[lang].strip() for lang in ("en", "ko")
            ):
                raise ValueError(f"{path}: {index}번째 항목에 비어 있지 않은 en과 ko가 필요합니다.")
            out_en.write(" ".join(row["en"].split()) + "\n")
            out_ko.write(" ".join(row["ko"].split()) + "\n")


def publish(index, state, run_dir, path, *, rerun=False):
    name = STAGES[index]
    work = run_dir / "details" / name
    work.mkdir(parents=True, exist_ok=True)
    attempt = Path(mkdtemp(prefix="attempt-", dir=work))
    reuse_translations = (index == 6 and not rerun and name in state["completed"]
                          and not (run_dir / PLAIN_TRANSLATION).is_file()
                          and all((run_dir / n).is_file() for n in COT_TRANSLATIONS))
    outputs = run_stage(index, attempt, state["config"], run_dir, reuse_translations=reuse_translations)
    names = ARTEFACTS[index]
    if len(names) != len(outputs) or not all(output.is_file() for output in outputs):
        raise RuntimeError(f"{name}의 필수 출력 파일이 누락됐습니다: {attempt}")
    if index in {0, 1, 2, 5} and not all(output.stat().st_size for output in outputs[:1]):
        raise ValueError(f"{name}의 다음 단계 입력이 비어 있습니다: {attempt}")
    invalidated = tuple(artifact for names_at_stage in ARTEFACTS[index:] for artifact in names_at_stage)
    fallback_plain = attempt / PLAIN_TRANSLATION
    if index == 7 and fallback_plain.is_file():
        invalidated += (PLAIN_TRANSLATION,)
        names += (PLAIN_TRANSLATION,)
        outputs.append(fallback_plain)
    for artifact in invalidated:
        link = run_dir / artifact
        if (link.exists() or link.is_symlink()) and not link.is_symlink():
            raise FileExistsError(f"관리 대상 경로에 일반 파일이 있습니다: {link}")
    for following in STAGES[index:]:
        state["completed"].pop(following, None)
    for artifact in invalidated:
        (run_dir / artifact).unlink(missing_ok=True)
    for filename, output in zip(names, outputs):
        (run_dir / filename).symlink_to(output.relative_to(run_dir))
    state["completed"][name] = str(attempt.relative_to(run_dir))
    save_state(path, state)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for operation in ("run", "init"):
        command = sub.add_parser(operation)
        for name in (*PATH_OPTIONS, "run_dir", "config"):
            command.add_argument("--" + name.replace("_", "-"), type=lambda value: Path(value).expanduser().absolute())
        command.add_argument("--stage", type=int, choices=range(1, len(STAGES) + 1),
                             help="시작 단계 번호 (1~8)")
        command.add_argument("--input", type=lambda value: Path(value).expanduser().absolute(),
                             help="시작 단계에 넘길 파일 또는 결과 폴더")
        command.add_argument("--domain")
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--eval-python", default=os.environ.get("GLOSSARY_EVAL_PYTHON"),
                             help="08단계 COMET 평가용 Python 실행 파일")
        command.add_argument("--bge-model", default=os.environ.get("GLOSSARY_BGE_MODEL") or "BAAI/bge-m3")
        command.add_argument("--gemma-model", default=os.environ.get("GLOSSARY_GEMMA_MODEL") or "google/gemma-4-12B-it")
        command.add_argument("--gemma-revision", default="main")
        command.add_argument("--load-in-4bit", action="store_true")
        command.add_argument("--local-files-only", action="store_true")
        command.add_argument("--allow-record-errors", action=argparse.BooleanOptionalAction, default=True,
                             help="검증된 목표 용어가 있으면 04단계의 개별 판정 오류를 기록하고 계속")
        command.add_argument("--max-sentences", type=int, default=0)
        command.add_argument("--eval-limit", type=int, default=0)
        command.add_argument("--batch-size", type=int, default=4)
    stage = sub.add_parser("stage")
    stage.add_argument("stage", choices=STAGES)
    stage.add_argument("--run-dir", type=lambda value: Path(value).expanduser().absolute(), required=True)
    stage.add_argument("--rerun", action="store_true")
    for name in ("eval-en", "eval-ko"):
        stage.add_argument("--" + name, type=lambda value: Path(value).expanduser().absolute(),
                           help="06-sentence-selection 재실행 시 새 평가 문장 파일")
    return parser.parse_args()


def main():
    args = parse_args()
    run_options = {value.split("=", 1)[0] for value in sys.argv[2:] if value.startswith("--")}
    run_dir = args.run_dir or ROOT / "result" / datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S_%f")
    details = run_dir / "details"
    state_path = details / "pipeline.json"
    if args.command == "stage" and not state_path.is_file():
        raise ValueError(f"초기화된 실행이 필요합니다: {run_dir}")
    details.mkdir(parents=True, exist_ok=True)
    with (details / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.is_file():
            if args.command == "init" or (args.command == "run" and run_options - {"--run-dir", "--stage"}):
                raise ValueError("기존 실행은 --run-dir과 시작 단계 --stage만 지정할 수 있습니다.")
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                      if key in (*PATH_OPTIONS, *TEXT_OPTIONS, *BOOL_OPTIONS, *INT_OPTIONS)}
            if args.config:
                if run_options - {"--config", "--run-dir", "--stage", "--input"}:
                    raise ValueError("--config와 함께 지정할 수 있는 옵션은 --run-dir, --stage, --input입니다.")
                config.update(load_config(args.config))
            start = args.stage or 1
            if args.input and args.stage is None or (start > 1 and not args.input):
                raise ValueError("새 실행에서 외부 결과로 시작하려면 --stage N과 --input 경로를 함께 지정하세요.")
            if start <= 4 and (not config["domain"] or not config["domain"].strip()):
                raise ValueError("새 실행에는 목표 분야(domain)가 필요합니다.")
            if min(config["max_sentences"], config["eval_limit"]) < 0 or config["batch_size"] < 1:
                raise ValueError("문장 제한은 0 이상, 배치 크기는 1 이상이어야 합니다.")
            imported = start_input_files(start, args.input, config) if args.input else {}
            validate_inputs(config, start)
            state = {"config": config, "completed": {}, "start_stage": start,
                     "imported_inputs": register_inputs(imported, run_dir)}
            save_state(state_path, state)
        print(f"[RESULT] {run_dir}", flush=True)
        if args.command == "init":
            return
        first = state.get("start_stage", 1) - 1
        if args.command == "run":
            indexes = range(args.stage - 1 if args.stage else first, len(STAGES))
        else:
            indexes = [STAGES.index(args.stage)]
        if indexes[0] < first:
            raise ValueError(f"이 실행은 {first + 1:02d}단계부터 시작하도록 설정됐습니다.")
        for artifact in state.get("imported_inputs", {}):
            if not (run_dir / artifact).is_file():
                raise ValueError(f"가져온 결과 파일이 없습니다: {run_dir / artifact}")
        if args.command == "stage" and (args.eval_en or args.eval_ko):
            if args.stage != STAGES[5] or not (args.rerun and args.eval_en and args.eval_ko):
                raise ValueError("평가 파일을 교체하려면 06-sentence-selection에 두 경로와 --rerun을 지정하세요.")
            state["config"].update(eval_en=str(args.eval_en), eval_ko=str(args.eval_ko), eval_json=None)
            validate_inputs(state["config"], first + 1)
        for index in indexes:
            name = STAGES[index]
            required = ARTEFACTS if index != 7 else (*ARTEFACTS[:6], COT_TRANSLATIONS, ARTEFACTS[7])
            if any(previous not in state["completed"] or not all((run_dir / file).is_file() for file in outputs)
                   for previous, outputs in zip(STAGES[first:index], required[first:index])):
                raise ValueError(f"{name} 전에 앞 단계를 완료해야 합니다.")
            if (name in state["completed"] and all((run_dir / path).is_file() for path in ARTEFACTS[index])
                    and not getattr(args, "rerun", False)):
                print(f"[SKIP] {name} 완료", flush=True)
                continue
            publish(index, state, run_dir, state_path, rerun=getattr(args, "rerun", False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"오류: {error}")
