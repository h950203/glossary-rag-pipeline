"""2.1.1 축소 실행 로그와 호환되는 COMET·LaBSE 누적 평가기."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from comet import download_model, load_from_checkpoint
from transformers import AutoModel, AutoTokenizer


############### 설정 구간 ###############
# 실행 전에 아래 경로와 언어만 지정하세요. CLI 인자는 필요하지 않습니다.
RUN_CONFIG = {
    "input_log_file": "log/experiments_log.json",
    "source_lang": "en",
    "target_lang": "ko",
    "output_log_file": "log/evaluation_log.json",
    # 실험 번호·파일명 규칙과 무관하게, 실험 로그가 가리키는 parameter와
    # prompt 파일이 현재 폴더에 실제 존재하는 완료 실험만 평가합니다.
    "parameter_dir": "parameters",
    "prompt_dir": "prompts",
    "require_existing_prompt_file": True,
}
############### 설정 구간 끝 ###############

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"


def load_json(path: Path) -> dict[str, Any]:
    """JSON 객체를 읽고, 읽기 쉬운 오류를 제공합니다."""
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"올바른 JSON 로그가 아닙니다: {path} ({error})") from error

    if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
        raise ValueError(f"'experiments' 목록이 없는 로그입니다: {path}")
    return data


def load_output_log(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {"schema_version": "1.0", "experiments": []}


def save_json(path: Path, data: dict[str, Any]) -> None:
    """최고 기록을 로그 상단에 두고 JSON을 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_data = {
        "schema_version": data.get("schema_version", "1.0"),
        "best_evaluations": data.get("best_evaluations", {}),
        **{
            key: value
            for key, value in data.items()
            if key not in {"schema_version", "best_evaluations", "experiments"}
        },
        "experiments": data["experiments"],
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(ordered_data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def load_text_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def get_evaluation_paths(experiment: dict[str, Any], target_lang: str) -> tuple[Path, Path, Path]:
    """2.1.1의 input/output 경로를 우선 사용하고 이전 로그도 지원합니다."""
    input_record = experiment.get("input", {})
    output_record = experiment.get("output", {})
    hyperparameters = experiment.get("hyperparameters", {})
    input_file = input_record.get("file") if isinstance(input_record, dict) else None
    output_txt = output_record.get("txt") if isinstance(output_record, dict) else None
    input_file = input_file or hyperparameters.get("input_file")
    output_txt = output_txt or (
        f"{hyperparameters['output_file']}.txt"
        if isinstance(hyperparameters.get("output_file"), str) else None
    )
    if not isinstance(input_file, str) or not isinstance(output_txt, str):
        raise ValueError("input.file 또는 output.txt가 없습니다.")

    source_path = Path(input_file)
    reference_path = source_path.with_suffix(f".{target_lang}")
    hypothesis_path = Path(output_txt)
    return source_path, reference_path, hypothesis_path


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def calculate_labse(
    candidates: list[str], references: list[str], tokenizer: Any, model: Any, device: str
) -> float:
    """LaBSE 문장 임베딩의 후보-참조 평균 코사인 유사도를 반환합니다."""
    print(f"  [LaBSE] 계산 중... (문장 수: {len(candidates)})")
    similarities: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), 32):
            candidate_batch = candidates[start : start + 32]
            reference_batch = references[start : start + 32]
            encoded = tokenizer(
                candidate_batch + reference_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            embeddings = functional.normalize(
                mean_pooling(model(**encoded), encoded["attention_mask"]), p=2, dim=1
            )
            batch_size = len(candidate_batch)
            similarities.extend((embeddings[:batch_size] * embeddings[batch_size:]).sum(dim=1).cpu().tolist())
    return sum(similarities) / len(similarities)


def calculate_comet(
    sources: list[str], references: list[str], candidates: list[str], comet_model: Any, device: str
) -> float:
    """PyTorch Lightning predict 루프를 거치지 않고 COMET 점수를 계산합니다."""
    print(f"  [COMET] 계산 중... (문장 수: {len(candidates)})")
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, candidates, references)]
    scores: list[float] = []
    comet_model.to(device)
    comet_model.eval()
    with torch.no_grad():
        for start in range(0, len(data), 8):
            model_inputs = comet_model.prepare_sample(data[start : start + 8], stage="predict")
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            batch_scores = comet_model(**model_inputs).score
            scores.extend(batch_scores.detach().cpu().tolist())
    return sum(scores) / len(scores)


def rebuild_best_evaluations(output_log: dict[str, Any]) -> None:
    """완료된 모든 평가에서 언어쌍 및 지표별 최고 기록을 다시 만듭니다."""
    best_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    for experiment in output_log["experiments"]:
        if not isinstance(experiment, dict):
            continue
        evaluation = experiment.get("evaluation", {})
        if evaluation.get("status") != "completed":
            continue
        language_pair = f"{evaluation['source_lang']}-{evaluation['target_lang']}"
        pair_best = best_evaluations.setdefault(language_pair, {})
        for metric in ("comet", "labse"):
            score = evaluation.get(metric)
            if not isinstance(score, (int, float)):
                continue
            if metric not in pair_best or score > pair_best[metric]["score"]:
                source_path, _, hypothesis_path = get_evaluation_paths(
                    experiment, evaluation["target_lang"]
                )
                pair_best[metric] = {
                    "experiment_id": experiment.get("id"),
                    "score": score,
                    "sentence_count": evaluation.get("sentence_count"),
                    "input_file": str(source_path),
                    "output_file": str(hypothesis_path),
                }
    output_log["best_evaluations"] = best_evaluations


def placeholder_evaluation(source_lang: str, target_lang: str, reason: str) -> dict[str, Any]:
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "status": "skipped",
        "reason": reason,
        "comet": "-",
        "labse": "-",
    }


def resolve_config_file(recorded_path: Any, directory: Path) -> Path | None:
    """로그 경로 또는 현재 config 폴더의 같은 파일명을 해석합니다."""
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        return None
    direct = Path(recorded_path)
    if direct.is_file():
        return direct
    by_name = directory / direct.name
    return by_name if by_name.is_file() else None


def has_existing_config_files(
    experiment: dict[str, Any], parameter_dir: Path, prompt_dir: Path, require_prompt: bool
) -> bool:
    """번호 규칙 대신 로그의 parameter/prompt 파일 존재 여부로 대상 선정."""
    files = experiment.get("files", {})
    if not isinstance(files, dict):
        return False
    parameter_record = (
        files.get("hyperparameter_file")
        or files.get("parameter_file")
        or files.get("parameters_file")
    )
    prompt_record = files.get("prompt_policy_file") or files.get("prompt_file")
    parameter_path = resolve_config_file(parameter_record, parameter_dir)
    prompt_path = resolve_config_file(prompt_record, prompt_dir)
    return parameter_path is not None and (prompt_path is not None or not require_prompt)


def main() -> None:
    input_log_path = Path(RUN_CONFIG["input_log_file"])
    output_log_path = Path(RUN_CONFIG["output_log_file"])
    source_lang = RUN_CONFIG["source_lang"].lower()
    target_lang = RUN_CONFIG["target_lang"].lower()
    parameter_dir = Path(RUN_CONFIG["parameter_dir"])
    prompt_dir = Path(RUN_CONFIG["prompt_dir"])
    require_prompt = bool(RUN_CONFIG.get("require_existing_prompt_file", True))
    if not source_lang or not target_lang:
        raise ValueError("source_lang와 target_lang을 지정하세요.")

    input_log = load_json(input_log_path)
    output_log = load_output_log(output_log_path)
    evaluated_ids = {
        experiment.get("id")
        for experiment in output_log["experiments"]
        if isinstance(experiment, dict) and experiment.get("id")
    }
    pending_experiments = [
        experiment for experiment in input_log["experiments"]
        if (
            isinstance(experiment, dict)
            and experiment.get("id") not in evaluated_ids
            and has_existing_config_files(experiment, parameter_dir, prompt_dir, require_prompt)
        )
    ]
    print(
        f"입력 로그: {input_log_path} (총 {len(input_log['experiments'])}개, "
        f"parameter/prompt 파일 존재 조건: {parameter_dir} / {prompt_dir})"
    )
    print(f"출력 로그: {output_log_path} (기록됨 {len(evaluated_ids)}개, 평가 대상 {len(pending_experiments)}개)")

    comet_model = None
    labse_tokenizer = None
    labse_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for experiment in pending_experiments:
        record = deepcopy(experiment)
        experiment_id = record.get("id", "<id 없음>")
        print(f"\n[{experiment_id}] 처리 중...")

        if record.get("status") != "completed":
            record["evaluation"] = placeholder_evaluation(
                source_lang, target_lang, "experiment_status_is_not_completed"
            )
        else:
            try:
                source_path, reference_path, hypothesis_path = get_evaluation_paths(record, target_lang)
                missing_paths = [path for path in (source_path, reference_path, hypothesis_path) if not path.exists()]
                if missing_paths:
                    raise FileNotFoundError(", ".join(str(path) for path in missing_paths))

                sources = load_text_file(source_path)
                references = load_text_file(reference_path)
                candidates = load_text_file(hypothesis_path)
                sentence_count = min(len(sources), len(references), len(candidates))
                if sentence_count == 0:
                    raise ValueError("평가할 문장이 없습니다.")
                if len({len(sources), len(references), len(candidates)}) != 1:
                    print(f"  경고: 문장 수 불일치 → {sentence_count}개로 평가합니다.")
                sources, references, candidates = (
                    sources[:sentence_count], references[:sentence_count], candidates[:sentence_count]
                )

                if comet_model is None:
                    print(f"  [COMET] 모델 로딩 중: {COMET_MODEL_NAME}")
                    comet_model = load_from_checkpoint(download_model(COMET_MODEL_NAME))
                    print(f"  [LaBSE] 모델 로딩 중: {LABSE_MODEL_NAME}")
                    labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
                    labse_model = AutoModel.from_pretrained(LABSE_MODEL_NAME).to(device)

                comet_score = calculate_comet(
                    sources, references, candidates, comet_model, device
                )
                labse_score = calculate_labse(candidates, references, labse_tokenizer, labse_model, device)
                record["evaluation"] = {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "status": "completed",
                    "sentence_count": sentence_count,
                    "comet": comet_score,
                    "labse": labse_score,
                }
                print(f"  COMET: {comet_score:.4f}, LaBSE: {labse_score:.4f}")
            except (FileNotFoundError, ValueError, OSError) as error:
                print(f"  평가 생략: {error}")
                record["evaluation"] = placeholder_evaluation(source_lang, target_lang, str(error))

        output_log["experiments"].append(record)
        rebuild_best_evaluations(output_log)
        save_json(output_log_path, output_log)

    rebuild_best_evaluations(output_log)
    save_json(output_log_path, output_log)
    print(f"\n완료: {len(pending_experiments)}개 기록. 결과 로그: {output_log_path}")


if __name__ == "__main__":
    main()
