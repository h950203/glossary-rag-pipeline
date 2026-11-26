"""Independent Gemma runtime and checkpoint helpers for terminology review.

This module has no dependency on extract_pairs.py and performs no extraction.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
MODEL = "google/gemma-4-12B-it"


def without_thought(raw):
    """Keep only the final answer, even when the processor parser falls back."""
    final = re.sub(r"<\|channel>thought\b.*?<channel\|>", "", raw, flags=re.S)
    # An unfinished thought is not a final answer that JSON repair may approve.
    return "" if "<|channel>thought" in final else final


def open_thought(raw):
    return raw.rfind("<|channel>thought") > raw.rfind("<channel|>")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def placement_devices(model, gpu_count):
    """Single-device Transformers loads may omit hf_device_map entirely."""
    device_map = getattr(model, "hf_device_map", None)
    devices = (device_map.values() if device_map else
               (parameter.device for parameter in model.parameters()))
    used = {str(device).removeprefix("cuda:") for device in devices}
    if used & {"cpu", "disk", "meta"}:
        raise RuntimeError("모델이 GPU 외의 장치로 배치됐습니다. 선택 GPU의 여유 메모리를 "
                           f"확인하거나 GPU를 추가하세요. 배치: {sorted(used)}")
    expected = {str(i) for i in range(gpu_count)}
    if not expected.issubset(used):
        raise RuntimeError(f"모든 선택 GPU에 모델을 배치하지 못했습니다: {sorted(used)}")
    return used


class Generator:
    def __init__(self, args, revision):
        load_started = time.perf_counter()
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다")
        gpu_count = torch.cuda.device_count()
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(전체)')}; "
              f"인식된 GPU {gpu_count}개", flush=True)
        common = {"revision": revision, "local_files_only": args.local_files_only}
        if args.cache_dir:
            common["cache_dir"] = args.cache_dir
        if getattr(args, "profile_timing", False):
            print("[시간] 모델·프로세서 로딩 시작", flush=True)
        processor_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(args.model, **common)
        processor_seconds = time.perf_counter() - processor_started
        options = {**common, "device_map": {"": 0} if gpu_count == 1 else "balanced",
                   "dtype": torch.float16}
        # Reserve memory on every GPU for KV cache and generation activations.
        options["max_memory"] = {i: int(torch.cuda.mem_get_info(i)[0] * 0.85)
                                 for i in range(torch.cuda.device_count())}
        if args.load_in_4bit:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        weights_started = time.perf_counter()
        self.model = AutoModelForMultimodalLM.from_pretrained(args.model, **options).eval()
        used = placement_devices(self.model, gpu_count)
        print(f"GPU 배치 (프로세스 내부 번호): {sorted(used)}", flush=True)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.torch, self.args = torch, args
        torch.manual_seed(42)
        self.load_timing = {"processor_seconds": processor_seconds,
                            "weights_and_placement_seconds": time.perf_counter() - weights_started,
                            "total_seconds": time.perf_counter() - load_started}
        if getattr(args, "profile_timing", False):
            print(f"[시간] 로딩 완료: {self.load_timing['total_seconds']:.2f}초 "
                  f"(프로세서 {processor_seconds:.2f}초, "
                  f"가중치·배치 {self.load_timing['weights_and_placement_seconds']:.2f}초)", flush=True)

    def _sync_timing(self):
        # Synchronize only in diagnostic mode; normal execution stays unchanged.
        if (getattr(self.args, "profile_timing", False)
                and self.model.get_input_embeddings().weight.device.type == "cuda"):
            for index in range(self.torch.cuda.device_count()):
                self.torch.cuda.synchronize(index)

    def generate(self, messages, max_new_tokens=None):
        started = time.perf_counter()
        profile = getattr(self.args, "profile_timing", False)
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
                    return_dict=True, return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=getattr(self.args, "enable_thinking", False))
        length = inputs["input_ids"].shape[-1]
        if length > self.args.max_input_tokens:
            raise RuntimeError(f"입력 {length} tokens: 한도 초과, 원문을 자르지 않고 중단합니다")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        self._sync_timing()
        prepared = time.perf_counter()
        requested_budget = max_new_tokens or self.args.max_new_tokens
        budget = requested_budget
        text_config = getattr(self.model.config, "text_config", self.model.config)
        context = (text_config.get("max_position_embeddings") if isinstance(text_config, dict)
                   else getattr(text_config, "max_position_embeddings", None))
        if isinstance(context, int):
            budget = min(budget, context - length)
        if budget < 1:
            raise RuntimeError("모델 context에 최종 응답을 생성할 공간이 없습니다")
        thinking = getattr(self.args, "enable_thinking", False)
        # Reserve part of the TOTAL budget for the final JSON. If reasoning does
        # not finish, explicitly close Gemma's thought channel and continue it.
        first_budget = (min(getattr(self.args, "max_thinking_tokens", 2048), max(1, budget // 2))
                        if thinking else budget)
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        forced_end = False
        first_seconds, continuation_seconds = 0.0, 0.0
        first_tokens, continuation_tokens, injected_tokens = 0, 0, 0
        continuation_input_tokens = 0
        if profile:
            print(f"[시간] 입력 {length} tokens; 1차 생성 시작 (상한 {first_budget})", flush=True)
        with self.torch.inference_mode():
            self._sync_timing()
            first_started = time.perf_counter()
            out = self.model.generate(**inputs, max_new_tokens=first_budget,
                                      do_sample=False, use_cache=True)
            self._sync_timing()
            first_seconds = time.perf_counter() - first_started
            used = out.shape[-1] - length
            first_tokens = used
            if profile:
                print(f"[시간] 1차 생성 {first_seconds:.2f}초 / {first_tokens} tokens", flush=True)
            if thinking and used >= first_budget and int(out[0, -1]) not in eos_ids and used < budget:
                raw = self.processor.decode(out[0][length:], skip_special_tokens=False)
                if open_thought(raw):
                    closing = self.processor.tokenizer.encode("\n<channel|>\n", add_special_tokens=False)
                    if used + len(closing) < budget:
                        tail = self.torch.tensor([closing], dtype=out.dtype, device=out.device)
                        out = self.torch.cat((out, tail), dim=-1)
                        injected_tokens = len(closing)
                        forced_end = True
                        print(f"CoT 사고 한도 {first_budget} tokens 도달: 사고 채널을 닫고 최종 JSON 생성", flush=True)
                remaining = budget - (out.shape[-1] - length)
                continuation = {**inputs, "input_ids": out, "attention_mask": self.torch.ones_like(out)}
                for key in ("mm_token_type_ids", "token_type_ids"):
                    if key in continuation:
                        continuation[key] = self.torch.zeros_like(out)
                continuation_input_tokens = out.shape[-1]
                if profile:
                    print(f"[시간] 이어 생성 시작: {continuation_input_tokens} tokens 재입력 "
                          f"(이전 호출 KV 캐시 미전달)", flush=True)
                self._sync_timing()
                continuation_started = time.perf_counter()
                out = self.model.generate(**continuation, max_new_tokens=remaining,
                                          do_sample=False, use_cache=True)
                self._sync_timing()
                continuation_seconds = time.perf_counter() - continuation_started
                continuation_tokens = out.shape[-1] - continuation_input_tokens
                if profile:
                    print(f"[시간] 이어 생성 {continuation_seconds:.2f}초 / "
                          f"{continuation_tokens} tokens", flush=True)
        self.last_truncated = out.shape[-1] - length >= budget and int(out[0, -1]) not in eos_ids
        self.last_generation = {"requested_budget": requested_budget, "effective_budget": budget,
                                "generated_tokens": out.shape[-1] - length,
                                "thinking_token_limit": first_budget if thinking else 0,
                                "forced_thought_end": forced_end, "truncated": self.last_truncated,
                                "input_tokens": length, "first_generated_tokens": first_tokens,
                                "continuation_generated_tokens": continuation_tokens,
                                "injected_tokens": injected_tokens,
                                "continuation_input_tokens": continuation_input_tokens,
                                "first_tokens_per_second": first_tokens / first_seconds if first_seconds else None,
                                "continuation_tokens_per_second": (continuation_tokens / continuation_seconds
                                                                    if continuation_seconds else None),
                                "timing": {"prepare_seconds": prepared - started,
                                           "first_generate_seconds": first_seconds,
                                           "continuation_generate_seconds": continuation_seconds}}
        try:
            return self._final_response(out, length)
        finally:
            elapsed = time.perf_counter() - started
            self.last_generation["timing"].update(
                total_seconds=elapsed,
                other_seconds=max(0.0, elapsed - (prepared - started) - first_seconds - continuation_seconds))

    def _final_response(self, out, length):
        raw = self.processor.decode(out[0][length:], skip_special_tokens=False)
        self.last_raw_response = raw
        final_raw = without_thought(raw)
        if not final_raw.strip():
            return ""
        try:
            parsed = self.processor.parse_response(final_raw)
        except (ValueError, TypeError, AttributeError):
            parsed = None
        content = parsed.get("content") if isinstance(parsed, dict) else None
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            joined = "\n".join(x["text"] for x in content if isinstance(x, dict)
                               and x.get("type") == "text" and isinstance(x.get("text"), str))
            if joined.strip():
                return joined
        if getattr(self.args, "enable_thinking", False):
            return final_raw
        return self.processor.decode(out[0][length:], skip_special_tokens=True)


def decode_object(response):
    """Recover JSON syntax only; the caller must still validate every field."""
    value = response.strip().lstrip("\ufeff")
    if not value:
        raise ValueError("모델 응답이 비어 있습니다")
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    notes = []
    try:
        obj = json.loads(value)
    except ValueError:
        start = value.find("{")
        if start < 0:
            raise ValueError("응답에서 JSON 객체를 찾지 못했습니다")
        candidate = value[start:]
        try:
            obj, end = json.JSONDecoder().raw_decode(candidate)
            if candidate[end:].strip().startswith(("{", "[")):
                raise ValueError("여러 JSON 객체가 반환됐습니다")
            notes.append("JSON 외부 설명 제거")
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
            except ImportError as exc:
                raise ValueError("JSON 문법 오류: json-repair 설치 또는 모델 재요청이 필요합니다") from exc
            obj = repair_json(candidate, return_objects=True)
            notes.append("json-repair로 JSON 문법 복구")
    if not isinstance(obj, dict):
        raise ValueError("단일 JSON 객체가 필요합니다")
    return obj, notes


def request(generator, system, payload, validator, retries=2):
    error = None
    previous = ""
    attempts = []
    request_started = time.perf_counter()
    generator.last_request = {"attempts": attempts, "generations": [], "validation_seconds": 0.0}
    budget = generator.args.max_new_tokens if isinstance(generator, Generator) else None
    ceiling = max(budget, getattr(generator.args, "max_retry_tokens", budget * 4)) if budget is not None else None
    for attempt in range(retries + 1):
        generator.last_request["attempt_count"] = attempt + 1
        message = json.dumps(payload, ensure_ascii=False)
        if error:
            message += ("\nPrevious response invalid. Correct: " + error[:600]
                        + "\nPrevious response (may be truncated):\n" + previous[:2500]
                        + "\nReturn ONE complete JSON object only. Use double quotes and valid commas/colons.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
        if getattr(getattr(generator, "args", None), "profile_timing", False):
            print(f"[시간] 요청 {attempt + 1}/{retries + 1} 시작", flush=True)
        if isinstance(generator, Generator):
            response = generator.generate(messages, max_new_tokens=budget)
        else:
            response = generator.generate(messages)
        generation = getattr(generator, "last_generation", None)
        if generation:
            generator.last_request["generations"].append({"attempt": attempt + 1, **generation})
        validation_started = time.perf_counter()
        try:
            if getattr(generator, "last_truncated", False):
                raise ValueError(f"출력이 생성 토큰 한도({budget})에서 잘렸습니다")
            result = validator(response)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            attempts.append({"attempt": attempt + 1, "error": error, "response": response,
                             "raw_response": getattr(generator, "last_raw_response", response)})
            if getattr(getattr(generator, "args", None), "profile_timing", False):
                print(f"[시간] 응답 검증 실패 (요청 {attempt + 1}): {error}", flush=True)
            if budget is not None and getattr(generator, "last_truncated", False):
                if generation and generation["effective_budget"] < budget:
                    error += "; 모델 context 한도 때문에 생성 예산을 더 늘릴 수 없습니다"
                    break
                next_budget = min(ceiling, budget * 2)
                if next_budget <= budget:
                    error += f"; 재시도 상한 {ceiling} tokens 도달"
                    break
                if attempt < retries:
                    print(f"응답 재요청 {attempt + 1}/{retries}: 생성 한도 {budget} → {next_budget} tokens", flush=True)
                budget = next_budget
            previous = response
        else:
            generator.last_request["total_seconds"] = time.perf_counter() - request_started
            return result, None
        finally:
            generator.last_request["validation_seconds"] += time.perf_counter() - validation_started
    generator.last_request["total_seconds"] = time.perf_counter() - request_started
    return None, error


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(path, signature):
    entries, valid_end = [], 0
    with path.open("rb") as f:
        header = json.loads(f.readline())
        if header.get("signature") != signature:
            raise ValueError("입력·모델·설정이 체크포인트와 다릅니다. 새 출력 폴더를 사용하세요")
        valid_end = f.tell()
        while raw := f.readline():
            try:
                entries.append(json.loads(raw))
            except (ValueError, UnicodeError):
                if f.read():
                    raise ValueError("체크포인트 중간 손상")
                break
            valid_end = f.tell()
    with path.open("r+b") as f:
        f.truncate(valid_end)
        f.seek(valid_end - 1)
        if f.read(1) != b"\n":
            f.seek(0, 2)
            f.write(b"\n")
    return header, entries
