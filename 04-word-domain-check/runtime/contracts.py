"""Input normalization and mechanical validation; semantic decisions come from the chat."""
import json


REASON_LIMIT = 720
MEMBERSHIP_REASON_LIMIT = 480


def text(value, name, limit=None, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name}: 문자열이 필요합니다")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{name}: {limit}자 이내로 작성하세요")
    return value


def normalize(original, line):
    if not isinstance(original, dict):
        raise ValueError("입력은 용어·문장을 담은 JSON 객체여야 합니다")
    if "surface_a" in original or "surface_b" in original:
        terms = {"ko": text(original.get("surface_a"), "surface_a"),
                 "en": text(original.get("surface_b"), "surface_b")}
        examples = original.get("sample_sentences")
        if not isinstance(examples, list) or not examples:
            raise ValueError("sample_sentences 배열이 필요합니다")
        contexts = []
        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("각 예문은 ko_text·en_text 객체여야 합니다")
            contexts.append({"ko": text(example.get("ko_text"), "ko_text"),
                             "en": text(example.get("en_text"), "en_text")})
    else:
        terms = {"text": text(original.get("term"), "term")}
        if "sentence" in original and "sentences" in original:
            raise ValueError("sentence 또는 sentences 중 하나를 사용하세요")
        values = [original["sentence"]] if "sentence" in original else original.get("sentences")
        if not isinstance(values, list) or not values:
            raise ValueError("sentence 문자열 또는 sentences 문자열 배열이 필요합니다")
        contexts = [{"text": text(value, "sentence")} for value in values]
    unique, seen = [], set()
    for index, context in enumerate(contexts, 1):
        key = json.dumps(context, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append({"id": f"c{index}", **context})
    return {"id": f"L{line:06d}", "input_line": line, "terms": terms,
            "contexts": unique, "original": original}


def model_item(item, max_examples=1):
    contexts = item["contexts"][:max_examples] if max_examples else item["contexts"]
    return {"id": item["id"], "terms": item["terms"], "contexts": contexts,
            "more_contexts": len(item["contexts"]) - len(contexts)}


def validate_decision(value, item, taxonomy):
    required = {"id", "status", "meaning", "assignments", "reason"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"evidence", "proposals"}:
        raise ValueError("id, status, meaning, assignments, reason 및 선택적 evidence, proposals만 사용하세요")
    if value["id"] != item["id"]:
        raise ValueError("입력 ID가 다릅니다")
    status = value["status"]
    if not isinstance(status, str) or status not in {"accept", "reject", "pending"}:
        raise ValueError("status는 accept, reject, pending 중 하나입니다")
    meaning = text(value["meaning"], "meaning", 240, empty=status != "accept")
    reason = text(value["reason"], "reason", REASON_LIMIT)
    assignments, evidence, proposals = value["assignments"], value.get("evidence", []), value.get("proposals", [])
    if not all(isinstance(v, list) for v in (assignments, evidence, proposals)):
        raise ValueError("assignments, evidence, proposals는 배열이어야 합니다")
    if status == "accept" and (not assignments or not evidence):
        raise ValueError("accept에는 도메인 배정과 원문 인용이 필요합니다")
    if status != "accept" and (assignments or proposals):
        raise ValueError("확인한 대분류가 있으면 accept로 보존하고, reject/pending에는 배정·제안을 비우세요")
    if len(evidence) > 10 or len(proposals) > 10:
        raise ValueError("인용·제안은 각각 최대 10개입니다")
    checked, seen, parents = [], set(), set()
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {"domain", "criterion", "reason"}:
            raise ValueError("각 배정에는 domain, criterion, reason이 필요합니다")
        domain = taxonomy.resolve(assignment["domain"])
        if domain != assignment["domain"]:
            raise ValueError("분류체계의 정식 전체 경로를 사용하세요: " + domain)
        if domain in seen:
            raise ValueError("배정 도메인이 중복됩니다: " + domain)
        criterion = assignment["criterion"]
        if not isinstance(criterion, str) or criterion not in taxonomy.criterion(domain):
            raise ValueError(domain + "의 criterion ID를 사용하세요: " + ", ".join(taxonomy.criterion(domain)))
        checked.append({"domain": domain, "criterion": criterion,
                        "reason": text(assignment["reason"], "assignment.reason", MEMBERSHIP_REASON_LIMIT)})
        seen.add(domain)
        parents.add(domain.split("/")[0])
    contexts = {c["id"]: c for c in item["contexts"]}
    quotes = []
    for ref in evidence:
        if not isinstance(ref, dict) or set(ref) != {"context", "language", "quote"}:
            raise ValueError("인용에는 context, language, quote가 필요합니다")
        context_id, language = ref["context"], ref["language"]
        if not isinstance(context_id, str) or not isinstance(language, str):
            raise ValueError("context·language는 문자열이어야 합니다")
        context = contexts.get(context_id)
        quote = text(ref["quote"], "quote", 400)
        if context is None or language not in {"ko", "en", "text"} or language not in context or quote not in context[language]:
            raise ValueError("인용이 이 용어의 원문 문장과 일치하지 않습니다")
        quotes.append({"context": context_id, "language": language, "quote": quote})
    proposed, paths = [], set()
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != {"path", "reason"}:
            raise ValueError("제안에는 path, reason이 필요합니다")
        path = taxonomy.proposal(proposal["path"])
        if path in paths or path.split("/")[0] not in parents:
            raise ValueError("제안은 확인한 대분류 아래의 중복되지 않은 새 소분류여야 합니다")
        proposed.append({"path": path,
                         "reason": text(proposal["reason"], "proposal.reason", MEMBERSHIP_REASON_LIMIT)})
        paths.add(path)
    # A child's membership implies its registered parent. Shared concepts can stop at the parent.
    child_parents = {domain.split("/")[0] for domain in seen if "/" in domain}
    leaves = sorted({domain for domain in seen if "/" in domain} | (parents - child_parents))
    return {"id": item["id"], "status": status, "meaning": meaning, "reason": reason,
            "assignments": checked, "evidence": quotes, "proposals": proposed,
            "domains": leaves, "major_domains": sorted(parents)}


def response_rows(raw):
    if not raw.strip():
        raise ValueError("응답 파일이 비어 있습니다")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
        for line, value in enumerate(raw.splitlines(), 1):
            if value.strip():
                try:
                    parsed.append(json.loads(value))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"응답 {line}행 JSON 문법 오류: {exc.msg}") from exc
    if isinstance(parsed, dict) and set(parsed) == {"items"}:
        parsed = parsed["items"]
    elif isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("응답은 JSONL 객체, JSON 배열 또는 items 배열이어야 합니다")
    ids = [row.get("id") for row in parsed]
    if any(not isinstance(ident, str) for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError("응답 ID가 없거나 중복됩니다")
    return parsed
