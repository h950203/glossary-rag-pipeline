"""Independent interventions over the v1.9.1 interpretation/decision procedure."""
from runtime.contracts import MEMBERSHIP_REASON_LIMIT, text, validate_decision
from runtime.gemma_backend import decode_object


PROFILES = {
    "1.9.2": "concept_gate",
    "1.9.3": "major_recheck",
    "1.9.4": "child_review",
    "1.9.4.1": "child_review",
    "1.9.5": "assignment_codes",
}

DECISION_FIELDS = ("id", "status", "meaning", "reason", "assignments", "evidence", "proposals")


def assignment_catalog(rules):
    entries = ((domain, criterion, definition) for domain, criteria in sorted(rules["criteria"].items())
               for criterion, definition in sorted(criteria.items()))
    return {f"A{index:03d}": {"domain": domain, "criterion": criterion, "definition": definition}
            for index, (domain, criterion, definition) in enumerate(entries, 1)}


def classify_payload(packet, rules, catalog=None):
    if catalog is None:
        return {**packet, "taxonomy": rules}
    return {**packet, "assignment_catalog": catalog, "taxonomy_policy": rules["policy"]}


def validate_coded_decision(raw, scope, taxonomy, catalog):
    value, _ = decode_object(raw)
    assignments = value.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("assignments 배열이 필요합니다")
    decoded = []
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {"assignment_id", "reason"}:
            raise ValueError("배정에는 assignment_id와 reason만 사용하세요")
        code = assignment["assignment_id"]
        if not isinstance(code, str) or code not in catalog:
            raise ValueError("assignment_catalog에서 실제 키를 그대로 사용하세요")
        entry = catalog[code]
        decoded.append({"domain": entry["domain"], "criterion": entry["criterion"], "reason": assignment["reason"]})
    return validate_decision({**value, "assignments": decoded}, scope, taxonomy)


def validate_major_recheck(raw, scope, taxonomy, additional_ids):
    value, _ = decode_object(raw)
    decision = validate_decision(value, scope, taxonomy)
    if decision["proposals"] or any("/" in domain for domain in decision["domains"]):
        raise ValueError("구제 판정에서는 대분류만 배정하고 proposals를 비우세요")
    if decision["status"] == "accept" and not any(e["context"] in additional_ids for e in decision["evidence"]):
        raise ValueError("대분류 구제 채택에는 additional_context_ids 중 새 예문의 정확한 인용이 필요합니다")
    return decision


def review_fields(decision, rules):
    return [{"domain": domain, "child_criteria": rules["criteria"][domain],
             "parent": domain.split("/")[0], "parent_scope": rules["criteria"][domain.split("/")[0]]["scope"]}
            for domain in decision["domains"] if "/" in domain]


def validate_child_review(raw, scope, fields):
    value, _ = decode_object(raw)
    if set(value) != {"id", "checks"} or value["id"] != scope["id"] or not isinstance(value["checks"], list):
        raise ValueError("실제 입력 id와 checks 배열을 반환하세요")
    expected, seen = {f["domain"] for f in fields}, set()
    for check in value["checks"]:
        if not isinstance(check, dict) or set(check) != {"domain", "verdict", "reason"}:
            raise ValueError("각 검토에는 domain, verdict, reason이 필요합니다")
        if check["domain"] not in expected or check["domain"] in seen:
            raise ValueError("검토할 소분류를 정확히 한 번씩 반환하세요")
        if check["verdict"] not in {"keep", "parent", "hold"}:
            raise ValueError("소분류 검토 verdict는 keep, parent, hold 중 하나입니다")
        text(check["reason"], "check.reason", MEMBERSHIP_REASON_LIMIT)
        seen.add(check["domain"])
    if seen != expected:
        raise ValueError("검토할 소분류마다 결과가 필요합니다")
    return value


def apply_child_review(decision, review, scope, taxonomy):
    checks = {c["domain"]: c for c in review["checks"]}
    original = decision["assignments"]
    kept = {a["domain"]: a for a in original if a["domain"] not in checks or checks[a["domain"]]["verdict"] == "keep"}
    for domain, check in checks.items():
        if check["verdict"] == "parent":
            parent = domain.split("/")[0]
            kept.setdefault(parent, {"domain": parent, "criterion": "scope", "reason": check["reason"]})
    if all(check["verdict"] == "keep" for check in checks.values()):
        return decision
    value = {key: decision[key] for key in DECISION_FIELDS}
    parents = {domain.split("/")[0] for domain in kept}
    value.update(status="accept" if kept else "pending", assignments=list(kept.values()),
                 reason="소분류 검토에서 확인된 소속만 보존함" if kept else "소분류와 대분류 소속을 확정하기 어려움",
                 proposals=[p for p in decision["proposals"] if p["path"].split("/")[0] in parents])
    return validate_decision(value, scope, taxonomy)
