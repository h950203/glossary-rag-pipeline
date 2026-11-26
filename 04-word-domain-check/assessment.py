"""Measure agreement with a saved Codex run or a user-reviewed gold file."""
from collections import Counter


def in_target(domains, target):
    if target["minor"]:
        return target["path"] in domains
    return any(domain.split("/")[0] == target["major"] for domain in domains)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(items, predictions, reference, target, kind):
    total = Counter()
    confusion = Counter()
    cases = []
    for item in items:
        ident = item["id"]
        if ident not in reference:
            continue
        expected = reference[ident]
        predicted = predictions[ident]
        expected_status, predicted_status = expected["status"], predicted["status"]
        expected_domains, predicted_domains = set(expected["domains"]), set(predicted["domains"])
        expected_majors = {d.split("/")[0] for d in expected_domains}
        predicted_majors = {d.split("/")[0] for d in predicted_domains}
        expected_target = in_target(expected_domains, target)
        predicted_target = in_target(predicted_domains, target)
        exact = expected_status == predicted_status and expected_domains == predicted_domains
        total.update({
            "compared": 1, "exact_decision": int(exact), "status_match": int(expected_status == predicted_status),
            "reference_accepted": int(expected_status == "accept"),
            "full_domain_match_on_reference_accept": int(expected_status == "accept" and exact),
            "major_match_on_reference_accept": int(expected_status == "accept" and expected_majors == predicted_majors),
            "reference_rejected": int(expected_status == "reject"),
            "reference_reject_predicted_accept": int(expected_status == "reject" and predicted_status == "accept"),
            "reference_pending_predicted_accept": int(expected_status == "pending" and predicted_status == "accept"),
            "ref_target": int(expected_target), "pred_target": int(predicted_target),
            "target_overlap": int(expected_target and predicted_target),
            "operational_errors": int(predicted_status == "error"),
            "unprocessed": int(predicted_status == "unprocessed"),
        })
        confusion[(expected_status, predicted_status)] += 1
        if not exact:
            cases.append({"id": ident, "terms": item["terms"], "reference_status": expected_status,
                          "reference_domains": sorted(expected_domains), "reference_reason": expected.get("reason"),
                          "predicted_status": predicted_status, "predicted_domains": sorted(predicted_domains),
                          "predicted_reason": predicted.get("reason"),
                          "reference_target": expected_target, "predicted_target": predicted_target})
    return {
        "benchmark": kind, "counts": dict(total), "unlabelled": len(items) - total["compared"],
        "status_confusion": [{"reference": a, "prediction": b, "count": count}
                             for (a, b), count in sorted(confusion.items())],
        "rates": {
            "status_agreement": ratio(total["status_match"], total["compared"]),
            "exact_decision_agreement": ratio(total["exact_decision"], total["compared"]),
            "domain_agreement_when_reference_accepts": ratio(total["full_domain_match_on_reference_accept"],
                                                             total["reference_accepted"]),
            "major_agreement_when_reference_accepts": ratio(total["major_match_on_reference_accept"],
                                                            total["reference_accepted"]),
            "target_precision_against_reference": ratio(total["target_overlap"], total["pred_target"]),
            "target_recall_against_reference": ratio(total["target_overlap"], total["ref_target"]),
            "accept_rate_when_reference_rejects": ratio(total["reference_reject_predicted_accept"],
                                                        total["reference_rejected"]),
        },
    }, cases
