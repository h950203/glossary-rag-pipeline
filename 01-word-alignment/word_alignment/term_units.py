"""Contextual nominal/verbal units with immutable source spans and canonical forms."""
from functools import lru_cache
from pathlib import Path
import unicodedata

from .lab_common import digest, file_hash

SCHEMA = "term-units-v1.1"
NOUN = {"NNG", "NNP", "SL", "SH"}
NOMINAL = NOUN | {"SN", "NR", "MM", "XPN", "XSN"}
PREDICATE = {"VV", "VA", "XSV", "XSA", "VCP"}
LINKS = {"의", "및", "·", "ㆍ", "-", "/"}


def norm(text):
    return unicodedata.normalize("NFC", text).casefold()


def term_settings(args):
    from importlib.metadata import version
    return {"schema": SCHEMA, "max_term_tokens": args.max_term_tokens,
            "term_types": sorted(args.term_types), "code": file_hash(Path(__file__)),
            "nltk": version("nltk"), "kiwipiepy": version("kiwipiepy")}


class TermAnalyzer:
    def __init__(self, args):
        import nltk
        from nltk.tag import PerceptronTagger
        from nltk.stem import WordNetLemmatizer
        root = Path(__file__).resolve().parents[1]
        nltk.data.path.insert(0, str(root / "nltk_data"))
        try:
            self.tagger = PerceptronTagger(lang="eng")
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("patients", "n")
        except LookupError as exc:
            raise ValueError("NLTK 자원이 없습니다. 이 프로젝트의 setup_resources.py를 실행하세요") from exc
        self.lemma = lru_cache(maxsize=100000)(lemmatizer.lemmatize)
        self.joiner = None
        self.args = args
        self.signature = digest(term_settings(args))

    def analyze(self, row, suite):
        result = {}
        # Kiwi primitive analysis is cached by TokenizerSuite: no second parse
        # when the selected Korean profile already used Kiwi.
        primitives = suite.tokens(row["ko"], "ko", "kiwi_morph")
        self.joiner = getattr(suite, "resources", {}).get("kiwi", self.joiner)
        morphs, seen = [], set()
        for token in primitives:
            pieces = token.get("morphs") or [{"form": token["text"], "tag": token["pos"][0],
                                               "start": token["start"], "end": token["end"]}]
            for piece in pieces:
                identity = tuple(piece[k] for k in ("form", "tag", "start", "end"))
                if identity not in seen:
                    morphs.append(piece); seen.add(identity)
        en = row["tokens"]["en"]
        tagged = self.tagger.tag([t["text"] for t in en])
        english = [{"form": t["text"], "tag": tag, "start": t["start"], "end": t["end"]}
                   for t, (_, tag) in zip(en, tagged)]
        result["ko"] = self.korean(row["ko"], morphs, row["tokens"]["ko"])
        result["en"] = self.english(row["en"], english, row["tokens"]["en"])
        row["term_analysis"] = {"schema": SCHEMA, "signature": self.signature,
                                "ko_morphs": morphs, "en_tokens": english}
        row["term_units"] = result

    def unit(self, text, language, kind, pieces, profile, canonical):
        start, end = pieces[0]["start"], pieces[-1]["end"]
        ids = [i for i, t in enumerate(profile) if t["start"] < end and start < t["end"]]
        # Avoid pretending a component inside a merged lexical token is a
        # separate source word; complete compound units remain eligible.
        if not ids or profile[ids[0]]["start"] != start or profile[ids[-1]]["end"] != end:
            return None
        tokens = [norm(value) for value in canonical if value.strip()]
        if not tokens:
            return None
        normalized = " ".join(tokens)
        display = normalized
        if language == "ko":
            display = display.replace(" 의 ", "의 ").replace(" · ", "·").replace(" ㆍ ", "ㆍ")
        surface = text[start:end]
        return {"unit_id": digest([language, kind, start, end, tokens])[:24], "type": kind,
                "surface": surface, "span": [start, end], "canonical": display, "normalized": normalized,
                "tokens": tokens, "pos": [p["tag"] for p in pieces], "token_ids": ids,
                "analysis_schema": SCHEMA, "analysis_signature": self.signature,
                "normalization_operations": [] if norm(surface) == display else ["canonical_form"],
                "shape_valid": True}

    @staticmethod
    def ko_np(pieces):
        if not pieces or pieces[0]["tag"] in {"XSN", "XSV", "XSA", "VCP", "EP", "ETM", "JKG", "JC"}:
            return False
        if pieces[0]["form"] in {"후", "전", "중", "간", "등", "및", "때"} or pieces[-1]["form"] in {"후", "전", "중", "간", "등", "및", "때"}:
            return False
        last = pieces[-1]
        if last["tag"] not in NOUN | {"XSN"} or (last["tag"] == "XSN" and last["form"] == "적"):
            return False
        for i, p in enumerate(pieces):
            tag = p["tag"]
            if tag in NOMINAL:
                continue
            if tag in {"JKG", "JC", "SW"} and p["form"] in LINKS and 0 < i < len(pieces)-1:
                continue
            if tag in PREDICATE | {"XR", "EP", "ETM"}:
                # Predicates are allowed only in a completed attributive block,
                # followed by a nominal head, never as a finite clause.
                if tag == "ETM":
                    if not any(q["tag"] in PREDICATE for q in pieces[:i]):
                        return False
                elif not any(q["tag"] == "ETM" for q in pieces[i+1:]):
                    return False
                continue
            return False
        return True

    @staticmethod
    def ko_parts(text, pieces):
        parts, i, prefix = [], 0, ""
        while i < len(pieces):
            j = i + 1
            while j < len(pieces) and pieces[j]["start"] <= pieces[j-1]["end"]:
                j += 1
            word = pieces[i:j]
            if any(p["tag"] == "ETM" for p in word):
                parts.append(text[word[0]["start"]:word[-1]["end"]]); i = j; continue
            p = pieces[i]
            if p["tag"] == "XPN":
                prefix += p["form"]
            elif p["tag"] == "XSN" and parts:
                parts[-1] += p["form"]
            elif p["tag"] in NOUN | {"SN", "NR", "MM"}:
                parts.append(prefix + p["form"]); prefix = ""
            elif p["form"] in LINKS:
                parts.append(p["form"])
            i += 1
        return parts

    def korean(self, text, morphs, profile):
        values = []
        if "nominal" in self.args.term_types:
            for start in range(len(morphs)):
                for end in range(start+1, min(len(morphs), start+self.args.max_term_tokens)+1):
                    pieces = morphs[start:end]
                    if (end < len(morphs) and morphs[end]["tag"] in {"XSV", "XSA", "VCP"}
                            and morphs[end]["start"] <= pieces[-1]["end"]):
                        continue
                    if self.ko_np(pieces):
                        unit = self.unit(text, "ko", "nominal", pieces, profile, self.ko_parts(text, pieces))
                        if unit:
                            values.append(unit)
        if "verbal" in self.args.term_types:
            for i, p in enumerate(morphs):
                if p["tag"] not in {"VV", "XSV"}:
                    continue
                start, end = i, i+1
                if p["tag"] == "XSV":
                    while start and morphs[start-1]["tag"] in NOUN | {"XSN", "XPN"} and morphs[start-1]["end"] >= morphs[start]["start"]:
                        start -= 1
                    if start == i:
                        continue
                while end < len(morphs) and morphs[end]["tag"] in {"EP", "EF", "EC", "ETM"}:
                    end += 1
                base = [(q["form"], q["tag"]) for q in morphs[start:i+1]] + [("다", "EF")]
                if self.joiner is None:
                    from kiwipiepy import Kiwi
                    self.joiner = Kiwi(num_workers=1)
                canonical = self.joiner.join(base)
                unit = self.unit(text, "ko", "verbal", morphs[start:end], profile, [canonical])
                if unit:
                    values.append(unit)
        return self.deduplicate(values)

    def english(self, text, tokens, profile):
        values = []
        if "nominal" in self.args.term_types:
            for start in range(len(tokens)):
                for end in range(start+1, min(len(tokens),start+self.args.max_term_tokens)+1):
                    pieces = tokens[start:end]
                    if not pieces[-1]["tag"].startswith("NN"):
                        continue
                    if pieces[0]["tag"] in {"IN", "DT", "CC", "POS"} or pieces[0]["form"] in {"'s", "’s"}:
                        continue
                    allowed = True
                    for i, q in enumerate(pieces):
                        tag, form = q["tag"], q["form"].casefold()
                        if tag.startswith(("NN", "JJ")) or tag in {"CD", "VBG", "VBN"}:
                            continue
                        if 0 < i < len(pieces)-1 and ((tag=="IN" and form in {"of","for","in"}) or tag=="POS" or form in {"-","/","·"}):
                            continue
                        allowed = False; break
                    if not allowed:
                        continue
                    canonical = [self.lemma(norm(q["form"]), "n") if q["tag"] in {"NNS","NNPS"} else norm(q["form"]) for q in pieces]
                    unit = self.unit(text,"en","nominal",pieces,profile,canonical)
                    if unit:
                        values.append(unit)
        if "verbal" in self.args.term_types:
            for i, q in enumerate(tokens):
                if not q["tag"].startswith("VB") or norm(q["form"]) in {"be","am","is","are","was","were","been","being"}:
                    continue
                following = i + 1
                while following < len(tokens) and tokens[following]["tag"].startswith("RB"):
                    following += 1
                base = self.lemma(norm(q["form"]), "v")
                if following < len(tokens) and ((base == "have" and tokens[following]["tag"] == "VBN")
                                                or (base == "do" and tokens[following]["tag"] == "VB")):
                    continue
                end = i+2 if i+1 < len(tokens) and tokens[i+1]["tag"] == "RP" else i+1
                canonical = base
                if end > i+1:
                    canonical += " " + norm(tokens[i+1]["form"])
                unit = self.unit(text,"en","verbal",tokens[i:end],profile,[canonical])
                if unit:
                    values.append(unit)
        return self.deduplicate(values)

    @staticmethod
    def deduplicate(values):
        result = {(u["type"], *u["span"], u["canonical"]): u for u in values}
        return [result[key] for key in sorted(result)]
