"""Named tokenizer profiles independent of the BGE matching engine."""
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import re

from ..lab_common import digest
from ..lab_tokens import load_lexicon, merge_compounds
from .common import BACKENDS, original_tokens, validate_tokens
from . import english, korean  # Register built-ins without importing NLP packages.

DEFAULT_METHODS = ("C_bge", "C_nltk_treebank")
METHODS_FILE = Path(__file__).resolve().parents[2] / "tokenization_methods.json"


def load_methods(path=None):
    data = json.loads(METHODS_FILE.read_text(encoding="utf-8"))
    methods = data["methods"]
    if path:
        extra = json.loads(Path(path).read_text(encoding="utf-8"))
        methods.update(extra.get("methods", extra))
    for name, profile in methods.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name):
            raise ValueError(f"잘못된 방식 이름: {name}")
        if not isinstance(profile, dict) or any((lang, profile.get(lang)) not in BACKENDS for lang in ("en", "ko")):
            raise ValueError(f"지원하지 않는 토큰화 조합: {name}: {profile}")
    return methods


def method_signature(profile, lexicon):
    backend_info = {}
    for lang in ("en", "ko"):
        backend = BACKENDS[lang, profile[lang]]
        versions = {}
        for package in backend["packages"]:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        backend_info[lang] = {"name": profile[lang], "versions": versions,
                              "code": inspect.getsource(backend["function"])}
    if profile["ko"] == "kiwi_compound":
        backend_info["ko"]["kiwi_code"] = inspect.getsource(korean.kiwi_tokens)
        backend_info["ko"]["merge_code"] = inspect.getsource(merge_compounds)
        backend_info["ko"]["lexicon"] = sorted(lexicon)
    return digest([backend_info, inspect.getsource(original_tokens), inspect.getsource(validate_tokens)])


class TokenizerSuite:
    def __init__(self, args, resolved_revision=None):
        self.args = args
        self.lexicon = load_lexicon(args.compound_lexicon)
        self.resources, self.memo = {}, {}
        self.resolved_revision = resolved_revision

    def begin_row(self):
        self.memo.clear()

    def resource(self, name, factory):
        if name not in self.resources:
            self.resources[name] = factory()
        return self.resources[name]

    def tokens(self, text, language, backend):
        key = language, backend, text
        if key not in self.memo:
            values = BACKENDS[language, backend]["function"](self, text)
            validate_tokens(text, values)
            self.memo[key] = values
        return [dict(t, pos=list(t.get("pos", []))) for t in self.memo[key]]

    def bge_tokenizer(self):
        def make():
            from transformers import AutoConfig, AutoTokenizer
            options = {"cache_dir": str(self.args.model_cache), "local_files_only": self.args.local_files_only,
                       "revision": self.resolved_revision or self.args.bge_revision, "trust_remote_code": False}
            config = AutoConfig.from_pretrained(self.args.bge_model, **options)
            self.resolved_revision = getattr(config, "_commit_hash", None)
            if self.resolved_revision:
                options["revision"] = self.resolved_revision
            tokenizer = AutoTokenizer.from_pretrained(self.args.bge_model, use_fast=True, **options)
            if not tokenizer.is_fast:
                raise ValueError("원문 위치 보존을 위해 fast tokenizer가 필요합니다")
            return tokenizer
        return self.resource("bge_subword", make)
