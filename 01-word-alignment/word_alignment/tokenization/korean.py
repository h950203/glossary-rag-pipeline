"""Korean segmentation strategies; each keeps exact original character spans."""
import re

from ..lab_tokens import merge_compounds
from .common import original_tokens, register, subwords


@register("ko", "kiwi_morph", ("kiwipiepy",))
def kiwi_tokens(suite, text):
    def make():
        from kiwipiepy import Kiwi
        return Kiwi(num_workers=suite.args.threads)
    tokenizer = suite.resource("kiwi", make)
    parsed = list(tokenizer.tokenize(text))
    result = original_tokens(text, [(t.start, t.start + t.len, [t.tag]) for t in parsed])
    for token in result:
        token["morphs"] = [{"form": t.form, "tag": t.tag.split("-")[0],
                            "start": t.start, "end": t.start + t.len}
                           for t in parsed if t.start < token["end"] and token["start"] < t.start + t.len]
    return result


@register("ko", "kiwi_compound", ("kiwipiepy",))
def compound_tokens(suite, text):
    return merge_compounds(text, suite.tokens(text, "ko", "kiwi_morph"), suite.lexicon)


@register("ko", "eojeol")
def eojeol_tokens(suite, text):
    return [{"text": m.group(), "start": m.start(), "end": m.end(), "pos": []}
            for m in re.finditer(r"\S+", text)]


register("ko", "bge_subword", ("transformers", "tokenizers"))(subwords)
