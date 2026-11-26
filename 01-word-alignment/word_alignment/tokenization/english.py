"""English segmentation only; no stemming/lemmatization that changes surfaces."""
from .common import original_tokens, register, subwords


@register("en", "spacy", ("spacy",))
def spacy_tokens(suite, text):
    def make():
        import spacy
        return spacy.blank("en")
    tokenizer = suite.resource("spacy", make)
    return original_tokens(text, [(t.idx, t.idx + len(t.text), [])
                                  for t in tokenizer(text) if not t.is_space])


@register("en", "nltk_treebank", ("nltk",))
def treebank_tokens(suite, text):
    def make():
        from nltk.tokenize import TreebankWordTokenizer
        return TreebankWordTokenizer()
    tokenizer = suite.resource("nltk_treebank", make)
    return original_tokens(text, [(a, b, []) for a, b in tokenizer.span_tokenize(text)])


@register("en", "nltk_wordpunct", ("nltk",))
def wordpunct_tokens(suite, text):
    def make():
        from nltk.tokenize import WordPunctTokenizer
        return WordPunctTokenizer()
    tokenizer = suite.resource("nltk_wordpunct", make)
    return original_tokens(text, [(a, b, []) for a, b in tokenizer.span_tokenize(text)])


register("en", "bge_subword", ("transformers", "tokenizers"))(subwords)
