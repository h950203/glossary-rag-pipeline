#!/usr/bin/env python3
"""Install the two small NLTK data resources used by the alignment-pair grouping pipeline."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RESOURCES = (("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
             ("wordnet", "corpora/wordnet.zip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "nltk_data")
    args = parser.parse_args()
    import nltk
    args.data_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(args.data_dir))
    for package, resource in RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, download_dir=str(args.data_dir), raise_on_error=True):
                raise RuntimeError(f"NLTK 자원 다운로드 실패: {package}")
        print(f"준비 완료: {package}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
