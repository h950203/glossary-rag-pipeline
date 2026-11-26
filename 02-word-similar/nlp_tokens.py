"""Rules loading only; all linguistic analysis is supplied by word-alignment v1.1."""
import json
from pathlib import Path

def load_rules(path=None):
    source=Path(path) if path else Path(__file__).resolve().parent/'rules.json'
    data=json.loads(source.read_text(encoding='utf-8-sig'))
    if data.get('schema_version') != 1:
        raise ValueError('rules schema_version=1이 필요합니다')
    return data
