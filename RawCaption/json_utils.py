#!/usr/bin/env python3
import json
from typing import Any, Dict, List


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: List[Dict[str, Any]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2) 