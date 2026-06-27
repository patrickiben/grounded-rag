from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    id: str
    title: str
    text: str


def load_corpus(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        Document(id=d["id"], title=d.get("title", ""), text=d["text"])
        for d in data["documents"]
    ]
