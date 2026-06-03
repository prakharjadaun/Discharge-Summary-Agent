from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)


class PDFCache:
    def __init__(self, cache_dir: str = ".pdf_cache"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, file_path: str) -> dict | None:
        sha = self._file_hash(file_path)
        cache_file = self._dir / f"{sha}.json"
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as e:
            _log.warning("Cache read failed for %s: %s", file_path, e)
            return None

    def put(self, file_path: str, pages: list[dict]) -> None:
        sha = self._file_hash(file_path)
        entry = {
            "sha256": sha,
            "path": file_path,
            "page_count": len(pages),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
        }
        cache_file = self._dir / f"{sha}.json"
        cache_file.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _file_hash(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
