"""Result cache for slow volume analyses: memory LRU in front of gzip JSON files on disk.

A BraTS case or an uploaded volume analysed once with the same settings is returned
instantly afterwards, including after a server restart.
"""

import gzip
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


class ResultCache:
    def __init__(self, directory: Path, max_memory_items: int = 6, max_disk_mb: int = 500):
        self.dir = directory
        self.max_items = max_memory_items
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self._mem: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                return self._mem[key]
        path = self.dir / f"{key}.json.gz"
        if not path.exists():
            return None
        try:
            value = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        self._remember(key, value)
        return value

    def put(self, key: str, value: dict) -> None:
        self._remember(key, value)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f"{key}.tmp"
            tmp.write_bytes(gzip.compress(json.dumps(value).encode(), compresslevel=5))
            tmp.replace(self.dir / f"{key}.json.gz")
            self._trim_disk()
        except OSError:
            pass  # a cache write failure must never fail the request

    def _remember(self, key: str, value: dict) -> None:
        with self._lock:
            self._mem[key] = value
            self._mem.move_to_end(key)
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)

    def _trim_disk(self) -> None:
        files = sorted(self.dir.glob("*.json.gz"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > self.max_disk_bytes:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
