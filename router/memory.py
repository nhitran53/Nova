"""Local-date, query-only daily context shared across voice sessions."""
from datetime import datetime
from pathlib import Path


class DailyMemory:
    def __init__(self, directory=None, now=datetime.now):
        self.directory = Path(directory) if directory is not None else Path(__file__).parent / "memory"
        self.now = now

    def path(self):
        return self.directory / f"{self.now().date().isoformat()}.md"

    def append(self, query):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path().open("a", encoding="utf-8") as stream:
            stream.write(query + "\n")

    def load(self):
        path = self.path()
        return path.read_text(encoding="utf-8") if path.exists() else ""
