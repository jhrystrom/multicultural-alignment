import json
import pickle
from pathlib import Path


def read_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def read_json(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]


def write_pickle(obj, file_path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def write_jsonl(data: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_list(file_path: Path) -> list[str]:
    return file_path.read_text(encoding="utf-8").split("\n")


def write_json(obj, file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def find_path(file_pattern: str, path: Path = Path(".")) -> Path:
    return next(path.glob(file_pattern))


def find_latest_path(file_pattern: str, directory: Path = Path(".")) -> Path:
    """Find the latest file in a directory that matches a pattern."""
    return max(directory.glob(file_pattern), key=lambda f: f.stat().st_mtime)


def append_to_file(content: list[str], file_path: Path) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        for line in content:
            f.write(line + "\n")
