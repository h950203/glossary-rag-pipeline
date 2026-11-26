"""Timestamped run storage; internal artifacts remain under details/."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

RUN_NAME = re.compile(r"\d{6}_\d{6}\Z")


def new_run(root):
    root.mkdir(parents=True, exist_ok=True)
    # Concurrent starts in the same second wait for the next real timestamp.
    for _ in range(200):
        path = root / datetime.now().strftime("%y%m%d_%H%M%S")
        try:
            path.mkdir()
        except FileExistsError:
            time.sleep(0.025)
        else:
            (path / "details").mkdir()
            return path
    raise ValueError("결과 시각 폴더를 만들 수 없습니다. 잠시 후 재실행하세요")


def run_dirs(root):
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and RUN_NAME.fullmatch(p.name)),
                  key=lambda p: p.name, reverse=True)


def latest_run(root, complete=False):
    for path in run_dirs(root):
        meta = path / "details/run.json"
        if not meta.is_file():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        if not complete or data.get("status") == "complete":
            return path
    raise ValueError(f"{'완료된 ' if complete else ''}실행 결과가 없습니다: {root}")


def resolve_run(path):
    path = Path(path).expanduser().resolve()
    if path.name == "details":
        path = path.parent
    if not (path / "details/run.json").is_file():
        raise ValueError(f"v1.2 실행 폴더가 아닙니다: {path}")
    return path


def write_json(path, data):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


@contextmanager
def logged_run(path, project, resume=False):
    """Run-level lock covers engine, exports and publishing as one operation."""
    import fcntl
    details = path / "details"
    with (details / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"같은 실행 폴더에서 작업 중입니다: {path}") from exc
        meta_path = details / "run.json"
        meta = json.loads(meta_path.read_text()) if resume else {}
        started = time.monotonic()
        meta.update(project=project, status="running", result_dir=str(path),
                    local_started_at=datetime.now().astimezone().isoformat())
        write_json(meta_path, meta)
        with (details / "run.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"결과 폴더: {path}", flush=True)
                try:
                    yield meta
                    meta["status"] = "complete"
                    meta.pop("error", None)
                except BaseException as exc:
                    meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                error=f"{type(exc).__name__}: {exc}")
                    print(f"실행 미완료: {meta['error']}", file=sys.stderr, flush=True)
                    raise
                finally:
                    meta["last_elapsed_seconds"] = round(time.monotonic() - started, 3)
                    meta["local_finished_at"] = datetime.now().astimezone().isoformat()
                    write_json(meta_path, meta)
                    print(f"종료: {meta['status']}; {meta['last_elapsed_seconds']:.2f}초; {path}", flush=True)


def promote(source, target):
    """Move public artifacts, leaving no duplicate large files or symlinks."""
    if not source.is_file():
        raise ValueError(f"생성된 결과가 없습니다: {source}")
    os.replace(source, target)
