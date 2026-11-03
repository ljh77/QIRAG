"""Resolve historical input paths to packaged filenames."""
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PATH_MAP = json.loads(
    (PACKAGE_ROOT / "metadata" / "runtime_path_mapping.json").read_text(
        encoding="utf-8"
    )
)

def resolve(value):
    """Resolve a known packaged input; leave unknown values unchanged."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    raw = os.fspath(value)
    if not raw:
        return value
    p = Path(raw)
    if p.is_absolute():
        try:
            key = p.relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            return value
    else:
        key = p.as_posix()
        if key.startswith("./"):
            key = key[2:]
    mapped = PATH_MAP.get(key)
    if mapped is None:
        return value
    result = PACKAGE_ROOT / mapped
    return str(result) if isinstance(value, str) else result

def listdir(value):
    """Expose historical filenames for a mapped input directory."""
    p = Path(value)
    try:
        key = p.relative_to(PACKAGE_ROOT).as_posix() if p.is_absolute() else p.as_posix()
    except ValueError:
        return os.listdir(value)
    entries = {
        Path(old).name for old in PATH_MAP
        if Path(old).parent.as_posix() == key
    }
    if p.is_dir():
        entries.update(os.listdir(p))
    if entries:
        return sorted(entries)
    return os.listdir(value)

def isdir(value):
    """Recognize directories represented by mapped input files."""
    p = Path(value)
    try:
        key = p.relative_to(PACKAGE_ROOT).as_posix() if p.is_absolute() else p.as_posix()
    except ValueError:
        return os.path.isdir(value)
    return p.is_dir() or any(
        old.startswith(key.rstrip("/") + "/") for old in PATH_MAP
    )


# Publication inputs are immutable; new outputs belong to runtime_outputs.
def output_path(value):
    """Route writes under the package to a separate runtime directory."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    raw = os.fspath(value)
    p = Path(raw)
    absolute = p if p.is_absolute() else Path.cwd() / p
    absolute = absolute.resolve()
    try:
        rel = absolute.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return value

    if rel.startswith("runtime_outputs/"):
        result = absolute
    else:
        reverse = {target: source for source, target in PATH_MAP.items()}
        historical = reverse.get(rel, rel)
        result = PACKAGE_ROOT / "runtime_outputs" / historical
    result.parent.mkdir(parents=True, exist_ok=True)
    return str(result) if isinstance(value, str) else result

def open_path(value, mode="r"):
    """Separate write destinations from archived input paths."""
    if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
        return output_path(value)
    return resolve(value)

def _archive_guard(event, args):
    """Reject direct writes or replacement of published snapshot files."""
    protected = ("data", "results", "metadata", "dataset_sample", "scripts")

    def is_protected(value):
        if not isinstance(value, (str, os.PathLike)):
            return False
        try:
            rel = Path(value).resolve().relative_to(PACKAGE_ROOT)
        except (ValueError, OSError):
            return False
        return bool(rel.parts) and rel.parts[0] in protected

    if event == "open":
        path, mode, flags = args
        writing = (
            isinstance(mode, str)
            and any(flag in mode for flag in ("w", "a", "x", "+"))
        )
        if isinstance(flags, int):
            writing = writing or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        if writing and is_protected(path):
            raise PermissionError(
                "Published snapshots are read-only; use runtime_outputs: "
                + str(path)
            )
    elif event in ("os.rename", "os.remove", "os.rmdir"):
        targets = args[:2] if event == "os.rename" else args[:1]
        if any(is_protected(p) for p in targets):
            raise PermissionError("Cannot replace or delete published snapshots.")

import sys as _sys
_sys.addaudithook(_archive_guard)
