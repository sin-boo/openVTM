"""Patch SDAnimePose.exe __main__ without a full PyInstaller rebuild."""

from __future__ import annotations

import importlib
import marshal
import shutil
import tempfile
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter

EXE = Path(r"E:\Ai-model\ai_vtuber\real_stream_SDAnime\dist\SDAnimePose\SDAnimePose.exe")
SRC = Path(r"E:\Ai-model\ai_vtuber\real_stream_SDAnime\backend\__main__.py")
SITE = Path(
    r"E:\Ai-model\ai_vtuber\pipeline\i1\torch_train\.venv\Lib\site-packages"
)

IDNA_PREFIX = """import codecs
def _sdanime_idna_encode(s, errors='strict'):
    return s.encode('ascii', errors), len(s)
def _sdanime_idna_decode(b, errors='strict'):
    return b.decode('ascii', errors), len(b)
codecs.register(
    lambda name: codecs.CodecInfo(
        name='idna', encode=_sdanime_idna_encode, decode=_sdanime_idna_decode
    )
    if name == 'idna'
    else None
)
"""

RTH_SOURCES = {
    "pyi_rth_pkgutil": SITE / "PyInstaller" / "hooks" / "rthooks" / "pyi_rth_pkgutil.py",
    "pyi_rth_multiprocessing": SITE / "PyInstaller" / "hooks" / "rthooks" / "pyi_rth_multiprocessing.py",
    "pyi_rth_setuptools": SITE / "PyInstaller" / "hooks" / "rthooks" / "pyi_rth_setuptools.py",
    "pyi_rth_pkgres": SITE / "PyInstaller" / "hooks" / "rthooks" / "pyi_rth_pkgres.py",
    "pyi_rth_inspect": SITE / "PyInstaller" / "hooks" / "rthooks" / "pyi_rth_inspect.py",
    "pyi_rth_tensorflow": SITE
    / "_pyinstaller_hooks_contrib"
    / "rthooks"
    / "pyi_rth_tensorflow.py",
}


def to_pyc(marshaled: bytes, dest: Path) -> None:
    """Wrap marshaled code object in a PEP 552-ish PYC header."""
    magic = importlib.util.MAGIC_NUMBER
    # flags=0, hash/timestamp=0, size=0
    header = magic + (0).to_bytes(4, "little") + (0).to_bytes(8, "little")
    dest.write_bytes(header + marshaled)


def main() -> None:
    raw = SRC.read_text(encoding="utf-8")
    for line in (
        "import encodings.idna  # noqa: F401\n",
        "import encodings.ascii  # noqa: F401\n",
        "import encodings.utf_8  # noqa: F401\n",
    ):
        raw = raw.replace(line, "")
    # Strip module docstring + future so we can rebuild a valid header.
    lines = raw.splitlines(keepends=True)
    i = 0
    docstring = ""
    if lines and lines[0].lstrip().startswith('"""'):
        docstring = lines[0]
        i = 1
        if '"""' not in lines[0][3:]:
            while i < len(lines):
                docstring += lines[i]
                if '"""' in lines[i]:
                    i += 1
                    break
                i += 1
        while i < len(lines) and lines[i].strip() == "":
            docstring += lines[i]
            i += 1
    future = ""
    if i < len(lines) and lines[i].startswith("from __future__"):
        future = lines[i]
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
    body = "".join(lines[i:])
    source = f'{docstring}{future}{IDNA_PREFIX}\n{body}'

    backup = EXE.with_suffix(".exe.bak")
    if not backup.is_file():
        shutil.copy2(EXE, backup)
        print(f"backup -> {backup}")
    else:
        # Always patch from backup so repeats are safe
        shutil.copy2(backup, EXE)
        print("restored from backup before patch")

    reader = CArchiveReader(str(EXE))
    archive_start = reader._start_offset  # noqa: SLF001
    bootloader = EXE.read_bytes()[:archive_start]
    print(f"bootloader={len(bootloader)} bytes, toc={list(reader.toc)}")

    tmp = Path(tempfile.mkdtemp(prefix="sdanime_patch_"))
    main_py = tmp / "__main__.py"
    main_py.write_text(source, encoding="utf-8")

    entries: list[tuple[str, str, bool, str]] = []
    for name, toc_entry in reader.toc.items():
        typecode = toc_entry[4]
        compress_flag = bool(toc_entry[3])
        data = reader.extract(name)

        if name == "__main__":
            entries.append((name, str(main_py), True, "s"))
            print("queued replacement __main__")
            continue

        if typecode in {"s", "s1", "s2"} and name in RTH_SOURCES and RTH_SOURCES[name].is_file():
            entries.append((name, str(RTH_SOURCES[name]), True, typecode))
            continue

        if typecode in {"s", "s1", "s2"}:
            # Fallback: reconstruct a temporary .py is impossible from marshal reliably;
            # wrap as .pyc and store as module-like — use pyc path with type m after header.
            pyc = tmp / f"{name}.pyc"
            to_pyc(data, pyc)
            # Bootloader script entries must stay type s. Last resort: write source that execs marshal.
            stub = tmp / f"{name}.py"
            stub.write_text(
                "import marshal, sys\n"
                f"exec(marshal.loads({data!r}), globals())\n",
                encoding="utf-8",
            )
            entries.append((name, str(stub), True, "s"))
            continue

        if typecode in {"m", "M"}:
            pyc = tmp / f"{name.replace('.', '_')}.pyc"
            to_pyc(data, pyc)
            entries.append((name, str(pyc), compress_flag, typecode))
            continue

        # PYZ and other binary blobs
        out = tmp / ("blob_" + name.replace("/", "_").replace("\\", "_").replace(".", "_"))
        out.write_bytes(data)
        entries.append((name, str(out), False if typecode == "z" else compress_flag, typecode))

    archive_path = tmp / "archive.bin"
    CArchiveWriter(str(archive_path), entries, pylib_name="python313.dll")
    archive = archive_path.read_bytes()
    print(f"new archive={len(archive)} bytes")

    patched = bootloader + archive
    verify_path = tmp / "verify.exe"
    verify_path.write_bytes(patched)
    verify = CArchiveReader(str(verify_path))
    assert "__main__" in verify.toc
    print("verify OK")

    EXE.write_bytes(patched)
    print(f"updated {EXE} ({EXE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
