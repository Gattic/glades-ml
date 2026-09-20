#!/usr/bin/env python3
"""Build-tree and relocated install-tree public consumer/model-read regressions.
Run after a CPU library build. All outputs stay under glades-ml/build/.
"""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(tempfile.mkdtemp(prefix="package-consumer-", dir=ROOT / "build"))


def run(*args, cwd=RUN):
    result = subprocess.run([str(a) for a in args], cwd=cwd, text=True, capture_output=True,
                            env={**os.environ, "GLADES_MODEL_VERIFY_FILES": "0"})
    with (RUN / "commands.log").open("a") as log:
        log.write(f"{args!r} cwd={cwd}\n{result.stdout}{result.stderr}\nexit={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {args!r}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def prediction(output):
    return next(line for line in output.splitlines() if line.startswith("PRED "))


print(f"Package regression evidence: {RUN}", flush=True)
for kind in ("build", "install"):
    if kind == "build":
        package = ROOT / "build"
    else:
        stage = RUN / "stage"
        run("cmake", "--install", ROOT / "build", "--prefix", stage)
        relocated = RUN / "relocated-prefix"
        stage.rename(relocated)
        package = relocated / "share/glades/cmake"
        exports = "".join(p.read_text() for p in package.glob("*.cmake"))
        assert str(ROOT / "include") not in exports
        assert str(ROOT / "Backend") not in exports
    build = RUN / f"{kind}-consumer"
    run("cmake", "-S", ROOT / "tests/package-consumer", "-B", build, f"-Dglades_DIR={package}")
    run("cmake", "--build", build, "--parallel", "2")
    binary = build / "package_consumer"
    writer = RUN / f"{kind}-writer"
    writer.mkdir()
    expected = prediction(run(binary, "create", "fixture", cwd=writer))
    original = writer / "database/models/fixture"
    model = RUN / f"{kind}-relocated model"
    shutil.copytree(original, model)
    reader = RUN / f"{kind}-empty-read-only-cwd"
    reader.mkdir()
    before = snapshot(model)
    for path in model.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    model.chmod(0o555)
    reader.chmod(0o555)
    assert prediction(run(binary, "read", model, cwd=reader)) == expected
    assert prediction(run(binary, "read", str(model) + "/", cwd=reader)) == expected
    run(binary, "wrong-shape", model, cwd=reader)
    assert before == snapshot(model), "reader mutated model files"
    assert list(reader.iterdir()) == [], "reader created files (init/CWD dependency)"
    run(binary, "reject", "relative/model", cwd=reader)
    run(binary, "reject", RUN / "missing-package", cwd=reader)
    link = RUN / f"{kind}-symlink"
    link.symlink_to(model, target_is_directory=True)
    run(binary, "reject", link, cwd=reader)
    run(binary, "reject", str(link) + "/", cwd=reader)
    for case in ("manifest-missing", "weights-missing", "architecture-missing", "magic", "checksum",
                 "truncated", "metadata", "architecture-size", "file-symlink", "version", "net-type"):
        bad = RUN / f"{kind}-{case}"
        shutil.copytree(original, bad)
        manifest = bad / "manifest.txt"
        weights = bad / "weights.bin"
        if case.endswith("-missing"):
            (bad / {"manifest-missing": "manifest.txt", "weights-missing": "weights.bin",
                    "architecture-missing": "nninfo.csv"}[case]).unlink()
        elif case == "magic":
            manifest.write_text(manifest.read_text().replace("GLADES_MODEL", "BROKEN_MODEL"))
        elif case == "version":
            manifest.write_text(manifest.read_text().replace("version=3\n", "version=4\n"))
        elif case == "net-type":
            manifest.write_text(manifest.read_text().replace("netType=0\n", "netType=1\n"))
        elif case == "checksum":
            data = bytearray(weights.read_bytes()); data[-1] ^= 1; weights.write_bytes(data)
        elif case == "truncated":
            weights.write_bytes(weights.read_bytes()[:20])
        elif case == "metadata":
            manifest.write_text("\n".join(line for line in manifest.read_text().splitlines()
                                         if not line.startswith("weights.")) + "\n")
        elif case == "architecture-size":
            with (bad / "nninfo.csv").open("a") as out:
                out.write("\n")
        elif case == "file-symlink":
            weights.unlink(); weights.symlink_to(original / "weights.bin")
        run(binary, "reject", bad, cwd=reader)
    print(f"PASS {kind}: exported headers/link, relocation, read-only deterministic load, strict failures", flush=True)
print("PASS public package regression")
