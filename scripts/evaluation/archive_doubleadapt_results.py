"""Archive recovered DoubleAdapt runs and local source snapshots without retraining."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/paper_external_baselines_20261106"
SUITE = ROOT / "experiments/csi300_aligned/models/doubleadapt_gru_90x10_suite_20261013"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_zip(path, originals):
    members = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for info in archive.infolist():
            with archive.open(info) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == originals[info.filename], info.filename
            members.append({"name": info.filename, "bytes": info.file_size, "sha256": h.hexdigest()})
    return {"file": str(path.relative_to(OUT)).replace(chr(92), "/"),
            "bytes": path.stat().st_size, "sha256": sha256(path), "members": members}


def main():
    target = OUT / "archives"
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in (100, 101, 102):
        original = SUITE / f"seed{seed}"
        files = sorted(p for p in original.rglob("*") if p.is_file())
        assert {p.name for p in files} >= {"predictions.csv", "task_membership.npz", "online_latest.pt", "test_initial.pt"}
        path = target / f"DoubleAdapt_seed{seed}.zip"
        hashes = {}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for p in files:
                name = str(p.relative_to(original)).replace(chr(92), "/")
                assert p.suffix in (".csv", ".npz", ".pt"), p
                archive.write(p, name)
                hashes[name] = sha256(p)
        entries.append(verify_zip(path, hashes))

    # Preserve both local code packages, plus the source-only subset of the
    # larger historical package. Archive paths make differing snapshots explicit.
    code_path = target / "DoubleAdapt_code_snapshots.zip"
    hashes, code_sources = {}, []
    with zipfile.ZipFile(code_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for package, prefix in (("doubleadapt_code_v2.zip", "v2"),
                                ("doubleadapt_code_final.zip", "final"),
                                ("doubleadapt_source.zip", "historical_source")):
            source = ROOT / ".runtime" / package
            code_sources.append({"source": ".runtime/" + package, "sha256": sha256(source)})
            with zipfile.ZipFile(source) as old:
                for name in old.namelist():
                    if name.endswith("/") or name.startswith("experiments/"):
                        continue
                    assert name.endswith((".py", ".txt", ".md", ".toml")), name
                    data = old.read(name)
                    key = prefix + "/" + name
                    archive.writestr(key, data)
                    hashes[key] = hashlib.sha256(data).hexdigest()
    entries.append(verify_zip(code_path, hashes))

    logs = OUT / "historical_verification"
    logs.mkdir(exist_ok=True)
    log_sources = []
    for p in sorted((SUITE / "verification").glob("*.log")):
        # Copy logs as text artifacts; never execute archived instructions.
        dest = logs / (p.stem + ".txt")
        shutil.copy2(p, dest)
        log_sources.append({"file": str(dest.relative_to(OUT)).replace(chr(92), "/"), "sha256": sha256(dest)})
    result = {"archive_date": "2026-11-06", "status": "all_member_hashes_verified",
              "archives": entries, "code_package_sources": code_sources, "historical_logs": log_sources,
              "excluded_duplicate": "_import/seed102 matches canonical seed102; not duplicated",
              "missing": "Per-run run_config.json and status.json absent. Source snapshots/defaults are not proof of actual training settings."}
    (OUT / "archive_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archives": len(entries), "members": sum(len(e["members"]) for e in entries),
                      "total_zip_bytes": sum(e["bytes"] for e in entries)}))


if __name__ == "__main__":
    main()
