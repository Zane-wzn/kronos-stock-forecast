"""Export local HCA checkpoints and compact experiment evidence for migration."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from src.kronos_continual.hca_portable import load_model


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    models = ROOT / "experiments/csi300_aligned/models"
    source = models / "hca_online_full_20261020"
    out = ROOT / "models/hca_best"
    out.mkdir(parents=True, exist_ok=True)
    entries = {
        "prediction_static": models / "prediction_adapter/model.pt",
        "prediction_online_final": source / "streams/Prediction_A/checkpoints/A/2026-06-05.pt",
        "hca_pca_final": source / "streams/Prediction_HCA_PCA/checkpoints/A/2026-06-05.pt",
        "hca_random_final": source / "streams/Prediction_HCA_Random/checkpoints/A/2026-06-05.pt",
        "hca_frozen_head_final": models / "hca_gain_attribution_20261028/Prediction_Live_HCA_FrozenHead/checkpoints/A/2026-06-05.pt",
        "hca_dct_final": models / "prediction_dct_full_20261102/Prediction_Live_DCT/checkpoints/A/2026-06-05.pt",
    }
    assets = []
    for name, path in entries.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=(name == "prediction_static"))
        state = checkpoint if name == "prediction_static" else checkpoint["model"]
        target = out / f"{name}.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(target))
        restored = load_model(target)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items())
        assets.append({"name": name, "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                       "source_sha256": sha256(path), "file": target.name,
                       "sha256": sha256(target), "bytes": target.stat().st_size,
                       "tensor_values": sum(v.numel() for v in state.values()),
                       "version": None if name == "prediction_static" else checkpoint["version"],
                       "state_roundtrip": "bitwise_equal"})
    # Keep the two offline source checkpoints as original PyTorch files because
    # the historical runners verify their byte-level SHA256 identities.
    for name, path in (("prediction_static_original.pt", entries["prediction_static"]),
                       ("rankic_original.pt", ROOT / ".runtime/D_model_rankic_pilot_original.pt")):
        shutil.copy2(path, out / name)
    # Preserve exact stream resume state separately; journals and forecast history
    # remain prerequisites for the historical engine's directory-based resume.
    resume = source / "streams/Prediction_HCA_PCA/checkpoints/A/2026-06-05.pt"
    shutil.copy2(resume, out / "hca_pca_training_state.pt")
    basis = json.loads((source / "basis_manifest.json").read_text(encoding="utf-8"))
    write_json(out / "basis_manifest.json", basis)
    shutil.copy2(source / "run_config.json", out / "historical_run_config.json")
    write_json(out / "manifest.json", {
        "archive_date": "2026-11-05", "primary": "hca_pca_final.safetensors",
        "selection": "Retained HCA-PCA main configuration; retrospective record, not a validation-selected global optimum",
        "input": {"shape": [None, 512], "lookback": 90, "dtype": "float32",
                  "feature": "Frozen Kronos decode_s1 context[:, -1, :] before the trainable Adapter",
                  "backbone": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                  "locally_available_model_snapshot": "901c26c1332695a2a8f243eb2f37243a37bea320",
                  "locally_available_tokenizer_snapshot": "0e0117387f39004a9016484a186a908917e22426",
                  "historical_feature_extraction_revision": "not recorded; local snapshots alone do not establish historical revision",
                  "historical_hidden_sha256": "20e91f77ac595f119885a99f7a3c7f96672403879e9a85213bd5b9d55f349310"},
        "output": {"shape": [None, 10], "target": "close[o+k]/close[o]-1, k=1..10 trading days",
                   "path_signal": "mean over 10 horizons", "endpoint_signal": "horizon 10"},
        "online_state_after": "2026-06-05 after prediction and mature-label update; next prediction uses version216",
        "historical_evaluation": "2025-07-01/2026-06-05; 226 dates; 66103 keys; prequential, not a fixed final-checkpoint backtest",
        "training_state": {"file": "hca_pca_training_state.pt", "sha256": sha256(out / "hca_pca_training_state.pt"),
                           "contains": "model, optimizer, consumed labels, RNG, next_pos, version",
                           "resume_limit": "Historical runner also requires matching journals and prediction shards; not a standalone full replay archive"},
        "original_offline_checkpoints": [{"file": name, "sha256": sha256(out / name)}
                                         for name in ("prediction_static_original.pt", "rankic_original.pt")],
        "assets": assets})

    # Real-data checks: prequential last-day predictions use the PREVIOUS day's state.
    from scripts.experiments import run_hca_online_full as hca
    hidden = np.load(ROOT / "experiments/csi300_aligned/reference/hidden_cache/hidden_test.npy", mmap_mode="r")
    meta = pd.read_csv(ROOT / "experiments/csi300_aligned/doubleadapt_cache_v2/meta.csv", dtype={"stock": str})
    test = meta.loc[meta.split.eq("test")].reset_index(drop=True)
    day = test.loc[test.as_of_date.eq("2026-06-05")]
    x = torch.tensor(np.array(hidden[day.index]), dtype=torch.float32)
    reference = pd.read_csv(source / "streams/Prediction_HCA_PCA/predictions.csv", dtype={"stock": str})
    expected = day[["as_of_date", "stock"]].merge(reference, on=["as_of_date", "stock"], validate="one_to_one")
    previous = torch.load(source / "streams/Prediction_HCA_PCA/checkpoints/A/2026-06-04.pt", map_location="cpu", weights_only=False)
    replay = load_model(out / "hca_pca_final.safetensors")
    replay.load_state_dict(previous["model"])
    with torch.inference_mode():
        pred = replay(x).numpy()
    cols = [f"pred_return_{k}" for k in range(1, 11)]
    error = float(np.max(np.abs(pred - expected[cols].to_numpy())))
    assert error <= 1e-7, error
    write_json(out / "verification.json", {
        "status": "passed", "exported_models": len(assets),
        "roundtrip": "All exported state tensors bitwise equal original checkpoint tensors",
        "last_day_prequential_rows": len(day), "last_day_prequential_max_abs_error": error,
        "prediction_version": int(expected.pred_version.iloc[0]),
        "final_state_version": 216,
        "feature_archive_sha256": sha256(ROOT / "experiments/csi300_aligned/reference/hidden_cache/hidden_test.npy"),
        "checked_feature_sha_matches_historical": True})
    assert sha256(ROOT / "experiments/csi300_aligned/reference/hidden_cache/hidden_test.npy") == "20e91f77ac595f119885a99f7a3c7f96672403879e9a85213bd5b9d55f349310"

    # Allowlist compact top-level evidence, leaving large raw panels/caches local.
    families = ["hca_online_full_20261020", "hca_gain_attribution_20261028", "cfus_round1_20261022",
                "spectral_validation_20261028", "prediction_dct_full_20261102", "prediction_split_residual_20261103",
                "temporal_attention_pilot_60x80_20261104", "token_adapter_smoke_20261105", "lora_factorial_cpu_20261105"]
    evidence = ROOT / "docs/hca_archive_20261105/evidence"
    files = []
    for name in families:
        for p in sorted((models / name).iterdir()):
            if p.is_file() and p.suffix in (".md", ".json", ".csv", ".html", ".svg") and p.stat().st_size <= 500_000:
                target = evidence / name / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                files.append({"file": str(target.relative_to(evidence.parent)).replace("\\", "/"),
                              "source": str(p.relative_to(ROOT)).replace("\\", "/"),
                              "bytes": p.stat().st_size, "sha256": sha256(target)})
    for name in ("mars_cl_result_check_20261028", "prediction_progression_20261102"):
        source_dir = ROOT / "experiments/csi300_aligned/evaluations" / name
        for p in sorted(source_dir.iterdir()):
            if p.is_file() and p.suffix in (".md", ".json", ".csv") and p.stat().st_size <= 500_000:
                target = evidence / name / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                files.append({"file": str(target.relative_to(evidence.parent)).replace("\\", "/"),
                              "source": str(p.relative_to(ROOT)).replace("\\", "/"),
                              "bytes": p.stat().st_size, "sha256": sha256(target)})
    write_json(evidence.parent / "evidence_manifest.json", {"files": files, "rule": "Top-level textual evidence <=500000 bytes per file; no nested predictions, raw market data or caches"})
    print(json.dumps({"models": len(assets), "evidence_files": len(files), "prequential_error": error}))


if __name__ == "__main__":
    main()
