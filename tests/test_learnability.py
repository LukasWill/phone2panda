"""The learnability check: verdict rule, pairing of saved evaluations, and the checkpoint path (with a stand-in
for the simulator evaluation, so it runs without torch or LIBERO)."""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2p.policy import learnability as L  # noqa: E402
from p2p.stats import diff_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def summary(k, n, execution):
    return {"success": k, "n": n, "rate": f"{k}/{n}", "execution": execution,
            "episodes": [{"init": i, "success": i < k} for i in range(n)]}


def test_diff_ci_newcombe_example():
    lo, hi = diff_ci(56, 70, 48, 80)                 # Newcombe (1998), method 10: 0.0524 to 0.3339
    assert abs(lo - 0.0524) < 1e-3 and abs(hi - 0.3339) < 1e-3


def test_verdicts():
    assert L.verdict(97, 100, 50, 50)["verdict"] == "consistent"
    assert L.verdict(57, 100, 6, 50)["verdict"] == "conflicting moves"
    assert L.verdict(80, 100, 34, 50)["verdict"] == "unclear"          # keeps 85%
    assert L.verdict(10, 100, 1, 50)["verdict"] == "policy too weak to judge"
    r = L.verdict(64, 100, 24, 50)                                      # conflicting, but a 16-point drop
    assert r["verdict"] == "conflicting moves" and not r["drop_beyond_noise"]


def test_saved_results_reproduce_readme_table():
    rows = {r["run"]: r for d in sorted((ROOT / "results" / "act").glob("*/")) for r in L.from_results(d)}
    assert len(rows) == 12
    assert {r["verdict"] for k, r in rows.items() if "vertical" in k or k in ("official_s0", "ours_consistent_intent_s0")} \
        == {"consistent"}
    assert sum(r["verdict"] == "conflicting moves" for r in rows.values()) == 7
    assert rows["ours_consistent_clean_long_s0"]["normal_evals"] == ["eval_60k_a", "eval_60k_b"]


def test_notebook_layout(tmp_path):
    for name, (k, ex) in {"eval": (28, "n_action_steps=25"), "eval_rerun": (29, "n_action_steps=25"),
                          "eval_n10": (25, "n_action_steps=10"), "eval_te": (6, "temporal_ensemble=0.01")}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "eval.json").write_text(json.dumps(summary(k, 50, ex)))
    (r,) = L.from_results(tmp_path)
    assert sorted(r["normal_evals"]) == ["eval", "eval_rerun"] and r["normal"].startswith("57%")


def test_checkpoint_path(tmp_path, monkeypatch):
    calls = []

    def evaluate(ckpt, init_ids, device, video_dir=None, temporal_ensemble=None, **kw):
        calls.append(temporal_ensemble)
        n = len(list(init_ids))
        return summary(n if temporal_ensemble is None else n // 2, n,
                       f"temporal_ensemble={temporal_ensemble}" if temporal_ensemble else "n_action_steps=25")

    monkeypatch.setitem(sys.modules, "p2p.policy.act", types.SimpleNamespace(evaluate=evaluate))
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    L.main([str(ckpt), "--episodes", "20", "--device", "cpu"])
    out = tmp_path / "learnability"
    r = json.loads((out / "learnability.json").read_text())
    assert calls == [None, L.TE_COEFF] and r["verdict"] == "conflicting moves"
    assert (out / "eval.json").exists() and (out / "eval_te.json").exists()
