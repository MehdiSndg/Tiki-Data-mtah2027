"""The experiment queue gates on complete summaries, never on a guessed finish time."""

import importlib.util
import json
from pathlib import Path


def _runner():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_short_team_screen.py"
    spec = importlib.util.spec_from_file_location("short_team_screen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_partial_and_complete_summary_gates(tmp_path, monkeypatch):
    mod = _runner()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "OUTPUT_ROOT", tmp_path / "outputs")
    assert mod.finished_summary("toy") is None
    path = tmp_path / "outputs" / "toy" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"dfl_lomo":', encoding="utf-8")
    assert mod.finished_summary("toy") is None
    path.write_text(json.dumps({"dfl_lomo": [{}], "dfl_lomo_eval": [{}], "metrica_eval": [{}] * 2}),
                    encoding="utf-8")
    assert mod.finished_summary("toy") is None
    path.write_text(json.dumps({"dfl_lomo": [{}], "dfl_lomo_eval": [{}], "metrica_eval": [{}] * 3}),
                    encoding="utf-8")
    assert mod.finished_summary("toy") is not None


def test_gpu_guard_rejects_utilization_and_visible_python_process(monkeypatch):
    mod = _runner()

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    for utilization, processes, expected in (("80", "", False), ("1", "python.exe", False),
                                              ("1", "", True)):
        def query(cmd, **_):
            return Result(utilization if "--query-gpu=utilization.gpu" in cmd else processes)
        monkeypatch.setattr(mod.subprocess, "run", query)
        assert mod.gpu_idle() is expected
