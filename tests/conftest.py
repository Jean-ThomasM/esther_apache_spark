from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# S'assure que `src` est dans sys.path pour tous les tests,
# sans avoir à le faire dans chaque fichier de test.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: Any
) -> None:
    """
    Affiche un petit résumé lisible des tests de pipeline
    Pandas / PySpark à la fin de pytest.
    """
    # On récupère uniquement les tests de ce fichier
    pipeline_reports = []
    for outcome in ("passed", "failed", "error", "skipped"):
        for rep in terminalreporter.stats.get(outcome, []):
            # on ne garde que la phase "call" (pas setup/teardown)
            if getattr(rep, "when", "call") != "call":
                continue
            if "test_pipeline_equivalences.py" in rep.location[0]:
                pipeline_reports.append((rep, outcome))

    if not pipeline_reports:
        return

    tr = terminalreporter
    tr.section("Résumé pipelines Pandas / PySpark", sep="=")

    # Détail par test de ce fichier
    for rep, outcome in pipeline_reports:
        name = rep.nodeid.split("::")[-1]
        tr.line(f"- {name}: {outcome.upper()}")

    # Petit résumé global
    n_passed = sum(1 for _, o in pipeline_reports if o == "passed")
    n_failed = sum(1 for _, o in pipeline_reports if o in {"failed", "error"})

    if n_failed == 0:
        tr.line(f"=> Tous les tests de pipeline ont PASSÉ ({n_passed} ok).")
    else:
        tr.line(
            f"=> Attention : {n_failed} test(s) de pipeline en échec "
            f"sur {len(pipeline_reports)}."
        )
