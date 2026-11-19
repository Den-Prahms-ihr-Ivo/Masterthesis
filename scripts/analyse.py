"""Create the analysis CSV tables from stored experiment results.

Run from the project root, with src on the import path:
    python scripts/analyse.py results/main --out results/main/analysis

I use the saved schedules to calculate the tables again, without calling
the solver. TABLES below lists the ten output files. Empty tables still get
their headers. Figures are not produced by this script yet.
"""

import csv
import argparse

from fractions import Fraction
from pathlib import Path

from repair import analysis, metrics, storage

# file name -> what it holds; write_tables writes exactly these (CSV, one header row)
TABLES = {
    "step_table.csv": "metrics.step_rows over every RQ1 / robustness / undersupplied result",
    "sequence_table.csv": "analysis.sequence_table",
    "instance_table.csv": "analysis.instance_table (primary)",
    "instance_table_all.csv": "analysis.instance_table(primary=False), the secondary analysis",
    "rq2_table.csv": "analysis.rq2_table",
    "h1.csv": "analysis.evaluate_h1",
    "h2.csv": "analysis.evaluate_h2",
    "h3.csv": "analysis.evaluate_h3",
    "burden_agreement.csv": "analysis.burden_sensitivity()['agreement']",
    "burden_h2_cells.csv": "analysis.burden_sensitivity()['h2_cells']",
}


def write_tables(results: Path, out: Path) -> list[Path]:
    """Read the stored results once and write the tables listed in TABLES.

    Calculate all tables before writing them, so a calculation error does not
    leave a partially updated set. Create the output directory if needed and
    return the written paths in TABLES order.
    """
    stored = list(storage.read_results(results))

    # Per-sequence metrics feed both primary and secondary instance summaries.
    sequences = analysis.sequence_table(stored)
    instances = analysis.instance_table(sequences)
    instances_all = analysis.instance_table(sequences, primary=False)
    rq2 = analysis.rq2_table(stored)

    sensitivity = analysis.burden_sensitivity(instances)

    # Compute everything before writing, so a calculation failure does not
    # leave a mixture of newly generated and old tables.
    tables = {
        "sequence_table.csv": sequences,
        "instance_table.csv": instances,
        "instance_table_all.csv": instances_all,
        "rq2_table.csv": rq2,
        "h1.csv": analysis.evaluate_h1(instances),
        "h2.csv": analysis.evaluate_h2(instances),
        "h3.csv": analysis.evaluate_h3(rq2),
        "burden_agreement.csv": sensitivity["agreement"],
        "burden_h2_cells.csv": sensitivity["h2_cells"],
    }

    step_columns = (
        "cell",
        "size_class",
        "instance",
        "under_supplied",
        "profile",
        "sequence",
        *metrics.STEP_KEYS,
    )

    step_rows = []
    for result in stored:
        if result.cell == "rq2":
            continue

        profile = result.sequence.profile
        identifiers = {
            "cell": result.cell,
            "size_class": result.instance.meta.size_class,
            "instance": result.instance.meta.seed,
            "under_supplied": result.instance.meta.under_supplied,
            "profile": getattr(profile, "value", profile),
            "sequence": result.sequence.seed,
        }

        for row in metrics.step_rows(result):
            step_rows.append(
                {
                    **identifiers,
                    **{
                        key: float(value) if isinstance(value, Fraction) else value
                        for key, value in row.items()
                    },
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    written = []

    for filename in TABLES:
        path = out / filename

        if filename == "step_table.csv":
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=step_columns)
                writer.writeheader()
                writer.writerows(step_rows)
        else:
            tables[filename].to_csv(
                path,
                index=False,
                encoding="utf-8",
                na_rep="",
            )

        written.append(path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    write_tables(args.results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
