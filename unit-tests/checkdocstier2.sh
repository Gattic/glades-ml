#!/usr/bin/env bash
# Asserts that the Tier 2 tables in docs/qnn-vs-dff.md match the isolated
# benchmark JSON emitted by the QNN-vs-DFF harness.
#
# The harness emits one JSON object per (task, model) run:
#   bash scripts/benchqnnvsdff.sh --tasks tier2 --json --out-dir DIR
# produces DIR/results.json.
#
# Usage:
#   bash unit-tests/checkdocstier2.sh                 # use committed snapshot
#   bash unit-tests/checkdocstier2.sh --json PATH     # use a specific JSON file
#   bash unit-tests/checkdocstier2.sh --generate      # re-run Tier 2 benchmarks
#
# Exit codes: 0 = every documented value matches, 1 = mismatch, 2 = usage error.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_FILE="$ROOT_DIR/docs/qnn-vs-dff.md"
SNAPSHOT_FILE="$ROOT_DIR/unit-tests/data/qnn_tier2_benchmark.json"
HARNESS="$ROOT_DIR/scripts/benchqnnvsdff.sh"

JSON_FILE=""
GENERATE=0

usage() {
	cat <<EOF
Usage: $(basename "$0") [options]

Verifies that the Tier 2 (SO(3)-symmetric molecular & orbital) tables in
docs/qnn-vs-dff.md match the benchmark JSON output.

Options:
  --json PATH    Compare against PATH (JSON Lines, one object per run)
  --generate     Re-run the Tier 2 benchmarks via scripts/benchqnnvsdff.sh
                 and compare against the freshly generated JSON
  --help, -h     Show this message

Without --json/--generate the committed snapshot
(unit-tests/data/qnn_tier2_benchmark.json) is used.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--json)
			JSON_FILE="$2"
			shift 2
			;;
		--generate)
			GENERATE=1
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

if [[ ! -f "$DOC_FILE" ]]; then
	echo "ERROR: documentation file not found: $DOC_FILE" >&2
	exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
	echo "ERROR: python3 is required to parse the JSON benchmark output." >&2
	exit 2
fi

if [[ "$GENERATE" -eq 1 ]]; then
	if [[ -n "$JSON_FILE" ]]; then
		echo "ERROR: --generate and --json are mutually exclusive." >&2
		exit 2
	fi
	GEN_DIR="$(mktemp -d)"
	trap 'rm -rf "$GEN_DIR"' EXIT
	echo "[checkdocstier2] Regenerating Tier 2 benchmark JSON..." >&2
	bash "$HARNESS" --tasks tier2 --json --out-dir "$GEN_DIR" >/dev/null
	JSON_FILE="$GEN_DIR/results.json"
elif [[ -z "$JSON_FILE" ]]; then
	JSON_FILE="$SNAPSHOT_FILE"
fi

if [[ ! -f "$JSON_FILE" ]]; then
	echo "ERROR: benchmark JSON not found: $JSON_FILE" >&2
	echo "       Run: bash scripts/benchqnnvsdff.sh --tasks tier2 --json --out-dir artifacts/qnn_tier2" >&2
	exit 2
fi

python3 - "$DOC_FILE" "$JSON_FILE" <<'PY'
import json
import re
import sys

doc_path, json_path = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- benchmark JSON
records = {}
with open(json_path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records[(obj["task"], obj["model_id"])] = obj

if not records:
    sys.stderr.write("ERROR: benchmark JSON contains no records: %s\n" % json_path)
    sys.exit(1)

# --------------------------------------------------------------- model mapping
def model_key(text):
    """Map a doc column header or a benchmark model label to (family, optimizer)."""
    t = text.lower()
    if "so3-equiv" in t:
        family = "so3equiv"
    elif "qnn-std" in t and "polar" in t:
        family = "stdpolar"
    elif "qnn-std" in t and "split" in t:
        family = "stdsplit"
    elif "dff" in t and "width" in t:
        family = "dffwidth"
    elif "dff" in t and "param" in t:
        family = "dffparam"
    else:
        raise ValueError("unrecognised model column: %r" % text)
    if "adamw" in t:
        optimizer = "adamw"
    elif "sgd+m" in t or "sgdm" in t:
        optimizer = "sgdm"
    else:
        raise ValueError("unrecognised optimizer in column: %r" % text)
    return (family, optimizer)

json_by_key = {}
for (task, model_id), obj in records.items():
    json_by_key[(task, model_key(obj["label"]))] = obj

# --------------------------------------------------------------------- doc rows
def task_key(label):
    t = label.lower()
    if "h₂o" in t or "h2o" in t:
        family = "mol_pes"
    elif "slater" in t:
        family = "orbital_slater"
    elif "kepler" in t:
        family = "orbital_kepler"
    elif "chiral" in t:
        family = "mol_chiral"
    else:
        raise ValueError("unrecognised task row: %r" % label)
    if re.search(r"\bood\b", t):
        return family + "_ood"
    if re.search(r"\bid\b", t):
        return family + "_id"
    raise ValueError("task row lacks (ID)/(OOD) marker: %r" % label)

NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def parse_cell(cell, kind):
    """Extract the documented metrics from one markdown table cell."""
    cell = cell.replace("*", "").replace("`", "")
    if kind == "chiral":
        m = re.match(r"\s*(%s)\s*%%" % NUMBER, cell)
        if not m:
            raise ValueError("cannot parse accuracy in cell: %r" % cell)
        out = [("test_acc_mean", m.group(1))]
        ce = re.search(r"CE=(%s)" % NUMBER, cell)
        if not ce:
            raise ValueError("cannot parse CE in cell: %r" % cell)
        out.append(("test_loss_mean", ce.group(1)))
        inv = re.search(r"Inv=(%s)" % NUMBER, cell)
        if inv:
            out.append(("equiv_err_mean", inv.group(1)))
        return out

    m = re.match(r"\s*(%s)" % NUMBER, cell)
    if not m:
        raise ValueError("cannot parse RMSE in cell: %r" % cell)
    out = [("reg_rmse_mean", m.group(1))]

    aux = re.search(r"(?:E|MAE)=(%s)" % NUMBER, cell)
    if kind == "joint4":
        if aux:
            out.append(("test_brier_mean", aux.group(1)))
        ang = re.search(r"(%s)\s*°" % NUMBER, cell)
        if not ang:
            raise ValueError("cannot parse angular error in cell: %r" % cell)
        out.append(("reg_aux_mean", ang.group(1)))
    else:
        if not aux:
            raise ValueError("cannot parse MAE in cell: %r" % cell)
        out.append(("reg_aux_mean", aux.group(1)))

    r2 = re.search(r"R\^2\s*=\s*(%s)" % NUMBER, cell)
    if not r2:
        raise ValueError("cannot parse R^2 in cell: %r" % cell)
    out.append(("reg_r2_mean", r2.group(1)))
    return out

def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]

with open(doc_path, "r", encoding="utf-8") as fh:
    doc_lines = fh.read().splitlines()

sections = [
    ("#### Joint Scalar + 3D Vector & Scalar Invariant Regression", "joint4"),
    ("#### Molecular Chirality 3-Class", "chiral"),
]

def find_table(heading):
    for idx, line in enumerate(doc_lines):
        if line.strip().startswith(heading):
            start = None
            for j in range(idx + 1, len(doc_lines)):
                if doc_lines[j].lstrip().startswith("|"):
                    start = j
                    break
                if doc_lines[j].startswith("#"):
                    break
            if start is None:
                raise ValueError("no table found under heading: %s" % heading)
            rows = []
            for j in range(start, len(doc_lines)):
                if doc_lines[j].lstrip().startswith("|"):
                    rows.append(doc_lines[j])
                else:
                    break
            return rows
    raise ValueError("heading not found in %s: %s" % (doc_path, heading))

checked = 0
failures = []
covered = set()

for heading, kind in sections:
    rows = find_table(heading)
    header = split_row(rows[0])
    if header[0] != "Task" and header[0] != "Split":
        failures.append("%s: unexpected first header cell %r" % (heading, header[0]))
        continue
    column_keys = []
    for cell in header[1:]:
        try:
            column_keys.append(model_key(cell))
        except ValueError as exc:
            failures.append("%s: %s" % (heading, exc))
            column_keys.append(None)

    for row in rows[2:]:
        cells = split_row(row)
        try:
            task = task_key(cells[0])
        except ValueError as exc:
            failures.append("%s: %s" % (heading, exc))
            continue
        for column_key, cell in zip(column_keys, cells[1:]):
            if column_key is None:
                continue
            record = json_by_key.get((task, column_key))
            if record is None:
                failures.append(
                    "%s: no benchmark JSON record for task=%s model=%s"
                    % (heading, task, "/".join(column_key)))
                continue
            covered.add((task, column_key))
            row_kind = "slater" if "slater" in cells[0].lower() else kind
            try:
                documented = parse_cell(cell, row_kind)
            except ValueError as exc:
                failures.append("%s / %s: %s" % (heading, cells[0], exc))
                continue
            for field, literal in documented:
                actual = record[field]
                decimals = len(literal.split(".")[1]) if "." in literal else 0
                if "e" in literal.lower():
                    # Scientific notation in the doc: compare relatively.
                    expected = float(literal)
                    tolerance = abs(expected) * 0.006 + 1e-12
                else:
                    expected = float(literal)
                    tolerance = 0.5 * (10.0 ** -decimals) * 1.000001 + 1e-12
                checked += 1
                if abs(actual - expected) > tolerance:
                    failures.append(
                        "%s / %s / %s.%s: doc=%s benchmark=%.10g (tolerance %g)"
                        % (heading, cells[0], "/".join(column_key), field, literal, actual, tolerance))

# Every documented Tier 2 task must have at least one comparable record.
documented_tasks = {t for (t, _k) in covered}
missing_tasks = [t for t in ("mol_pes_id", "mol_pes_ood", "mol_chiral_id", "mol_chiral_ood",
                             "orbital_slater_id", "orbital_slater_ood",
                             "orbital_kepler_id", "orbital_kepler_ood") if t not in documented_tasks]
if missing_tasks:
    failures.append("documented Tier 2 tables do not cover tasks: %s" % ", ".join(missing_tasks))

print("=" * 96)
print("Tier 2 docs vs. benchmark JSON")
print("  doc     : %s" % doc_path)
print("  json    : %s" % json_path)
print("  records : %d benchmark runs, %d documented values checked" % (len(records), checked))
print("=" * 96)

if failures:
    print("FAIL: %d mismatch(es)" % len(failures))
    for failure in failures:
        print("  - %s" % failure)
    sys.exit(1)

print("PASS: every documented Tier 2 value matches the benchmark JSON output.")
sys.exit(0)
PY
