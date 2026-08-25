#!/bin/bash
# Generate final EALRMN_PHASE1_GPU_RESULTS.md from the sweep JSONL + analysis.
set -e
cd "$(dirname "$0")"

# If both prod_v1 and prod_bc JSONLs exist, concatenate them.
if [ -f results/sweep_prod_v1.jsonl ] && [ -f results/sweep_prod_bc.jsonl ]; then
    cat results/sweep_prod_v1.jsonl results/sweep_prod_bc.jsonl > results/sweep_all.jsonl
    JSONL=${1:-results/sweep_all.jsonl}
else
    JSONL=${1:-results/sweep_prod_v1.jsonl}
fi
OUT=${2:-../EALRMN_PHASE1_GPU_RESULTS.md}

if [ ! -f "$JSONL" ]; then
    echo "Missing $JSONL"
    exit 1
fi

echo "Aggregating $JSONL..."
./aggregate "$JSONL" > results/agg.txt
./md_table "$JSONL" > results/tables.md
echo "  agg: $(wc -l < results/agg.txt) lines"
echo "  md_table: $(wc -l < results/tables.md) lines"

# The doc structure has been pre-written; we substitute data tables.
# We don't auto-edit the doc — instead we cat the tables alongside the template.
echo ""
echo "=== AUTO-GENERATED TABLES (paste into EALRMN_PHASE1_GPU_RESULTS.md) ==="
cat results/tables.md
echo ""
echo "=== FULL AGGREGATE TABLE ==="
cat results/agg.txt
