#!/bin/bash
# Run after prod_v1 + prod_bc sweep completes.
# Aggregates results, generates tables, updates the final results doc.
set -e
cd "$(dirname "$0")"

# Combine JSONL
if [ -f results/sweep_prod_v1.jsonl ] && [ -f results/sweep_prod_bc.jsonl ]; then
    cat results/sweep_prod_v1.jsonl results/sweep_prod_bc.jsonl > results/sweep_all.jsonl
    JSONL=results/sweep_all.jsonl
    echo "Combined Phase A + Phase B+C: $(wc -l < $JSONL) rows"
elif [ -f results/sweep_prod_v1.jsonl ]; then
    JSONL=results/sweep_prod_v1.jsonl
    echo "Phase A only: $(wc -l < $JSONL) rows"
else
    echo "No sweep data found"
    exit 1
fi

# Run aggregator
./aggregate "$JSONL" | tee results/final_agg.txt

# Run md_table
./md_table "$JSONL" | tee results/final_tables.md

# Summary of complete runs
echo ""
echo "=== COMPLETE RUNS ==="
echo "Phase A T=2048 (step 800):"
grep '"T":2048' "$JSONL" | grep '"step":800' | wc -l
echo "Phase B T=4096 (step 500):"
grep '"T":4096' "$JSONL" | grep '"step":500' | wc -l
echo "Phase C T=16384 (step 200 or 300):"
grep '"T":16384' "$JSONL" | grep -E '"step":(200|300)' | wc -l

echo ""
echo "Final aggregate at: results/final_agg.txt"
echo "Final tables at:    results/final_tables.md"
echo "Combined JSONL at:  $JSONL"
