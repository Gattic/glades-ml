#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$ROOT_DIR/scripts/benchqnnvsdff"
BUILD_DIR="$ROOT_DIR/build"
BENCH_BUILD_DIR="$BUILD_DIR/benchqnnvsdff"
BENCH_BIN="$BENCH_BUILD_DIR/bench_qnn_vs_dff"
DATA_DIR="$ROOT_DIR/unit-tests/datasets"

TASKS_CSV="xor,iris80,iris40"
MODELS_CSV=""
PREDICT_REPEATS_XOR=100
PREDICT_REPEATS_IRIS=50
SKIP_BUILD=0
NO_ARTIFACTS=0
DO_LIST=0
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR=""

usage() {
	cat <<EOF
Usage: $(basename "$0") [options]

Repeatable isolated-process benchmark comparing TYPE_QNN against TYPE_DFF on
XOR (20 seeds) and Iris (80/20 and 40/60 stratified splits, 5 splits x 5 seeds).
Each (task, model) runs in its own fresh OS process so peak RSS (ru_maxrss) and
live heap delta (mallinfo2) are measured without cross-model contamination.

Options:
  --tasks LIST                 Comma-separated tasks: xor,iris80,iris40 (default: xor,iris80,iris40)
  --models LIST                Comma-separated model IDs to run (default: all for selected tasks)
  --out-dir PATH               Directory for results.tsv and report.md
                               (default: artifacts/bench_qnn_vs_dff_<timestamp>)
  --no-artifacts               Do not save artifacts under artifacts/ (print tables to stdout only)
  --skip-build                 Skip building glades and the benchmark runner
  --predict-repeats-xor N      Full-split net.test() repeats for XOR latency (default: 100)
  --predict-repeats-iris N     Full-split net.test() repeats for Iris latency (default: 50)
  --list                       Build (unless --skip-build), list supported (task, model) pairs, and exit
  --help                       Show this message
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--tasks)
			TASKS_CSV="$2"
			shift 2
			;;
		--models)
			MODELS_CSV="$2"
			shift 2
			;;
		--out-dir)
			OUT_DIR="$2"
			shift 2
			;;
		--no-artifacts)
			NO_ARTIFACTS=1
			shift
			;;
		--skip-build)
			SKIP_BUILD=1
			shift
			;;
		--predict-repeats-xor)
			PREDICT_REPEATS_XOR="$2"
			shift 2
			;;
		--predict-repeats-iris)
			PREDICT_REPEATS_IRIS="$2"
			shift 2
			;;
		--list)
			DO_LIST=1
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

detect_cuda_flag() {
	if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && grep -q '^GLADES_ENABLE_CUDA:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"; then
		echo "ON"
	else
		echo "OFF"
	fi
}

build_harness() {
	local jobs
	jobs="$(nproc 2>/dev/null || echo 4)"

	if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
		echo "[benchqnnvsdff] Configuring glades in $BUILD_DIR..." >&2
		cmake -S "$ROOT_DIR" -B "$BUILD_DIR" >&2
	fi

	echo "[benchqnnvsdff] Building glades library..." >&2
	cmake --build "$BUILD_DIR" -j "$jobs" >&2

	local cuda_opt
	cuda_opt="$(detect_cuda_flag)"

	local need_configure=0
	if [[ ! -f "$BENCH_BUILD_DIR/CMakeCache.txt" ]]; then
		need_configure=1
	elif ! grep -q "^GLADES_ENABLE_CUDA:BOOL=${cuda_opt}$" "$BENCH_BUILD_DIR/CMakeCache.txt"; then
		need_configure=1
	fi

	if [[ "$need_configure" -eq 1 ]]; then
		echo "[benchqnnvsdff] Configuring benchmark worker (GLADES_ENABLE_CUDA=$cuda_opt)..." >&2
		cmake -S "$HARNESS_DIR" -B "$BENCH_BUILD_DIR" \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_PREFIX_PATH="$BUILD_DIR" \
			-DGLADES_ENABLE_CUDA="$cuda_opt" >&2
	fi

	echo "[benchqnnvsdff] Building benchmark worker..." >&2
	cmake --build "$BENCH_BUILD_DIR" -j "$jobs" >&2
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
	build_harness
elif [[ ! -x "$BENCH_BIN" ]]; then
	echo "ERROR: --skip-build was set but $BENCH_BIN does not exist." >&2
	exit 1
fi

if [[ "$DO_LIST" -eq 1 ]]; then
	"$BENCH_BIN" --list
	exit 0
fi

TMP_WORKDIR="$(mktemp -d)"
cleanup() {
	rm -rf "$TMP_WORKDIR"
}
trap cleanup EXIT

if [[ "$NO_ARTIFACTS" -eq 1 ]]; then
	RUN_OUT_DIR="$TMP_WORKDIR/out"
elif [[ -n "$OUT_DIR" ]]; then
	RUN_OUT_DIR="$OUT_DIR"
else
	RUN_OUT_DIR="$ROOT_DIR/artifacts/bench_qnn_vs_dff_${TIMESTAMP}"
fi
mkdir -p "$RUN_OUT_DIR"

TSV_FILE="$RUN_OUT_DIR/results.tsv"
REPORT_FILE="$RUN_OUT_DIR/report.md"

"$BENCH_BIN" --tsv-header > "$TSV_FILE"

csv_contains() {
	local csv="$1"
	local needle="$2"
	local IFS=','
	local item
	for item in $csv; do
		if [[ "$item" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

mapfile -t ALL_PAIRS < <("$BENCH_BIN" --list)
SELECTED_COUNT=0
for line in "${ALL_PAIRS[@]}"; do
	task="$(printf '%s' "$line" | cut -f1)"
	model_id="$(printf '%s' "$line" | cut -f2)"
	if ! csv_contains "$TASKS_CSV" "$task"; then
		continue
	fi
	if [[ -n "$MODELS_CSV" ]] && ! csv_contains "$MODELS_CSV" "$model_id"; then
		continue
	fi
	SELECTED_COUNT=$((SELECTED_COUNT + 1))
done

if [[ "$SELECTED_COUNT" -eq 0 ]]; then
	echo "ERROR: no (task, model) pairs matched --tasks='$TASKS_CSV' --models='$MODELS_CSV'." >&2
	exit 1
fi

IDX=0
for line in "${ALL_PAIRS[@]}"; do
	task="$(printf '%s' "$line" | cut -f1)"
	model_id="$(printf '%s' "$line" | cut -f2)"
	label="$(printf '%s' "$line" | cut -f3)"

	if ! csv_contains "$TASKS_CSV" "$task"; then
		continue
	fi
	if [[ -n "$MODELS_CSV" ]] && ! csv_contains "$MODELS_CSV" "$model_id"; then
		continue
	fi

	IDX=$((IDX + 1))
	repeats="$PREDICT_REPEATS_IRIS"
	if [[ "$task" == "xor" ]]; then
		repeats="$PREDICT_REPEATS_XOR"
	fi

	echo "[benchqnnvsdff] ($IDX/$SELECTED_COUNT) task=$task model=$model_id ($label)" >&2
	(
		cd "$TMP_WORKDIR"
		"$BENCH_BIN" \
			--task "$task" \
			--model "$model_id" \
			--data-dir "$DATA_DIR" \
			--predict-repeats "$repeats"
	) >> "$TSV_FILE"
done

render_markdown() {
	local tsv="$1"
	local git_rev
	git_rev="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
	local cuda_flag
	cuda_flag="$(detect_cuda_flag)"

	echo "# QNN (TYPE_QNN) vs. DFNN (TYPE_DFF) Isolated Benchmark Report"
	echo ""
	echo "- **Git HEAD**: \`$git_rev\`"
	echo "- **Build**: Release (\`-O3 -march=native\`, \`GLADES_ENABLE_CUDA=$cuda_flag\`)"
	echo "- **Isolation**: One fresh OS process per \`(task, model)\`; \`shmea::GLogger\` suppressed (\`LOG_FATAL\`, console off)"
	echo "- **Predict repeats**: XOR=\`${PREDICT_REPEATS_XOR}\`, Iris=\`${PREDICT_REPEATS_IRIS}\`"
	echo ""

	if awk -F'\t' 'NR>1 && $1=="xor" { found=1 } END { exit !found }' "$tsv"; then
		echo "## 1. XOR Gate (20 Seeds 100..119, Full-Batch batch=4)"
		echo ""
		echo "| Model | Params (Act/Stored) | Ep | Conv | Acc (%) | Ep->BCE<=0.17 mean (med) | Final BCE | F1 C0 (%) | F1 C1 (%) | Macro-F1 (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' 'NR>1 && $1=="xor" {
			printf "| %s | %s / %s | %s | %s/%s | %.2f +/- %.2f | %.1f (%.1f) | %.5f +/- %.5f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $6, $8, $7, $11, $12, $14, $15, $18, $19, $21, $22, $24, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi

	if awk -F'\t' 'NR>1 && $1=="iris80" { found=1 } END { exit !found }' "$tsv"; then
		echo "## 2. Iris 80/20 Split (120 Train / 30 Test, 5 Splits x 5 Seeds = 25 Runs, 300 Epochs)"
		echo ""
		echo "| Model | Params (Act/Stored) | Train Acc (%) | Test Acc (%) | Canon 42 (%) | Ep->95% / 98% | Train / Test CE | Brier | F1 Setosa (%) | F1 Versi (%) | F1 Virgi (%) | Macro-F1 (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' 'NR>1 && $1=="iris80" {
			printf "| %s | %s / %s | %.2f +/- %.2f | %.2f +/- %.2f | %.2f | %.1f / %.1f | %.4f / %.4f +/- %.4f | %.4f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $9, $10, $11, $12, $13, $14, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi

	if awk -F'\t' 'NR>1 && $1=="iris40" { found=1 } END { exit !found }' "$tsv"; then
		echo "## 3. Iris 40/60 Low-Data Split (60 Train / 90 Test, 5 Splits x 5 Seeds = 25 Runs, 300 Epochs)"
		echo ""
		echo "| Model | Params (Act/Stored) | Train Acc (%) | Test Acc (%) | Canon 42 (%) | Ep->95% / 98% | Train / Test CE | Brier | F1 Setosa (%) | F1 Versi (%) | F1 Virgi (%) | Macro-F1 (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' 'NR>1 && $1=="iris40" {
			printf "| %s | %s / %s | %.2f +/- %.2f | %.2f +/- %.2f | %.2f | %.1f / %.1f | %.4f / %.4f +/- %.4f | %.4f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $9, $10, $11, $12, $13, $14, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_markdown "$TSV_FILE" | tee "$REPORT_FILE"

if [[ "$NO_ARTIFACTS" -eq 0 ]]; then
	echo "[benchqnnvsdff] Saved raw TSV: $TSV_FILE" >&2
	echo "[benchqnnvsdff] Saved report : $REPORT_FILE" >&2
fi
