#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="$ROOT_DIR/scripts/benchqnnvsdff"
BUILD_DIR="$ROOT_DIR/build"
BENCH_BUILD_DIR="$BUILD_DIR/benchqnnvsdff"
BENCH_BIN="$BENCH_BUILD_DIR/bench_qnn_vs_dff"
DATA_DIR="$ROOT_DIR/unit-tests/datasets"
REPO_DATA_DIR="$ROOT_DIR/datasets"

TASKS_CSV="xor,iris80,iris40"
MODELS_CSV=""
PREDICT_REPEATS_XOR=100
PREDICT_REPEATS_IRIS=50
PREDICT_REPEATS_REG=20
SKIP_BUILD=0
NO_ARTIFACTS=0
JSON_MODE=0
DO_LIST=0
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR=""

usage() {
	cat <<EOF
Usage: $(basename "$0") [options]

Repeatable isolated-process benchmark comparing TYPE_QNN (standard & SO(3)-equivariant)
against TYPE_DFF across XOR, Iris (80/20 & 40/60), the Tier 1 suite (SO(3) 3D rotation,
SO(3) composition, 3/4-bit parity, Dow Jones OHLC, Wine), and the Tier 2 SO(3)-symmetric
molecular & orbital suite (H2O PES + force, chiral enantiomers, Slater-Koster p-p overlap,
and 3D Keplerian orbital energy + Laplace-Runge-Lenz eccentricity vector).
Each (task, model) runs in its own fresh OS process so peak RSS (ru_maxrss) and
live heap delta (mallinfo2) are measured without cross-model contamination.

Options:
  --tasks LIST                 Comma-separated tasks or aliases (default: xor,iris80,iris40)
                               Tasks  : xor, iris80, iris40, so3_rot_curve, so3_rot_ood,
                                        so3_comp_id, so3_comp_ood, xorparity3, parity3_odd,
                                        parity4, dow_ohlc, wine,
                                        mol_pes_id, mol_pes_ood, mol_chiral_id, mol_chiral_ood,
                                        orbital_slater_id, orbital_slater_ood,
                                        orbital_kepler_id, orbital_kepler_ood
                               Aliases: tier1, tier2, all, so3_comp, parity3,
                                        mol_pes, mol_chiral, orbital_slater, orbital_kepler
  --models LIST                Comma-separated model IDs to run (default: all for selected tasks)
  --out-dir PATH               Directory for results.tsv and report.md
                               (default: artifacts/bench_qnn_vs_dff_<timestamp>)
  --json                       Emit one JSON object per (task, model) run into
                               results.json instead of results.tsv/report.md
  --no-artifacts               Do not save artifacts under artifacts/ (print tables to stdout only)
  --skip-build                 Skip building glades and the benchmark runner
  --predict-repeats-xor N      Full-split net.test() repeats for truth-table tasks (default: 100)
  --predict-repeats-iris N     Full-split net.test() repeats for tabular/chiral tasks (default: 50)
  --predict-repeats-reg N      Full-split net.test() repeats for SO(3)/Dow/Tier-2 reg (default: 20)
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
		--json)
			JSON_MODE=1
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
		--predict-repeats-reg)
			PREDICT_REPEATS_REG="$2"
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

expand_task_aliases() {
	local raw="$1"
	local out=""
	local IFS=','
	local item expanded
	for item in $raw; do
		case "$item" in
			tier1)
				expanded="so3_rot_curve,so3_rot_ood,so3_comp_id,so3_comp_ood,xorparity3,parity3_odd,parity4,dow_ohlc,wine"
				;;
			tier2)
				expanded="mol_pes_id,mol_pes_ood,mol_chiral_id,mol_chiral_ood,orbital_slater_id,orbital_slater_ood,orbital_kepler_id,orbital_kepler_ood"
				;;
			all)
				expanded="xor,iris80,iris40,so3_rot_curve,so3_rot_ood,so3_comp_id,so3_comp_ood,xorparity3,parity3_odd,parity4,dow_ohlc,wine,mol_pes_id,mol_pes_ood,mol_chiral_id,mol_chiral_ood,orbital_slater_id,orbital_slater_ood,orbital_kepler_id,orbital_kepler_ood"
				;;
			so3_comp)
				expanded="so3_comp_id,so3_comp_ood"
				;;
			parity3)
				expanded="xorparity3,parity3_odd"
				;;
			mol_pes)
				expanded="mol_pes_id,mol_pes_ood"
				;;
			mol_chiral)
				expanded="mol_chiral_id,mol_chiral_ood"
				;;
			orbital_slater)
				expanded="orbital_slater_id,orbital_slater_ood"
				;;
			orbital_kepler)
				expanded="orbital_kepler_id,orbital_kepler_ood"
				;;
			*)
				expanded="$item"
				;;
		esac
		if [[ -z "$out" ]]; then
			out="$expanded"
		else
			out="${out},${expanded}"
		fi
	done
	printf '%s' "$out"
}

TASKS_CSV="$(expand_task_aliases "$TASKS_CSV")"

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
	if [[ ! -f "$BENCH_BUILD_DIR/CMakeCache.txt" ]] || [[ ! -f "$BENCH_BUILD_DIR/Makefile" ]]; then
		need_configure=1
	elif ! grep -q "^GLADES_ENABLE_CUDA:BOOL=${cuda_opt}$" "$BENCH_BUILD_DIR/CMakeCache.txt"; then
		need_configure=1
	fi

	if [[ "$need_configure" -eq 1 ]]; then
		local extra_cmake_args=()
		if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
			local cuda_compiler cuda_archs
			cuda_compiler="$(grep -E '^CMAKE_CUDA_COMPILER:' "$BUILD_DIR/CMakeCache.txt" | head -n1 | cut -d= -f2- || true)"
			cuda_archs="$(grep -E '^CMAKE_CUDA_ARCHITECTURES:' "$BUILD_DIR/CMakeCache.txt" | head -n1 | cut -d= -f2- || true)"
			if [[ -n "$cuda_compiler" ]]; then
				extra_cmake_args+=("-DCMAKE_CUDA_COMPILER=$cuda_compiler")
			fi
			if [[ -n "$cuda_archs" ]]; then
				extra_cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$cuda_archs")
			fi
		fi
		echo "[benchqnnvsdff] Configuring benchmark worker (GLADES_ENABLE_CUDA=$cuda_opt)..." >&2
		cmake --fresh -S "$HARNESS_DIR" -B "$BENCH_BUILD_DIR" \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_PREFIX_PATH="$BUILD_DIR" \
			-DGLADES_ENABLE_CUDA="$cuda_opt" \
			"${extra_cmake_args[@]}" >&2
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

if [[ "$JSON_MODE" -eq 1 ]]; then
	TSV_FILE="$RUN_OUT_DIR/results.json"
	REPORT_FILE=""
	: > "$TSV_FILE"
else
	TSV_FILE="$RUN_OUT_DIR/results.tsv"
	REPORT_FILE="$RUN_OUT_DIR/report.md"
	"$BENCH_BIN" --tsv-header > "$TSV_FILE"
fi

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
	case "$task" in
		xor|xorparity3|parity3_odd|parity4)
			repeats="$PREDICT_REPEATS_XOR"
			;;
		so3_rot_curve|so3_rot_ood|so3_comp_id|so3_comp_ood|dow_ohlc|mol_pes_id|mol_pes_ood|orbital_slater_id|orbital_slater_ood|orbital_kepler_id|orbital_kepler_ood)
			repeats="$PREDICT_REPEATS_REG"
			;;
	esac

	echo "[benchqnnvsdff] ($IDX/$SELECTED_COUNT) task=$task model=$model_id ($label)" >&2
	(
		cd "$TMP_WORKDIR"
		json_flag=()
		if [[ "$JSON_MODE" -eq 1 ]]; then
			json_flag=(--json)
		fi
		"$BENCH_BIN" \
			--task "$task" \
			--model "$model_id" \
			--data-dir "$DATA_DIR" \
			--repo-data-dir "$REPO_DATA_DIR" \
			--predict-repeats "$repeats" \
			"${json_flag[@]}"
	) >> "$TSV_FILE"
done

render_truth_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | Ep | Conv | Acc (%) | Ep->BCE<=0.17 mean (med) | Final BCE | F1 C0 (%) | F1 C1 (%) | Macro-F1 (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %s | %s/%s | %.2f +/- %.2f | %.1f (%.1f) | %.5f +/- %.5f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $6, $8, $7, $11, $12, $14, $15, $18, $19, $21, $22, $24, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_cls3_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	local c0="$4"
	local c1="$5"
	local c2="$6"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | Train Acc (%) | Test Acc (%) | Canon 42 (%) | Ep->95% / 98% | Train / Test CE | Brier | F1 ${c0} (%) | F1 ${c1} (%) | F1 ${c2} (%) | Macro-F1 (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.2f +/- %.2f | %.2f +/- %.2f | %.2f | %.1f / %.1f | %.4f / %.4f +/- %.4f | %.4f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $9, $10, $11, $12, $13, $14, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_chiral_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | Train Acc (%) | Test Acc (%) | Ep->95% / 98% | Test CE | Brier | F1 Planar (%) | F1 R (%) | F1 S (%) | Macro-F1 (%) | Invar Err (max) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.2f +/- %.2f | %.2f +/- %.2f | %.1f / %.1f | %.4f +/- %.4f | %.4f | %.2f | %.2f | %.2f | %.2f +/- %.2f | %.2e | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $9, $10, $11, $12, $14, $16, $18, $19, $20, $21, $22, $23, $24, $25, $39, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_so3_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	local rmse_col="$4"
	local ang_col="$5"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | ${rmse_col} | ${ang_col} | Test R2 | Equiv Err (max) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.4f +/- %.4f | %.2f +/- %.2f | %.4f | %.2e | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $34, $35, $36, $37, $38, $39, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_tier2_joint_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	local scalar_col="$4"
	local ang_col="$5"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | 4D RMSE | ${scalar_col} | ${ang_col} | Test R2 | Equiv Err (max) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.4f +/- %.4f | %.4f | %.2f +/- %.2f | %.4f | %.2e | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $34, $35, $20, $36, $37, $38, $39, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_tier2_scalar_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | Test RMSE | Test MAE | Test R2 | Invar Err (max) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.4f +/- %.4f | %.4f +/- %.4f | %.4f | %.2e | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $34, $35, $36, $37, $38, $39, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

render_dow_table() {
	local tsv="$1"
	local task_id="$2"
	local heading="$3"
	if awk -F'\t' -v t="$task_id" 'NR>1 && $1==t { found=1 } END { exit !found }' "$tsv"; then
		echo "$heading"
		echo ""
		echo "| Model | Params (Act/Stored) | Test RMSE (sigma) | Test RMSE (USD) | Test MAE (sigma) | Test R2 | Dir Acc (%) | Train (ms) | Predict us/split (us/smp) | Peak Heap Delta (KiB) | Core / w.bin (B) | Peak RSS (KiB) |"
		echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
		awk -F'\t' -v t="$task_id" 'NR>1 && $1==t {
			printf "| %s | %s / %s | %.4f +/- %.4f | $%.2f +/- $%.2f | %.4f | %.4f | %.2f +/- %.2f | %.2f +/- %.2f | %.2f (%.2f) | %.2f | %s / %s | %s |\n",
				$3, $4, $5, $34, $35, $36, $37, $20, $38, $11, $12, $26, $27, $28, $29, $30, $31, $32, $33
		}' "$tsv"
		echo ""
	fi
}

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
	echo "- **Predict repeats**: Truth-table=\`${PREDICT_REPEATS_XOR}\`, Tabular/Chiral=\`${PREDICT_REPEATS_IRIS}\`, SO(3)/Dow/Tier-2 Reg=\`${PREDICT_REPEATS_REG}\`"
	echo ""

	render_truth_table "$tsv" "xor" "## 1. XOR Gate (20 Seeds 100..119, Full-Batch batch=4)"
	render_cls3_table  "$tsv" "iris80" "## 2. Iris 80/20 Split (120 Train / 30 Test, 5 Splits x 5 Seeds = 25 Runs, 300 Epochs)" "Setosa" "Versi" "Virgi"
	render_cls3_table  "$tsv" "iris40" "## 3. Iris 40/60 Low-Data Split (60 Train / 90 Test, 5 Splits x 5 Seeds = 25 Runs, 300 Epochs)" "Setosa" "Versi" "Virgi"

	render_so3_table "$tsv" "so3_rot_curve" "## 4. Tier 1A: SO(3) 3D Rotation Sample-Efficiency Curve (N in {32, 64, 128, 256, 512}, N_test=256, 5 Seeds, 400 Epochs)" "3D RMSE" "Ang Err (deg)"
	render_so3_table "$tsv" "so3_rot_ood"   "## 5. Tier 1B: SO(3) 3D Rotation OOD (Opposite Hemisphere vz<-0.25 + Unseen Angle [pi/2, pi], N_train=128, N_test=256, 5 Seeds)" "3D RMSE" "Ang Err (deg)"
	render_so3_table "$tsv" "so3_comp_id"   "## 6. Tier 1C-ID: SO(3) Quaternion Composition q1 (x) q2 In-Distribution (N_train=128, N_test=256, 5 Seeds, 400 Epochs)" "4D RMSE" "Geod Err (deg)"
	render_so3_table "$tsv" "so3_comp_ood"  "## 7. Tier 1C-OOD: SO(3) Quaternion Composition q1 (x) q2 OOD Rotated Frame (N_train=128, N_test=256, 5 Seeds, 400 Epochs)" "4D RMSE" "Geod Err (deg)"

	render_truth_table "$tsv" "xorparity3"  "## 8. Tier 1D: In-Repo datasets/xorparity3.csv (8 Rows, Exact-One-Bit sum==1, 20 Seeds, batch=8)"
	render_truth_table "$tsv" "parity3_odd" "## 9. Tier 1E: Synthetic 3-Bit Odd Parity (8 Rows, 20 Seeds, batch=8)"
	render_truth_table "$tsv" "parity4"     "## 10. Tier 1F: Synthetic 4-Bit Odd Parity (16 Rows, 20 Seeds, batch=16)"

	render_dow_table   "$tsv" "dow_ohlc" "## 11. Tier 1G: In-Repo Dow Jones Weekly OHLC 4-Tuple Regression (360 Q1 Train -> 390 Q2 Test, 5 Seeds, 200 Epochs)"
	render_cls3_table  "$tsv" "wine"     "## 12. Tier 1H: In-Repo Wine Classification (13 Features, 3 Classes, 80/20 Stratified, 5 Splits x 5 Seeds = 25 Runs)" "C1" "C2" "C3"

	render_tier2_joint_table  "$tsv" "mol_pes_id"         "## 13. Tier 2A-ID: Triatomic H2O Molecular PES + Central-Atom Force In-Distribution (N_train=192, N_test=256, 5 Seeds, 400 Epochs)" "Energy RMSE" "Force Ang (deg)"
	render_tier2_joint_table  "$tsv" "mol_pes_ood"        "## 14. Tier 2A-OOD: Triatomic H2O Molecular PES + Central-Atom Force OOD Orientation (Train [0,30 deg] -> Test [60,180 deg], N_train=192, N_test=256, 5 Seeds)" "Energy RMSE" "Force Ang (deg)"
	render_chiral_table       "$tsv" "mol_chiral_id"      "## 15. Tier 2B-ID: Chiral Molecular Center 3-Class (Planar / R / S Enantiomer) In-Distribution (N_train=180, N_test=240, 5 Seeds, 300 Epochs)"
	render_chiral_table       "$tsv" "mol_chiral_ood"     "## 16. Tier 2B-OOD: Chiral Molecular Center 3-Class (Planar / R / S Enantiomer) OOD Orientation (Train [0,35 deg] -> Test [60,180 deg], N_train=180, N_test=240, 5 Seeds)"
	render_tier2_scalar_table "$tsv" "orbital_slater_id"  "## 17. Tier 2C-ID: Quantum Slater-Koster p-p Molecular Orbital Overlap In-Distribution (N_train=192, N_test=256, 5 Seeds, 400 Epochs)"
	render_tier2_scalar_table "$tsv" "orbital_slater_ood" "## 18. Tier 2C-OOD: Quantum Slater-Koster p-p Molecular Orbital Overlap OOD Hemisphere (Train r_hat.z>0.65 -> Test r_hat.z<-0.35, N_train=192, N_test=256, 5 Seeds)"
	render_tier2_joint_table  "$tsv" "orbital_kepler_id"  "## 19. Tier 2D-ID: 3D Keplerian Orbital Energy + Laplace-Runge-Lenz Eccentricity Vector In-Distribution (N_train=192, N_test=256, 5 Seeds, 400 Epochs)" "Energy RMSE" "LRL Ang (deg)"
	render_tier2_joint_table  "$tsv" "orbital_kepler_ood" "## 20. Tier 2D-OOD: 3D Keplerian Orbital Energy + Laplace-Runge-Lenz Eccentricity Vector OOD Orbital Plane (Train i in [0,25 deg] -> Test i in [65,180 deg], N_train=192, N_test=256, 5 Seeds)" "Energy RMSE" "LRL Ang (deg)"
}

if [[ "$JSON_MODE" -eq 1 ]]; then
	if [[ "$NO_ARTIFACTS" -eq 0 ]]; then
		echo "[benchqnnvsdff] Saved raw JSON: $TSV_FILE" >&2
	fi
else
	render_markdown "$TSV_FILE" | tee "$REPORT_FILE"

	if [[ "$NO_ARTIFACTS" -eq 0 ]]; then
		echo "[benchqnnvsdff] Saved raw TSV: $TSV_FILE" >&2
		echo "[benchqnnvsdff] Saved report : $REPORT_FILE" >&2
	fi
fi
