#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple


CTX_PREFIXES = {
    "role": "CTX role:",
    "recall": "CTX recall:",
    "subtype": "CTX subtype:",
}

BUCKET_RE = re.compile(r"([A-Za-z0-9_-]+)=([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)")
CASE_RE = re.compile(r"^Case:\s+(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw atlas-alt-bench context-family logs and emit AdamW-vs-BiMAP "
            "per-bucket delta tables."
        )
    )
    parser.add_argument(
        "path",
        help="Artifact directory containing raw/*.log, or the raw directory itself.",
    )
    parser.add_argument(
        "--ref",
        default="adamw",
        help="Reference optimizer suffix in raw log filenames (default: adamw).",
    )
    parser.add_argument(
        "--cand",
        default="bimap_lite",
        help="Candidate optimizer suffix in raw log filenames (default: bimap_lite).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output file prefix. Defaults to ctx_delta_<cand>_vs_<ref> in the artifact dir.",
    )
    return parser.parse_args()


def resolve_dirs(path_str: str) -> Tuple[pathlib.Path, pathlib.Path]:
    path = pathlib.Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    if path.is_dir() and (path / "raw").is_dir():
        return path, path / "raw"
    if path.is_dir() and path.name == "raw":
        return path.parent, path
    raise FileNotFoundError(f"expected artifact dir with raw/ or raw dir, got: {path}")


def detect_benchmark(text: str, fallback_name: str) -> str:
    for line in text.splitlines():
        match = CASE_RE.match(line.strip())
        if match:
            return match.group(1)
    return fallback_name


def parse_ctx_lines(text: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    parsed: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for section, prefix in CTX_PREFIXES.items():
            if not stripped.startswith(prefix):
                continue
            buckets: Dict[str, Tuple[float, float]] = {}
            for bucket, nll, share in BUCKET_RE.findall(stripped):
                buckets[bucket] = (float(nll), float(share))
            if buckets:
                parsed[section] = buckets
    return parsed


def match_optimizer_stem(stem: str, optimizer_suffix: str) -> Optional[str]:
    suffix = f"_{optimizer_suffix}"
    if stem.endswith(suffix):
        return ""
    epoch_match = re.search(rf"{re.escape(suffix)}_(e\d+)$", stem)
    if epoch_match:
        return epoch_match.group(1)
    return None


def benchmark_key(benchmark: str, run_tag: str) -> str:
    return benchmark if not run_tag else f"{benchmark}/{run_tag}"


def read_logs(raw_dir: pathlib.Path, optimizer_suffix: str) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    result: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for logfile in sorted(raw_dir.glob("*.log")):
        run_tag = match_optimizer_stem(logfile.stem, optimizer_suffix)
        if run_tag is None:
            continue
        text = logfile.read_text(encoding="utf-8")
        benchmark = detect_benchmark(text, logfile.stem)
        sections = parse_ctx_lines(text)
        if sections:
            result[benchmark_key(benchmark, run_tag)] = sections
    return result


def format_float(value: float) -> str:
    return f"{value:.5f}"


def write_tsv(
    path: pathlib.Path,
    ref_name: str,
    cand_name: str,
    rows: List[Tuple[str, str, str, float, float, float, float, float, float]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "benchmark\tsection\tbucket\t"
            f"{ref_name}_nll\t{cand_name}_nll\tdelta_nll\t"
            f"{ref_name}_share\t{cand_name}_share\tdelta_share\n"
        )
        for row in rows:
            benchmark, section, bucket, ref_nll, cand_nll, delta_nll, ref_share, cand_share, delta_share = row
            handle.write(
                "\t".join(
                    [
                        benchmark,
                        section,
                        bucket,
                        format_float(ref_nll),
                        format_float(cand_nll),
                        format_float(delta_nll),
                        format_float(ref_share),
                        format_float(cand_share),
                        format_float(delta_share),
                    ]
                )
                + "\n"
            )


def write_md(
    path: pathlib.Path,
    ref_name: str,
    cand_name: str,
    rows: List[Tuple[str, str, str, float, float, float, float, float, float]],
) -> None:
    grouped: Dict[Tuple[str, str], List[Tuple[str, float, float, float, float, float, float]]] = {}
    for row in rows:
        benchmark, section, bucket, ref_nll, cand_nll, delta_nll, ref_share, cand_share, delta_share = row
        grouped.setdefault((benchmark, section), []).append(
            (bucket, ref_nll, cand_nll, delta_nll, ref_share, cand_share, delta_share)
        )

    with path.open("w", encoding="utf-8") as handle:
        for (benchmark, section) in sorted(grouped.keys()):
            handle.write(f"## {benchmark} / {section}\n\n")
            handle.write(
                f"| Bucket | {ref_name} NLL | {cand_name} NLL | Delta NLL | "
                f"{ref_name} Share | {cand_name} Share | Delta Share |\n"
            )
            handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for bucket, ref_nll, cand_nll, delta_nll, ref_share, cand_share, delta_share in sorted(
                grouped[(benchmark, section)], key=lambda item: item[3]
            ):
                handle.write(
                    f"| {bucket} | {ref_nll:.5f} | {cand_nll:.5f} | {delta_nll:.5f} | "
                    f"{ref_share:.3f} | {cand_share:.3f} | {delta_share:.3f} |\n"
                )
            handle.write("\n")


def print_summary(
    ref_name: str,
    cand_name: str,
    rows: List[Tuple[str, str, str, float, float, float, float, float, float]],
) -> None:
    grouped: Dict[Tuple[str, str], List[Tuple[str, float, float, float]]] = {}
    for row in rows:
        benchmark, section, bucket, ref_nll, cand_nll, delta_nll, _, _, _ = row
        grouped.setdefault((benchmark, section), []).append((bucket, ref_nll, cand_nll, delta_nll))

    for (benchmark, section) in sorted(grouped.keys()):
        print(f"{benchmark} / {section}")
        for bucket, ref_nll, cand_nll, delta_nll in sorted(grouped[(benchmark, section)], key=lambda item: item[3]):
            print(
                f"  {bucket:>10s}  {ref_name}={ref_nll:.5f}  {cand_name}={cand_nll:.5f}  "
                f"delta={delta_nll:.5f}"
            )
        print()


def main() -> int:
    args = parse_args()
    artifact_dir, raw_dir = resolve_dirs(args.path)
    ref_logs = read_logs(raw_dir, args.ref)
    cand_logs = read_logs(raw_dir, args.cand)

    shared_benchmarks = sorted(set(ref_logs.keys()) & set(cand_logs.keys()))
    if not shared_benchmarks:
        print(
            f"no shared benchmarks with CTX rows found for ref={args.ref} cand={args.cand} in {raw_dir}",
            file=sys.stderr,
        )
        return 2

    prefix = args.out_prefix
    if prefix is None:
        prefix = f"ctx_delta_{args.cand}_vs_{args.ref}"
    out_prefix = artifact_dir / prefix

    rows: List[Tuple[str, str, str, float, float, float, float, float, float]] = []
    for benchmark in shared_benchmarks:
        ref_sections = ref_logs[benchmark]
        cand_sections = cand_logs[benchmark]
        for section in sorted(set(ref_sections.keys()) & set(cand_sections.keys())):
            ref_buckets = ref_sections[section]
            cand_buckets = cand_sections[section]
            for bucket in sorted(set(ref_buckets.keys()) & set(cand_buckets.keys())):
                ref_nll, ref_share = ref_buckets[bucket]
                cand_nll, cand_share = cand_buckets[bucket]
                rows.append(
                    (
                        benchmark,
                        section,
                        bucket,
                        ref_nll,
                        cand_nll,
                        cand_nll - ref_nll,
                        ref_share,
                        cand_share,
                        cand_share - ref_share,
                    )
                )

    tsv_path = pathlib.Path(f"{out_prefix}.tsv")
    md_path = pathlib.Path(f"{out_prefix}.md")
    write_tsv(tsv_path, args.ref, args.cand, rows)
    write_md(md_path, args.ref, args.cand, rows)
    print_summary(args.ref, args.cand, rows)
    print(f"Saved delta TSV: {tsv_path}")
    print(f"Saved delta Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
