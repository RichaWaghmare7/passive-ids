"""
Replay a pcap file or a labeled flow-level CSV (CIC-IDS2017 / UNSW-NB15
style) through PassiveIDSPipeline.

Two distinct replay modes exercise two distinct parts of the pipeline,
deliberately:

  --source pcap   raw packets -> pipeline.on_packet()
                  exercises capture/ -> flow_state/ -> features/ -> emission/

  --source csv    pre-extracted flow records -> pipeline.on_feature_vector_ready()
                  exercises models/ -> calibration/ -> alerting/ directly,
                  since public flow-level datasets cannot be replayed as raw
                  packets -- they've already been through someone else's
                  feature extraction.

Every downstream tier this touches is currently a stub (see the TODOs
left across the package), so this script is *expected* to report
NotImplementedError today. That's intentional: it's the harness those
TODOs get implemented against, not a finished demo. It reports exactly
where execution stops so progress is visible module by module as each
piece lands.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from passive_ids.pipeline import PassiveIDSPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset column mappings: public dataset column name -> internal feature key.
# Extend as more formats are supported. Ground-truth label columns are
# handled separately and never included in the feature vector passed to
# the pipeline -- that would be label leakage into what's supposed to be
# an unsupervised/production-shaped input.
# ---------------------------------------------------------------------------
DATASET_FORMATS: dict[str, dict[str, Any]] = {
    "cicids2017": {
        "label_column": "Label",
        "feature_map": {
            "Flow Duration": "flow_duration",
            "Total Fwd Packets": "fwd_packets",
            "Total Backward Packets": "bwd_packets",
            "Total Length of Fwd Packets": "fwd_bytes",
            "Total Length of Bwd Packets": "bwd_bytes",
            "Flow Bytes/s": "bytes_per_sec",
            "Flow Packets/s": "packets_per_sec",
            "Fwd Packet Length Mean": "fwd_pkt_len_mean",
            "Bwd Packet Length Mean": "bwd_pkt_len_mean",
            "Flow IAT Mean": "iat_mean",
            "Flow IAT Std": "iat_std",
            "SYN Flag Count": "syn_flag_count",
            "Destination Port": "dst_port",
        },
    },
    "unsw-nb15": {
        "label_column": "label",
        "feature_map": {
            "dur": "flow_duration",
            "spkts": "fwd_packets",
            "dpkts": "bwd_packets",
            "sbytes": "fwd_bytes",
            "dbytes": "bwd_bytes",
            "rate": "packets_per_sec",
            "sttl": "src_ttl",
            "dttl": "dst_ttl",
            "proto": "protocol",
            "service": "service",
            "state": "tcp_state",
        },
    },
}


def load_config(config_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in ("thresholds.yaml", "sla_matrix.yaml"):
        path = config_dir / name
        if path.exists():
            with open(path) as f:
                config[name.split(".")[0]] = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# PCAP replay -- exercises capture/ -> flow_state/ -> features/ -> emission/
# ---------------------------------------------------------------------------
def replay_pcap(
    path: Path, speed: str, limit: int | None
) -> Iterator[tuple[bytes, float]]:
    """Yield (raw_bytes, inter_packet_delay_seconds) in capture order.

    Uses scapy's PcapReader for streaming rather than rdpcap, so this
    doesn't load an entire multi-GB capture into memory at once --
    consistent with the streaming constraint the rest of this project
    is built around.
    """
    from scapy.utils import PcapReader

    prev_ts = None
    count = 0
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if limit is not None and count >= limit:
                return
            ts = float(packet.time)
            delay = 0.0
            if speed == "realtime" and prev_ts is not None:
                delay = max(0.0, ts - prev_ts)
            prev_ts = ts
            yield bytes(packet), delay
            count += 1


# ---------------------------------------------------------------------------
# CSV replay -- exercises models/ -> calibration/ -> alerting/ directly.
# ---------------------------------------------------------------------------
def replay_csv(
    path: Path, dataset_format: str, limit: int | None
) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yield (feature_vector, ground_truth_label) per row.

    ground_truth_label is kept separate from feature_vector -- never
    passed into the pipeline as a feature. Returned here so the caller
    can eventually compare pipeline output against it once alerting/
    is implemented; today it's just tallied for the summary.
    """
    spec = DATASET_FORMATS.get(dataset_format)
    if spec is None:
        raise ValueError(
            f"Unknown dataset_format '{dataset_format}'. "
            f"Known formats: {list(DATASET_FORMATS)}"
        )
    feature_map: dict[str, str] = spec["feature_map"]
    label_column: str = spec["label_column"]

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # CIC-IDS2017 CSVs commonly ship with leading whitespace in headers.
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for i, raw_row in enumerate(reader):
            if limit is not None and i >= limit:
                return
            row = {k.strip(): v for k, v in raw_row.items() if k is not None}
            feature_vector: dict[str, Any] = {}
            for src_col, dst_key in feature_map.items():
                if src_col in row:
                    feature_vector[dst_key] = _coerce_numeric(row[src_col])
            label = row.get(label_column)
            yield feature_vector, label


def _coerce_numeric(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / "config")
    pipeline = PassiveIDSPipeline(config)

    outcomes: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    start = time.monotonic()
    processed = 0

    if args.source == "pcap":
        for raw_bytes, delay in replay_pcap(Path(args.path), args.speed, args.limit):
            if delay:
                time.sleep(delay / args.speedup)
            try:
                pipeline.on_packet(raw_bytes)
                outcomes["processed_ok"] += 1
            except NotImplementedError:
                outcomes["not_yet_implemented"] += 1
            processed += 1
            if processed % 1000 == 0:
                _report_progress(processed, start)

    elif args.source == "csv":
        for feature_vector, label in replay_csv(
            Path(args.path), args.dataset_format, args.limit
        ):
            if label:
                label_counts[label] += 1
            try:
                pipeline.on_feature_vector_ready(feature_vector)
                outcomes["processed_ok"] += 1
            except NotImplementedError:
                outcomes["not_yet_implemented"] += 1
            processed += 1
            if processed % 1000 == 0:
                _report_progress(processed, start)

    else:
        raise ValueError(f"Unknown --source '{args.source}'")

    elapsed = time.monotonic() - start
    print("\n=== replay summary ===")
    print(f"source:              {args.source} ({args.path})")
    print(f"records processed:   {processed}")
    if elapsed > 0:
        print(f"elapsed:             {elapsed:.2f}s ({processed / elapsed:.1f} records/sec)")
    print(f"pipeline outcomes:   {dict(outcomes)}")
    if label_counts:
        print(f"ground-truth labels: {dict(label_counts)}")
    if outcomes["not_yet_implemented"]:
        print(
            "\nNote: not_yet_implemented counts above are expected until the "
            "corresponding pipeline tier is built -- see README.md build order."
        )


def _report_progress(processed: int, start: float) -> None:
    elapsed = time.monotonic() - start
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(f"  ...{processed} records ({rate:.0f}/sec)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["pcap", "csv"], required=True)
    parser.add_argument("--path", required=True, help="Path to .pcap or dataset .csv")
    parser.add_argument(
        "--dataset-format",
        choices=list(DATASET_FORMATS),
        default="cicids2017",
        help="Column mapping to use for --source csv",
    )
    parser.add_argument(
        "--speed",
        choices=["max", "realtime"],
        default="max",
        help="'max' replays as fast as possible; 'realtime' paces pcap "
        "replay using original packet timestamps",
    )
    parser.add_argument(
        "--speedup", type=float, default=1.0, help="Divide realtime delay by this factor"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N records (quick smoke tests)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
