from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_PATH = CARE_ROOT / "data" / "pair_merged_data" / "all_pair_data.tsv"
DEFAULT_PROTEIN_PATH = CARE_ROOT / "data" / "proteins_afdb.tsv"
DEFAULT_OUTPUT_PATH = CARE_ROOT / "data" / "pair_merged_data" / "rxn_ec_number.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rxn_id to EC number mapping from pair and protein data."
    )
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--protein-path", type=Path, default=DEFAULT_PROTEIN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def split_ec_numbers(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(";") if item.strip()]


def load_enzyme_ec_numbers(protein_path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with protein_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            enz_id = (row.get("Entry") or "").strip()
            if not enz_id:
                continue
            ec_numbers = split_ec_numbers((row.get("EC number") or "").strip())
            if ec_numbers:
                mapping[enz_id] = ec_numbers
    return mapping


def choose_ec_number(counts: Counter[str], first_seen: dict[str, int]) -> str:
    best_ec = ""
    best_count = -1
    best_order = float("inf")
    for ec_number, count in counts.items():
        order = first_seen[ec_number]
        if count > best_count or (count == best_count and order < best_order):
            best_ec = ec_number
            best_count = count
            best_order = order
    return best_ec


def build_rxn_ec_numbers(
    pair_path: Path, enzyme_to_ecs: dict[str, list[str]]
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    rxn_order: list[str] = []
    rxn_seen: set[str] = set()
    rxn_counts: dict[str, Counter[str]] = {}
    rxn_first_seen: dict[str, dict[str, int]] = {}
    stats = {
        "pair_rows": 0,
        "unique_rxn_ids": 0,
        "pair_rows_with_ec": 0,
        "pair_rows_without_ec": 0,
        "expanded_rxn_enz_ec_rows": 0,
        "rxn_ids_without_ec": 0,
    }

    global_order = 0
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stats["pair_rows"] += 1
            rxn_id = (row.get("rxn_id") or "").strip()
            enz_id = (row.get("enz_id") or "").strip()
            if not rxn_id:
                continue
            if rxn_id not in rxn_seen:
                rxn_seen.add(rxn_id)
                rxn_order.append(rxn_id)
            ec_numbers = enzyme_to_ecs.get(enz_id, [])
            if not ec_numbers:
                stats["pair_rows_without_ec"] += 1
                continue
            stats["pair_rows_with_ec"] += 1
            counts = rxn_counts.setdefault(rxn_id, Counter())
            first_seen = rxn_first_seen.setdefault(rxn_id, {})
            for ec_number in ec_numbers:
                counts[ec_number] += 1
                if ec_number not in first_seen:
                    first_seen[ec_number] = global_order
                global_order += 1
                stats["expanded_rxn_enz_ec_rows"] += 1

    stats["unique_rxn_ids"] = len(rxn_order)
    rxn_to_ec = {}
    for rxn_id in rxn_order:
        counts = rxn_counts.get(rxn_id)
        if counts:
            rxn_to_ec[rxn_id] = choose_ec_number(counts, rxn_first_seen[rxn_id])
        else:
            rxn_to_ec[rxn_id] = ""
            stats["rxn_ids_without_ec"] += 1
    return rxn_order, rxn_to_ec, stats


def write_output(output_path: Path, rxn_order: list[str], rxn_to_ec: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rxn_id", "EC number"])
        for rxn_id in rxn_order:
            writer.writerow([rxn_id, rxn_to_ec[rxn_id]])


def main() -> None:
    args = parse_args()
    enzyme_to_ecs = load_enzyme_ec_numbers(args.protein_path)
    rxn_order, rxn_to_ec, stats = build_rxn_ec_numbers(args.pair_path, enzyme_to_ecs)
    write_output(args.output_path, rxn_order, rxn_to_ec)
    print(f"Wrote {len(rxn_order)} reaction EC mappings to {args.output_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
