import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


CARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = CARE_ROOT / "task4" / "data"
DEFAULT_PAIR_PATH = DEFAULT_DATA_DIR / "pair_merged_data" / "all_pair_data.tsv"
DEFAULT_BRENDA_PATH = DEFAULT_DATA_DIR / "brenda_rxntxt_uniprot_washed.tsv"
DEFAULT_RHEA_PATH = DEFAULT_DATA_DIR / "cleaned_rhea_uniprot_washed.tsv"
DEFAULT_RHEA_NAME_SMILES_PATH = DEFAULT_DATA_DIR / "rhea_name_smiles.txt"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "pair_merged_data" / "reaction_aliases.tsv"
DEFAULT_UNMATCHED_PATH = DEFAULT_DATA_DIR / "pair_merged_data" / "reaction_aliases_unmatched.tsv"
DEFAULT_METADATA_PATH = DEFAULT_DATA_DIR / "pair_merged_data" / "reaction_aliases.metadata.json"


csv.field_size_limit(sys.maxsize)
RDLogger.DisableLog("rdApp.*")


def canonicalize_molecule(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule smiles: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_side(side: str) -> str:
    parts = [part.strip() for part in side.split(".") if part.strip()]
    canonical_parts = sorted(canonicalize_molecule(part) for part in parts)
    return ".".join(canonical_parts)


def canonicalize_reaction(rxn_smiles: str) -> str:
    parts = rxn_smiles.strip().split(">>")
    if len(parts) != 2:
        raise ValueError(f"invalid reaction smiles: {rxn_smiles}")
    reactants, products = parts
    return f"{canonicalize_side(reactants)}>>{canonicalize_side(products)}"


def render_reaction_text(std_rxn: str, molecule_name_by_smiles: dict[str, str]) -> tuple[str, bool]:
    reactants, products = std_rxn.split(">>")
    used_fallback = False

    def render_side(side: str) -> str:
        nonlocal used_fallback
        names = []
        for smiles in side.split("."):
            smiles = smiles.strip()
            if not smiles:
                continue
            name = molecule_name_by_smiles.get(smiles)
            if not name:
                name = smiles
                used_fallback = True
            names.append(name)
        return " + ".join(names)

    return f"{render_side(reactants)} = {render_side(products)}", used_fallback


def clean_value(value: str) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value.lower() == "nan":
        return ""
    return value


def split_uniprot_ids(value: str) -> list[str]:
    return [item.strip() for item in clean_value(value).split(";") if item.strip()]


def read_rhea_name_smiles(path: Path) -> dict[str, str]:
    molecule_name_by_smiles = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = clean_value(row.get("Name", ""))
            smiles = clean_value(row.get("SMILES", ""))
            if not name or not smiles:
                continue
            try:
                canonical_smiles = canonicalize_molecule(smiles)
            except Exception:  # noqa: BLE001
                continue
            molecule_name_by_smiles.setdefault(canonical_smiles, name)
    return molecule_name_by_smiles


def read_source_records(source_name: str, path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    failures = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            raw_rxn = clean_value(row.get("mapped_rxn", ""))
            if not raw_rxn:
                failures.append({"source": source_name, "row_idx": row_idx, "reason": "missing mapped_rxn"})
                continue
            try:
                std_rxn = canonicalize_reaction(raw_rxn)
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": source_name, "row_idx": row_idx, "reason": str(exc)})
                continue

            if source_name == "brenda":
                enz_ids = split_uniprot_ids(row.get("Uniprot ID", ""))
                rxn_text = clean_value(row.get("RXN_TEXT", ""))
                rhea_id = ""
            else:
                enz_ids = split_uniprot_ids(row.get("Uniprot ID", ""))
                rxn_text = ""
                rhea_id = clean_value(row.get("Rhea ID", ""))

            if not enz_ids:
                failures.append({"source": source_name, "row_idx": row_idx, "reason": "missing Uniprot ID"})
                continue

            base_record = {
                "source": source_name,
                "RXN_TEXT": rxn_text,
                "Rhea_ID": rhea_id,
                "std_rxn": std_rxn,
            }
            for enz_id in enz_ids:
                record = dict(base_record)
                record["enz_id"] = enz_id
                records.append(record)
    return records, failures


def first_nonempty_text(records: list[dict]) -> str:
    for record in records:
        if clean_value(record.get("RXN_TEXT", "")):
            return clean_value(record["RXN_TEXT"])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve alias reaction text for task4 pair data.")
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--brenda-path", type=Path, default=DEFAULT_BRENDA_PATH)
    parser.add_argument("--rhea-path", type=Path, default=DEFAULT_RHEA_PATH)
    parser.add_argument("--rhea-name-smiles-path", type=Path, default=DEFAULT_RHEA_NAME_SMILES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--unmatched-path", type=Path, default=DEFAULT_UNMATCHED_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    brenda_records, brenda_failures = read_source_records("brenda", args.brenda_path)
    rhea_records, rhea_failures = read_source_records("rhea", args.rhea_path)
    molecule_name_by_smiles = read_rhea_name_smiles(args.rhea_name_smiles_path)

    brenda_by_enz = defaultdict(list)
    rhea_by_enz = defaultdict(list)
    best_by_std = defaultdict(list)
    brenda_text_by_std = {}

    for record in brenda_records:
        brenda_by_enz[record["enz_id"]].append(record)
        best_by_std[record["std_rxn"]].append(record)
        if record["std_rxn"] not in brenda_text_by_std and record["RXN_TEXT"]:
            brenda_text_by_std[record["std_rxn"]] = record["RXN_TEXT"]

    for record in rhea_records:
        rhea_by_enz[record["enz_id"]].append(record)
        best_by_std[record["std_rxn"]].append(record)

    stats = Counter()
    failure_rows = brenda_failures + rhea_failures

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.pair_path.open(newline="") as pair_handle,
        args.output_path.open("w", newline="") as output_handle,
        args.unmatched_path.open("w", newline="") as unmatched_handle,
    ):
        reader = csv.DictReader(pair_handle, delimiter="\t")
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["RXN_TEXT", "Rhea_ID", "std_rxn", "enz_id"],
            delimiter="\t",
        )
        unmatched_writer = csv.DictWriter(
            unmatched_handle,
            fieldnames=["enz_id", "std_rxn"],
            delimiter="\t",
        )
        writer.writeheader()
        unmatched_writer.writeheader()

        for row_idx, row in enumerate(reader, start=1):
            stats["pair_rows"] += 1
            enz_id = clean_value(row.get("enz_id", ""))
            raw_rxn = clean_value(row.get("mapped_rxn", ""))
            if not enz_id or not raw_rxn:
                stats["pair_row_failures"] += 1
                failure_rows.append(
                    {
                        "source": "pair",
                        "row_idx": row_idx,
                        "reason": "missing enz_id or mapped_rxn",
                    }
                )
                continue

            try:
                std_rxn = canonicalize_reaction(raw_rxn)
            except Exception as exc:  # noqa: BLE001
                stats["pair_row_failures"] += 1
                failure_rows.append({"source": "pair", "row_idx": row_idx, "reason": str(exc)})
                continue

            matched_record = None
            match_type = ""

            for record in brenda_by_enz.get(enz_id, []):
                if record["std_rxn"] == std_rxn:
                    matched_record = dict(record)
                    match_type = "brenda_by_enz"
                    break

            if matched_record is None:
                for record in rhea_by_enz.get(enz_id, []):
                    if record["std_rxn"] == std_rxn:
                        matched_record = dict(record)
                        match_type = "rhea_by_enz"
                        break

            if matched_record is None:
                for record in best_by_std.get(std_rxn, []):
                    matched_record = dict(record)
                    match_type = f"{record['source']}_by_std"
                    break

            if matched_record is None:
                stats["unmatched"] += 1
                unmatched_writer.writerow({"enz_id": enz_id, "std_rxn": std_rxn})
                continue

            if not matched_record["RXN_TEXT"]:
                matched_record["RXN_TEXT"] = brenda_text_by_std.get(std_rxn, "")
                if match_type == "rhea_by_enz" and matched_record["RXN_TEXT"]:
                    match_type = "rhea_by_enz_plus_brenda_text"
                elif match_type == "rhea_by_std" and matched_record["RXN_TEXT"]:
                    match_type = "rhea_by_std_plus_brenda_text"

            if not matched_record["RXN_TEXT"]:
                matched_record["RXN_TEXT"], used_fallback = render_reaction_text(std_rxn, molecule_name_by_smiles)
                if match_type == "rhea_by_enz":
                    match_type = "rhea_by_enz_plus_rhea_name_smiles"
                elif match_type == "rhea_by_std":
                    match_type = "rhea_by_std_plus_rhea_name_smiles"
                elif match_type == "brenda_by_enz":
                    match_type = "brenda_by_enz_plus_rhea_name_smiles"
                if used_fallback:
                    stats["generated_rxn_text_with_partial_smiles_fallback"] += 1
                else:
                    stats["generated_rxn_text_full_name_coverage"] += 1

            writer.writerow(
                {
                    "RXN_TEXT": matched_record["RXN_TEXT"],
                    "Rhea_ID": matched_record["Rhea_ID"],
                    "std_rxn": std_rxn,
                    "enz_id": enz_id,
                }
            )
            stats["matched"] += 1
            stats[match_type] += 1
            if matched_record["RXN_TEXT"]:
                stats["matched_with_rxn_text"] += 1
            else:
                stats["matched_without_rxn_text"] += 1

    metadata = {
        "pair_path": str(args.pair_path),
        "brenda_path": str(args.brenda_path),
        "rhea_path": str(args.rhea_path),
        "rhea_name_smiles_path": str(args.rhea_name_smiles_path),
        "output_path": str(args.output_path),
        "unmatched_path": str(args.unmatched_path),
        "stats": dict(stats),
        "source_failures": failure_rows[:1000],
        "source_failure_count": len(failure_rows),
        "unique_rhea_name_smiles": len(molecule_name_by_smiles),
        "unique_brenda_std_rxn_with_text": len(brenda_text_by_std),
        "unique_std_rxn_in_sources": len(best_by_std),
    }
    with args.metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metadata["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
