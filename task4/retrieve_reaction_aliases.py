from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib import error, parse, request
from xml.etree import ElementTree

from rdkit import Chem
from rdkit import RDLogger


CARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = CARE_ROOT / "task4" / "data"
DEFAULT_PAIR_PATH = CARE_ROOT / "data" / "pair_merged_data" / "all_pair_data.tsv"
DEFAULT_BRENDA_PATH = DEFAULT_DATA_DIR / "brenda_rxntxt_uniprot_washed.tsv"
DEFAULT_RHEA_PATH = DEFAULT_DATA_DIR / "cleaned_rhea_uniprot_washed.tsv"
DEFAULT_RHEA_NAME_SMILES_PATH = DEFAULT_DATA_DIR / "rhea_name_smiles.txt"
DEFAULT_OUTPUT_PATH = CARE_ROOT / "data" / "pair_merged_data" / "reaction_aliases.tsv"
DEFAULT_UNMATCHED_PATH = CARE_ROOT / "data" / "pair_merged_data" / "reaction_aliases_unmatched.tsv"
DEFAULT_METADATA_PATH = CARE_ROOT / "data" / "pair_merged_data" / "reaction_aliases.metadata.json"
DEFAULT_RHEA_CACHE_PATH = CARE_ROOT / "data" / "pair_merged_data" / "rhea_alias_cache.json"


csv.field_size_limit(sys.maxsize)
RDLogger.DisableLog("rdApp.*")

ATOM_MAP_PATTERN = re.compile(r":\d+(?=])")


def remove_atom_map_annotations(smiles: str) -> str:
    return ATOM_MAP_PATTERN.sub("", smiles.strip())


def canonicalize_molecule_strict(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule smiles: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_molecule(smiles: str) -> tuple[str, str]:
    try:
        return canonicalize_molecule_strict(smiles), "rdkit"
    except Exception:  # noqa: BLE001
        unmapped_smiles = remove_atom_map_annotations(smiles)
        try:
            return canonicalize_molecule_strict(unmapped_smiles), "rdkit_after_unmap"
        except Exception:  # noqa: BLE001
            return unmapped_smiles, "raw_unmapped_fragment"


def canonicalize_side(side: str) -> tuple[str, Counter]:
    parts = [part.strip() for part in side.split(".") if part.strip()]
    canonical_parts = []
    mode_counts = Counter()
    for part in parts:
        canonical_part, mode = canonicalize_molecule(part)
        canonical_parts.append(canonical_part)
        mode_counts[mode] += 1
    canonical_parts.sort()
    return ".".join(canonical_parts), mode_counts


def canonicalize_reaction(rxn_smiles: str) -> tuple[str, Counter]:
    parts = rxn_smiles.strip().split(">>")
    if len(parts) != 2:
        raise ValueError(f"invalid reaction smiles: {rxn_smiles}")
    reactants, products = parts
    reactant_side, reactant_modes = canonicalize_side(reactants)
    product_side, product_modes = canonicalize_side(products)
    return f"{reactant_side}>>{product_side}", reactant_modes + product_modes


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


def normalize_rhea_id(rhea_id: str) -> str:
    value = clean_value(rhea_id)
    if not value:
        return ""
    if value.upper().startswith("RHEA:"):
        return value.split(":", 1)[1].strip()
    return value


def normalize_equation_text(equation: str) -> str:
    text = clean_value(equation)
    if not text:
        return ""
    for arrow in (" <=> ", " => ", " = "):
        if arrow in text:
            left, right = text.split(arrow, 1)
            return f"{left.strip()} = {right.strip()}"
    return text.replace("<=>", "=").replace("=>", "=").strip()


def load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def normalize_cache_bucket(data: dict, key: str) -> dict[str, str]:
    bucket = data.get(key, {})
    if not isinstance(bucket, dict):
        return {}
    return {str(bucket_key): clean_value(str(bucket_value)) for bucket_key, bucket_value in bucket.items()}


def http_get_text(url: str, timeout: float) -> str:
    req = request.Request(url, headers={"User-Agent": "CARE-task4-rhea-alias/1.0"})
    with request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_chebi_name_from_web(chebi_id: str, timeout: float) -> str:
    query = parse.urlencode({"chebiId": chebi_id})
    url = f"https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?{query}"
    xml_text = http_get_text(url, timeout)
    root = ElementTree.fromstring(xml_text)
    for tag in ("chebiAsciiName", "synonym"):
        element = root.find(f".//{{*}}{tag}")
        if element is not None and clean_value(element.text or ""):
            return clean_value(element.text or "")
    return ""


class WebCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        data = load_json_cache(path)
        self.rhea_text = normalize_cache_bucket(data, "rhea_text")
        self.rhea_miss = normalize_cache_bucket(data, "rhea_miss")
        self.chebi_name = normalize_cache_bucket(data, "chebi_name")
        self.chebi_miss = normalize_cache_bucket(data, "chebi_miss")
        self.updated = False

    def save(self) -> None:
        if not self.updated:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rhea_text": self.rhea_text,
            "rhea_miss": self.rhea_miss,
            "chebi_name": self.chebi_name,
            "chebi_miss": self.chebi_miss,
        }
        with self.path.open("w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


class ChebiNameResolver:
    def __init__(self, web_cache: WebCache, timeout: float) -> None:
        self.web_cache = web_cache
        self.timeout = timeout

    def resolve(self, chebi_id: str) -> tuple[str, str]:
        normalized_id = clean_value(chebi_id)
        if not normalized_id:
            return "", "missing_chebi_id"
        cached = clean_value(self.web_cache.chebi_name.get(normalized_id, ""))
        if cached:
            return cached, "cache"
        if normalized_id in self.web_cache.chebi_miss:
            return "", self.web_cache.chebi_miss[normalized_id]
        try:
            name = fetch_chebi_name_from_web(normalized_id, self.timeout)
        except Exception:  # noqa: BLE001
            self.web_cache.chebi_miss[normalized_id] = "fetch_failed"
            self.web_cache.updated = True
            return "", "fetch_failed"
        if not name:
            self.web_cache.chebi_miss[normalized_id] = "empty_response"
            self.web_cache.updated = True
            return "", "empty_response"
        self.web_cache.chebi_name[normalized_id] = name
        self.web_cache.updated = True
        return name, "api"


def get_rhea_entry_name(entry: dict, chebi_name_resolver: ChebiNameResolver) -> str:
    name = clean_value(entry.get("label", ""))
    if name:
        return name

    chebi_prefix = clean_value(entry.get("idprefix", "")).lower()
    chebi_id = clean_value(entry.get("id", ""))
    if chebi_prefix == "chebi" and chebi_id:
        try:
            chebi_name, _ = chebi_name_resolver.resolve(f"CHEBI:{chebi_id}")
            return chebi_name
        except Exception:  # noqa: BLE001
            return ""
    return ""


def build_rhea_side_expression(entries: list[dict], chebi_name_resolver: ChebiNameResolver) -> list[str]:
    names = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = get_rhea_entry_name(entry, chebi_name_resolver)
        if name:
            names.append(name)
    return names


def fetch_rhea_reaction_text_from_web(
    rhea_id: str,
    timeout: float,
    chebi_name_resolver: ChebiNameResolver,
) -> str:
    normalized_id = normalize_rhea_id(rhea_id)
    if not normalized_id:
        return ""
    url = f"https://www.rhea-db.org/rhea/{normalized_id}/json"
    payload = json.loads(http_get_text(url, timeout))

    equation_text = normalize_equation_text(payload.get("equation", ""))
    if equation_text:
        return equation_text

    left_names = build_rhea_side_expression(payload.get("left", []), chebi_name_resolver)
    right_names = build_rhea_side_expression(payload.get("right", []), chebi_name_resolver)

    if left_names and right_names:
        return f"{' + '.join(left_names)} = {' + '.join(right_names)}".strip()
    if left_names or right_names:
        return " + ".join(left_names + right_names)
    return ""


class RheaAliasResolver:
    def __init__(self, web_cache: WebCache, chebi_name_resolver: ChebiNameResolver, timeout: float) -> None:
        self.web_cache = web_cache
        self.chebi_name_resolver = chebi_name_resolver
        self.timeout = timeout

    def resolve(self, rhea_id: str) -> tuple[str, str]:
        normalized_id = normalize_rhea_id(rhea_id)
        if not normalized_id:
            return "", "missing_rhea_id"
        cached = clean_value(self.web_cache.rhea_text.get(normalized_id, ""))
        if cached:
            return cached, "cache"
        if normalized_id in self.web_cache.rhea_miss:
            return "", self.web_cache.rhea_miss[normalized_id]
        try:
            text = fetch_rhea_reaction_text_from_web(
                normalized_id,
                self.timeout,
                self.chebi_name_resolver,
            )
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError, ElementTree.ParseError):
            self.web_cache.rhea_miss[normalized_id] = "fetch_failed"
            self.web_cache.updated = True
            return "", "fetch_failed"
        except Exception:  # noqa: BLE001
            self.web_cache.rhea_miss[normalized_id] = "fetch_failed"
            self.web_cache.updated = True
            return "", "fetch_failed"
        if not text:
            self.web_cache.rhea_miss[normalized_id] = "empty_response"
            self.web_cache.updated = True
            return "", "empty_response"
        self.web_cache.rhea_text[normalized_id] = text
        self.web_cache.updated = True
        return text, "api"

    def save(self) -> None:
        self.web_cache.save()


def clean_value(value: str) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value.lower() == "nan":
        return ""
    return value


def split_uniprot_ids(value: str) -> list[str]:
    return [item.strip() for item in clean_value(value).split(";") if item.strip()]


def record_canonicalization_stats(stats: Counter, source_name: str, mode_counts: Counter) -> None:
    for mode, count in mode_counts.items():
        stats[f"{source_name}_molecule_{mode}"] += count


def read_rhea_name_smiles(path: Path, stats: Counter) -> dict[str, str]:
    molecule_name_by_smiles = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            name = clean_value(row.get("Name", ""))
            smiles = clean_value(row.get("SMILES", ""))
            if not name or not smiles:
                continue
            canonical_smiles, mode = canonicalize_molecule(smiles)
            stats[f"rhea_name_smiles_molecule_{mode}"] += 1
            molecule_name_by_smiles.setdefault(canonical_smiles, name)
    return molecule_name_by_smiles


def read_source_records(source_name: str, path: Path, stats: Counter) -> tuple[list[dict], list[dict]]:
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
                std_rxn, mode_counts = canonicalize_reaction(raw_rxn)
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": source_name, "row_idx": row_idx, "reason": str(exc)})
                continue
            record_canonicalization_stats(stats, source_name, mode_counts)

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


def build_source_indexes(
    brenda_records: list[dict],
    rhea_records: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]], dict[str, str]]:
    brenda_by_enz = defaultdict(list)
    rhea_by_enz = defaultdict(list)
    best_by_std = defaultdict(list)
    brenda_text_by_std = {}

    for record in brenda_records:
        brenda_by_enz[record["enz_id"]].append(record)
        best_by_std[record["std_rxn"]].append(record)
        if record["RXN_TEXT"] and record["std_rxn"] not in brenda_text_by_std:
            brenda_text_by_std[record["std_rxn"]] = record["RXN_TEXT"]

    for record in rhea_records:
        rhea_by_enz[record["enz_id"]].append(record)
        best_by_std[record["std_rxn"]].append(record)

    return brenda_by_enz, rhea_by_enz, best_by_std, brenda_text_by_std


def find_matching_record(
    enz_id: str,
    std_rxn: str,
    brenda_by_enz: dict[str, list[dict]],
    rhea_by_enz: dict[str, list[dict]],
    best_by_std: dict[str, list[dict]],
) -> tuple[dict | None, str]:
    for record in brenda_by_enz.get(enz_id, []):
        if record["std_rxn"] == std_rxn:
            return dict(record), "brenda_by_enz"

    for record in rhea_by_enz.get(enz_id, []):
        if record["std_rxn"] == std_rxn:
            return dict(record), "rhea_by_enz"

    for record in best_by_std.get(std_rxn, []):
        return dict(record), f"{record['source']}_by_std"

    return None, ""


def maybe_fill_from_brenda_text(record: dict, std_rxn: str, match_type: str, brenda_text_by_std: dict[str, str]) -> str:
    if record["RXN_TEXT"]:
        return match_type

    record["RXN_TEXT"] = brenda_text_by_std.get(std_rxn, "")
    if match_type == "rhea_by_enz" and record["RXN_TEXT"]:
        return "rhea_by_enz_plus_brenda_text"
    if match_type == "rhea_by_std" and record["RXN_TEXT"]:
        return "rhea_by_std_plus_brenda_text"
    return match_type


def build_generated_match_type(match_type: str, generated_via_rhea: bool) -> str:
    suffix = "rhea_id" if generated_via_rhea else "rhea_name_smiles"
    if match_type == "rhea_by_enz":
        return f"rhea_by_enz_plus_{suffix}"
    if match_type == "rhea_by_std":
        return f"rhea_by_std_plus_{suffix}"
    if match_type == "brenda_by_enz":
        return f"brenda_by_enz_plus_{suffix}"
    return match_type


def fill_reaction_text(
    record: dict,
    std_rxn: str,
    match_type: str,
    brenda_text_by_std: dict[str, str],
    molecule_name_by_smiles: dict[str, str],
    rhea_alias_resolver: RheaAliasResolver,
    stats: Counter,
) -> tuple[dict, str]:
    match_type = maybe_fill_from_brenda_text(record, std_rxn, match_type, brenda_text_by_std)
    if record["RXN_TEXT"]:
        return record, match_type

    generated_text, used_fallback = render_reaction_text(std_rxn, molecule_name_by_smiles)
    generated_via_rhea = False
    if used_fallback and record["Rhea_ID"]:
        rhea_text, rhea_source = rhea_alias_resolver.resolve(record["Rhea_ID"])
        if rhea_text:
            record["RXN_TEXT"] = rhea_text
            generated_via_rhea = True
            stats["generated_rxn_text_via_rhea_id"] += 1
            stats[f"generated_rxn_text_via_rhea_id_{rhea_source}"] += 1
        else:
            stats[f"generated_rxn_text_via_rhea_id_{rhea_source}"] += 1

    if not record["RXN_TEXT"]:
        record["RXN_TEXT"] = generated_text
        if used_fallback:
            stats["generated_rxn_text_with_partial_smiles_fallback"] += 1
        else:
            stats["generated_rxn_text_full_name_coverage"] += 1

    return record, build_generated_match_type(match_type, generated_via_rhea)


def tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve alias reaction text for task4 pair data.")
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--brenda-path", type=Path, default=DEFAULT_BRENDA_PATH)
    parser.add_argument("--rhea-path", type=Path, default=DEFAULT_RHEA_PATH)
    parser.add_argument("--rhea-name-smiles-path", type=Path, default=DEFAULT_RHEA_NAME_SMILES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--unmatched-path", type=Path, default=DEFAULT_UNMATCHED_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--rhea-cache-path", type=Path, default=DEFAULT_RHEA_CACHE_PATH)
    parser.add_argument("--rhea-timeout", type=float, default=10.0)
    args = parser.parse_args()

    stats = Counter()
    brenda_records, brenda_failures = read_source_records("brenda", args.brenda_path, stats)
    rhea_records, rhea_failures = read_source_records("rhea", args.rhea_path, stats)
    molecule_name_by_smiles = read_rhea_name_smiles(args.rhea_name_smiles_path, stats)
    web_cache = WebCache(args.rhea_cache_path)
    chebi_name_resolver = ChebiNameResolver(web_cache, args.rhea_timeout)
    rhea_alias_resolver = RheaAliasResolver(web_cache, chebi_name_resolver, args.rhea_timeout)
    brenda_by_enz, rhea_by_enz, best_by_std, brenda_text_by_std = build_source_indexes(
        brenda_records,
        rhea_records,
    )

    failure_rows = brenda_failures + rhea_failures

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp_path = tmp_path(args.output_path)
    unmatched_tmp_path = tmp_path(args.unmatched_path)
    metadata_tmp_path = tmp_path(args.metadata_path)

    with args.pair_path.open(newline="") as pair_handle, \
        output_tmp_path.open("w", newline="") as output_handle, \
        unmatched_tmp_path.open("w", newline="") as unmatched_handle:
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
                std_rxn, mode_counts = canonicalize_reaction(raw_rxn)
            except Exception as exc:  # noqa: BLE001
                stats["pair_row_failures"] += 1
                failure_rows.append({"source": "pair", "row_idx": row_idx, "reason": str(exc)})
                continue
            record_canonicalization_stats(stats, "pair", mode_counts)

            matched_record, match_type = find_matching_record(
                enz_id,
                std_rxn,
                brenda_by_enz,
                rhea_by_enz,
                best_by_std,
            )

            if matched_record is None:
                stats["unmatched"] += 1
                unmatched_writer.writerow({"enz_id": enz_id, "std_rxn": std_rxn})
                continue

            matched_record, match_type = fill_reaction_text(
                matched_record,
                std_rxn,
                match_type,
                brenda_text_by_std,
                molecule_name_by_smiles,
                rhea_alias_resolver,
                stats,
            )

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

    rhea_alias_resolver.save()

    metadata = {
        "pair_path": str(args.pair_path),
        "brenda_path": str(args.brenda_path),
        "rhea_path": str(args.rhea_path),
        "rhea_name_smiles_path": str(args.rhea_name_smiles_path),
        "rhea_cache_path": str(args.rhea_cache_path),
        "output_path": str(args.output_path),
        "unmatched_path": str(args.unmatched_path),
        "stats": dict(stats),
        "source_failures": failure_rows[:1000],
        "source_failure_count": len(failure_rows),
        "unique_rhea_name_smiles": len(molecule_name_by_smiles),
        "unique_brenda_std_rxn_with_text": len(brenda_text_by_std),
        "unique_std_rxn_in_sources": len(best_by_std),
    }
    with metadata_tmp_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    output_tmp_path.replace(args.output_path)
    unmatched_tmp_path.replace(args.unmatched_path)
    metadata_tmp_path.replace(args.metadata_path)

    print(json.dumps(metadata["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
