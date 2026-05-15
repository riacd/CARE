import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, Sampler

from CREEP.datasets.dataset_CREEP import encode_sequence

RDLogger.DisableLog("rdApp.*")
TASK4_RXN_PREPROCESS_VERSION = "v4_rxn_text_plus_ec_text"
ATOM_MAP_PATTERN = re.compile(r":\d+(?=])")


def _normalize_text_value(value):
    if pd.isna(value):
        return ""
    value = str(value)
    return value.strip().strip('"')


def _remove_atom_map_annotations(smiles):
    return ATOM_MAP_PATTERN.sub("", str(smiles).strip())


def _canonicalize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule smiles: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _canonicalize_molecule_tolerant(smiles):
    try:
        return _canonicalize_molecule(smiles)
    except Exception:
        unmapped_smiles = _remove_atom_map_annotations(smiles)
        try:
            return _canonicalize_molecule(unmapped_smiles)
        except Exception:
            return unmapped_smiles


def _canonicalize_side(side):
    parts = [part.strip() for part in str(side).split(".") if part.strip()]
    canonical_parts = [_canonicalize_molecule_tolerant(part) for part in parts]
    canonical_parts.sort()
    return ".".join(canonical_parts)


def _unmap_reaction(mapped_rxn):
    if not isinstance(mapped_rxn, str) or ">>" not in mapped_rxn:
        return None

    reactants, products = mapped_rxn.split(">>", 1)
    try:
        return f"{_canonicalize_side(reactants)}>>{_canonicalize_side(products)}"
    except Exception:
        return None


def _load_enzyme_db(enzyme_db_path):
    with open(enzyme_db_path) as f:
        return json.load(f)


def _build_preprocessed_rxn_key(pair_db_path, dependency_paths=None):
    digest = hashlib.sha256()
    path = Path(pair_db_path)
    stat = path.stat()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    for dependency_path in dependency_paths or []:
        dependency_path = Path(dependency_path)
        if not dependency_path.exists():
            digest.update(f"{dependency_path}:missing".encode("utf-8"))
            continue
        dep_stat = dependency_path.stat()
        digest.update(str(dependency_path.resolve()).encode("utf-8"))
        digest.update(str(dep_stat.st_size).encode("utf-8"))
        digest.update(str(dep_stat.st_mtime_ns).encode("utf-8"))
    digest.update(TASK4_RXN_PREPROCESS_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_preprocessed_rxn_path(preprocessed_dir, pair_db_path, dependency_paths=None):
    if preprocessed_dir is None:
        return None
    preprocessed_dir = Path(preprocessed_dir)
    pair_db_path = Path(pair_db_path)
    preprocess_key = _build_preprocessed_rxn_key(pair_db_path, dependency_paths=dependency_paths)
    return preprocessed_dir / f"{pair_db_path.stem}_deaam_{preprocess_key}.tsv"


def _first_present_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _load_reaction_aliases(alias_path):
    alias_path = Path(alias_path)
    alias_df = pd.read_csv(alias_path, sep="\t")
    alias_df = alias_df.rename(columns={"RXN_TEXT": "rxn_text"})
    alias_df["rxn_text"] = alias_df["rxn_text"].map(_normalize_text_value)
    keep_columns = [column for column in ["enz_id", "std_rxn", "rxn_text", "Rhea_ID"] if column in alias_df.columns]
    alias_df = alias_df[keep_columns].drop_duplicates(["enz_id", "std_rxn"]).reset_index(drop=True)
    return alias_df


def _load_rxn_ec_map(rxn_ec_number_path, ec_text_path):
    rxn_ec_df = pd.read_csv(rxn_ec_number_path, sep="\t")
    rxn_ec_df["EC number"] = rxn_ec_df["EC number"].fillna("").astype(str).str.strip()

    ec_text_df = pd.read_csv(ec_text_path)
    ec_text_df = ec_text_df[["EC number", "Text"]].copy()
    ec_text_df["EC number"] = ec_text_df["EC number"].fillna("").astype(str).str.strip()
    ec_text_df["Text"] = ec_text_df["Text"].map(_normalize_text_value)

    rxn_ec_df = rxn_ec_df.merge(ec_text_df, on="EC number", how="left")
    rxn_ec_df = rxn_ec_df.rename(columns={"EC number": "ec_number", "Text": "ec_text"})
    rxn_ec_df["ec_text"] = rxn_ec_df["ec_text"].map(_normalize_text_value)
    return rxn_ec_df[["rxn_id", "ec_number", "ec_text"]].drop_duplicates(["rxn_id"]).reset_index(drop=True)


def _compose_task4_text(rxn_text, ec_text, ec_number):
    rxn_text = _normalize_text_value(rxn_text)
    ec_text = _normalize_text_value(ec_text)
    _ = ec_number
    if not rxn_text and not ec_text:
        return ""
    return f"{rxn_text}\t{ec_text}"


def _load_or_preprocess_pair_db(
    pair_db_path,
    enzyme_db_path=None,
    preprocessed_dir=None,
    reaction_aliases_path=None,
    rxn_ec_number_path=None,
    ec_text_path=None,
):
    pair_db_path = Path(pair_db_path)
    pair_db = pd.read_csv(pair_db_path, sep="\t")

    required = {"rxn_id", "enz_id"}
    missing_required = required - set(pair_db.columns)
    if missing_required:
        raise ValueError(f"pair db missing required columns: {sorted(missing_required)}")

    pair_db = pair_db.drop_duplicates(["rxn_id", "enz_id"]).reset_index(drop=True)
    dependency_paths = [
        path
        for path in [reaction_aliases_path, rxn_ec_number_path, ec_text_path, enzyme_db_path]
        if path is not None
    ]
    preprocessed_rxn_path = _resolve_preprocessed_rxn_path(
        preprocessed_dir,
        pair_db_path,
        dependency_paths=dependency_paths,
    )

    if preprocessed_rxn_path is not None and preprocessed_rxn_path.exists():
        print(f"Loaded task4 de-AAM reactions from {preprocessed_rxn_path}")
        processed = pd.read_csv(preprocessed_rxn_path, sep="\t")
        if "text" in processed.columns:
            processed["text"] = processed["text"].map(_normalize_text_value)
        return processed

    reaction_column = _first_present_column(pair_db, ["reaction", "reaction_y", "reaction_x"])
    sequence_column = _first_present_column(pair_db, ["sequence", "Enzyme Seq"])
    text_column = _first_present_column(pair_db, ["text", "iupac_text", "ec_text"])

    if reaction_column is not None:
        pair_db["reaction"] = pair_db[reaction_column]
    elif "mapped_rxn" in pair_db.columns:
        pair_db["reaction"] = pair_db["mapped_rxn"].map(_unmap_reaction)
    else:
        pair_db["reaction"] = None
    pair_db["std_rxn"] = pair_db["reaction"]

    if sequence_column is None and enzyme_db_path is not None:
        enzyme_db = _load_enzyme_db(enzyme_db_path)
        pair_db["sequence"] = pair_db["enz_id"].map(enzyme_db)
    elif sequence_column is not None:
        pair_db["sequence"] = pair_db[sequence_column]
    else:
        pair_db["sequence"] = None

    if reaction_aliases_path is not None:
        alias_df = _load_reaction_aliases(reaction_aliases_path)
        pair_db = pair_db.merge(alias_df, on=["enz_id", "std_rxn"], how="left")
    else:
        pair_db["rxn_text"] = ""

    if rxn_ec_number_path is not None and ec_text_path is not None:
        rxn_ec_df = _load_rxn_ec_map(rxn_ec_number_path, ec_text_path)
        pair_db = pair_db.merge(rxn_ec_df, on="rxn_id", how="left")
    else:
        pair_db["ec_number"] = ""
        pair_db["ec_text"] = ""

    pair_db["rxn_text"] = pair_db.get("rxn_text", "").map(_normalize_text_value)
    pair_db["ec_number"] = pair_db.get("ec_number", "").fillna("").astype(str).str.strip()
    pair_db["ec_text"] = pair_db.get("ec_text", "").map(_normalize_text_value)

    if text_column is not None and "text" in pair_db.columns:
        pair_db["text"] = pair_db[text_column].map(_normalize_text_value)
    else:
        pair_db["text"] = pair_db.apply(
            lambda row: _compose_task4_text(row.get("rxn_text", ""), row.get("ec_text", ""), row.get("ec_number", "")),
            axis=1,
        )

    keep_columns = [
        column
        for column in [
            "rxn_id",
            "enz_id",
            "mapped_rxn",
            "std_rxn",
            "reaction",
            "sequence",
            "ec_number",
            "ec_text",
            "rxn_text",
            "iupac_text",
            "text",
        ]
        if column in pair_db.columns
    ]
    pair_db = pair_db[keep_columns].copy()

    if preprocessed_rxn_path is not None:
        preprocessed_rxn_path.parent.mkdir(parents=True, exist_ok=True)
        pair_db.to_csv(preprocessed_rxn_path, sep="\t", index=False)
        print(f"Saved task4 normalized pair db to {preprocessed_rxn_path}")

    return pair_db


def load_task4_pairs(
    split_file,
    pair_db_path,
    enzyme_db_path=None,
    preprocessed_dir=None,
    reaction_aliases_path=None,
    rxn_ec_number_path=None,
    ec_text_path=None,
):
    split_file = Path(split_file)
    pairs = pd.read_csv(split_file, sep="\t")

    if "rxn_id" not in pairs.columns or "enz_id" not in pairs.columns:
        raise ValueError("split file must contain rxn_id and enz_id columns")

    split_columns = [column for column in ["rxn_id", "enz_id", "reaction", "sequence", "text"] if column in pairs.columns]
    pairs = pairs[split_columns].drop_duplicates(["rxn_id", "enz_id"]).reset_index(drop=True)

    stats = {
        "input_pairs": len(pairs),
        "pretokenized_reaction_in_split": int("reaction" in pairs.columns),
        "pretokenized_sequence_in_split": int("sequence" in pairs.columns),
        "pretokenized_text_in_split": int("text" in pairs.columns),
    }

    missing_reaction = "reaction" not in pairs.columns
    missing_sequence = "sequence" not in pairs.columns
    missing_text = "text" not in pairs.columns
    if missing_reaction or missing_sequence or missing_text:
        pair_db = _load_or_preprocess_pair_db(
            pair_db_path,
            enzyme_db_path=enzyme_db_path,
            preprocessed_dir=preprocessed_dir,
            reaction_aliases_path=reaction_aliases_path,
            rxn_ec_number_path=rxn_ec_number_path,
            ec_text_path=ec_text_path,
        )
        merge_columns = ["rxn_id", "enz_id"]
        for column in ["reaction", "sequence", "text", "ec_number", "ec_text", "rxn_text", "iupac_text", "mapped_rxn", "std_rxn"]:
            if column in pair_db.columns:
                merge_columns.append(column)
        pairs = pairs.merge(
            pair_db[merge_columns].drop_duplicates(["rxn_id", "enz_id"]),
            on=["rxn_id", "enz_id"],
            how="left",
            suffixes=("", "_db"),
        )
        for field in ["reaction", "sequence", "text"]:
            db_field = f"{field}_db"
            if db_field in pairs.columns:
                if field in pairs.columns:
                    pairs[field] = pairs[field].fillna(pairs[db_field])
                else:
                    pairs[field] = pairs[db_field]
                pairs = pairs.drop(columns=[db_field])

    if "text" not in pairs.columns:
        pairs["text"] = ""
    pairs["text"] = pairs["text"].map(_normalize_text_value)

    stats["missing_reaction"] = int(pairs["reaction"].isna().sum()) if "reaction" in pairs.columns else len(pairs)
    stats["missing_sequence"] = int(pairs["sequence"].isna().sum()) if "sequence" in pairs.columns else len(pairs)
    stats["missing_text"] = int(pairs["text"].isna().sum()) if "text" in pairs.columns else len(pairs)
    stats["empty_text"] = int((pairs["text"] == "").sum()) if "text" in pairs.columns else len(pairs)

    pairs = pairs.dropna(subset=["reaction", "sequence"]).reset_index(drop=True)
    stats["usable_pairs"] = len(pairs)
    stats["unique_reactions"] = int(pairs["rxn_id"].nunique())
    stats["unique_enzymes"] = int(pairs["enz_id"].nunique())
    return pairs, stats


class Task4CREEPDataset(Dataset):
    def __init__(
        self,
        split_file,
        pair_db_path,
        enzyme_db_path,
        protein_tokenizer,
        text_tokenizer,
        reaction_tokenizer,
        protein_max_sequence_len,
        text_max_sequence_len,
        reaction_max_sequence_len,
        preprocessed_dir=None,
        reaction_aliases_path=None,
        rxn_ec_number_path=None,
        ec_text_path=None,
    ):
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.text_max_sequence_len = text_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len
        self.pairs, self.stats = load_task4_pairs(
            split_file,
            pair_db_path,
            enzyme_db_path=enzyme_db_path,
            preprocessed_dir=preprocessed_dir,
            reaction_aliases_path=reaction_aliases_path,
            rxn_ec_number_path=rxn_ec_number_path,
            ec_text_path=ec_text_path,
        )

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        protein_sequence = " ".join(row["sequence"])
        text_sequence = row["text"]
        reaction_sequence = row["reaction"]

        protein_input_ids, protein_attention_mask = encode_sequence(
            protein_sequence,
            self.protein_tokenizer,
            self.protein_max_sequence_len,
        )
        text_input_ids, text_attention_mask = encode_sequence(
            text_sequence,
            self.text_tokenizer,
            self.text_max_sequence_len,
        )
        reaction_input_ids, reaction_attention_mask = encode_sequence(
            reaction_sequence,
            self.reaction_tokenizer,
            self.reaction_max_sequence_len,
        )

        return {
            "rxn_id": row["rxn_id"],
            "enz_id": row["enz_id"],
            "text": text_sequence,
            "protein_sequence_input_ids": protein_input_ids,
            "protein_sequence_attention_mask": protein_attention_mask,
            "text_sequence_input_ids": text_input_ids,
            "text_sequence_attention_mask": text_attention_mask,
            "reaction_sequence_input_ids": reaction_input_ids,
            "reaction_sequence_attention_mask": reaction_attention_mask,
        }

    def __len__(self):
        return len(self.pairs)


class Task4PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_batches, shuffle=True, seed=42):
        if batch_size <= 1:
            raise ValueError("batch_size must be greater than 1 for contrastive training.")

        self.pairs = dataset.pairs.reset_index(drop=True).copy()
        self.pairs["pair_idx"] = self.pairs.index
        self.pairs["rxn_idx"] = self.pairs["rxn_id"].factorize()[0]
        self.pairs["enz_idx"] = self.pairs["enz_id"].factorize()[0]
        self.rxn_groups = dict(list(self.pairs.groupby("rxn_idx", sort=False)["enz_idx"]))
        self.enz_groups = dict(list(self.pairs.groupby("enz_idx", sort=False)["rxn_idx"]))
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _mask_2hops(self, center_pair, candidates):
        rxn_neighbors = self.rxn_groups[center_pair["rxn_idx"]]
        enz_neighbors = self.enz_groups[center_pair["enz_idx"]]
        mask = np.isin(candidates["enz_idx"].to_numpy(), rxn_neighbors)
        mask |= np.isin(candidates["rxn_idx"].to_numpy(), enz_neighbors)
        return candidates.loc[~mask]

    def _reset_candidates(self, rng):
        if not self.shuffle:
            return self.pairs.copy()
        random_state = int(rng.integers(0, 2**31 - 1))
        return self.pairs.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def _sample_row(self, candidates, rng):
        sample_pos = int(rng.integers(0, len(candidates)))
        return candidates.iloc[sample_pos]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches_yielded = 0
        candidates = self._reset_candidates(rng)

        while batches_yielded < self.num_batches:
            if len(candidates) == 0:
                candidates = self._reset_candidates(rng)

            batch = []
            working = candidates
            while len(batch) < self.batch_size:
                if len(working) == 0:
                    batch = []
                    candidates = self._reset_candidates(rng)
                    working = candidates
                    continue

                sample = self._sample_row(working, rng)
                batch.append(int(sample["pair_idx"]))
                working = self._mask_2hops(sample, working)

            candidates = working
            yield batch
            batches_yielded += 1
