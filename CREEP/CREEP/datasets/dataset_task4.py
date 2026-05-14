import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, Sampler

from CREEP.datasets.dataset_CREEP import encode_sequence

RDLogger.DisableLog("rdApp.warning")
TASK4_RXN_PREPROCESS_VERSION = "v1"


def _unmap_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))


def _unmap_reaction(mapped_rxn):
    if not isinstance(mapped_rxn, str) or ">>" not in mapped_rxn:
        return None

    reactants, products = mapped_rxn.split(">>", 1)
    try:
        reactants = _unmap_smiles(reactants)
        products = _unmap_smiles(products)
        if reactants is None or products is None:
            return None
        return f"{reactants}>>{products}"
    except Exception:
        return None


def _load_enzyme_db(enzyme_db_path):
    with open(enzyme_db_path) as f:
        return json.load(f)


def _build_preprocessed_rxn_key(pair_db_path):
    digest = hashlib.sha256()
    path = Path(pair_db_path)
    stat = path.stat()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(TASK4_RXN_PREPROCESS_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_preprocessed_rxn_path(preprocessed_dir, pair_db_path):
    if preprocessed_dir is None:
        return None
    preprocessed_dir = Path(preprocessed_dir)
    pair_db_path = Path(pair_db_path)
    preprocess_key = _build_preprocessed_rxn_key(pair_db_path)
    return preprocessed_dir / f"{pair_db_path.stem}_deaam_{preprocess_key}.tsv"


def _first_present_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _normalize_text_value(value):
    if pd.isna(value):
        return ""
    value = str(value)
    return value.strip().strip('"')


def _load_or_preprocess_pair_db(pair_db_path, enzyme_db_path=None, preprocessed_dir=None):
    pair_db_path = Path(pair_db_path)
    pair_db = pd.read_csv(pair_db_path, sep="\t")

    required = {"rxn_id", "enz_id"}
    missing_required = required - set(pair_db.columns)
    if missing_required:
        raise ValueError(f"pair db missing required columns: {sorted(missing_required)}")

    pair_db = pair_db.drop_duplicates(["rxn_id", "enz_id"]).reset_index(drop=True)
    preprocessed_rxn_path = _resolve_preprocessed_rxn_path(preprocessed_dir, pair_db_path)

    if preprocessed_rxn_path is not None and preprocessed_rxn_path.exists():
        print(f"Loaded task4 de-AAM reactions from {preprocessed_rxn_path}")
        processed = pd.read_csv(preprocessed_rxn_path, sep="\t")
        if "text" in processed.columns:
            processed["text"] = processed["text"].map(_normalize_text_value)
        return processed

    reaction_column = _first_present_column(pair_db, ["reaction", "reaction_y", "reaction_x"])
    sequence_column = _first_present_column(pair_db, ["sequence"])
    text_column = _first_present_column(pair_db, ["text", "iupac_text", "ec_text"])

    if reaction_column is not None:
        pair_db["reaction"] = pair_db[reaction_column]
    elif "mapped_rxn" in pair_db.columns:
        pair_db["reaction"] = pair_db["mapped_rxn"].map(_unmap_reaction)
    else:
        pair_db["reaction"] = None

    if sequence_column is None and enzyme_db_path is not None:
        enzyme_db = _load_enzyme_db(enzyme_db_path)
        pair_db["sequence"] = pair_db["enz_id"].map(enzyme_db)
    elif sequence_column is not None:
        pair_db["sequence"] = pair_db[sequence_column]
    else:
        pair_db["sequence"] = None

    if text_column is not None:
        pair_db["text"] = pair_db[text_column].map(_normalize_text_value)
    else:
        pair_db["text"] = ""

    keep_columns = [
        column
        for column in ["rxn_id", "enz_id", "mapped_rxn", "reaction", "sequence", "ec_number", "ec_text", "iupac_text", "text"]
        if column in pair_db.columns
    ]
    pair_db = pair_db[keep_columns].copy()

    if preprocessed_rxn_path is not None:
        preprocessed_rxn_path.parent.mkdir(parents=True, exist_ok=True)
        pair_db.to_csv(preprocessed_rxn_path, sep="\t", index=False)
        print(f"Saved task4 normalized pair db to {preprocessed_rxn_path}")

    return pair_db


def load_task4_pairs(split_file, pair_db_path, enzyme_db_path=None, preprocessed_dir=None):
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
        )
        merge_columns = ["rxn_id", "enz_id"]
        for column in ["reaction", "sequence", "text", "ec_number", "ec_text", "iupac_text", "mapped_rxn"]:
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
