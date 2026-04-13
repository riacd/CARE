import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, Sampler

from CREEP.datasets.dataset_CREEP import encode_sequence

RDLogger.DisableLog("rdApp.warning")
TASK3_RXN_PREPROCESS_VERSION = "v2"


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
        enzyme_db = json.load(f)
    return enzyme_db


def _build_preprocessed_rxn_key(pair_db_path):
    digest = hashlib.sha256()
    path = Path(pair_db_path)
    stat = path.stat()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(TASK3_RXN_PREPROCESS_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_preprocessed_rxn_path(preprocessed_dir, pair_db_path):
    if preprocessed_dir is None:
        return None
    preprocessed_dir = Path(preprocessed_dir)
    pair_db_path = Path(pair_db_path)
    preprocess_key = _build_preprocessed_rxn_key(pair_db_path)
    return preprocessed_dir / f"{pair_db_path.stem}_deaam_{preprocess_key}.tsv"


def _load_or_preprocess_pair_db(pair_db_path, preprocessed_dir=None):
    pair_db = pd.read_csv(pair_db_path, sep="\t", usecols=["rxn_id", "enz_id", "mapped_rxn"])
    pair_db = pair_db.drop_duplicates(["rxn_id", "enz_id"]).reset_index(drop=True)
    preprocessed_rxn_path = _resolve_preprocessed_rxn_path(preprocessed_dir, pair_db_path)
    if preprocessed_rxn_path is not None and preprocessed_rxn_path.exists():
        print(f"Loaded task3 de-AAM reactions from {preprocessed_rxn_path}")
        return pd.read_csv(preprocessed_rxn_path, sep="\t")

    pair_db["reaction"] = pair_db["mapped_rxn"].map(_unmap_reaction)
    if preprocessed_rxn_path is not None:
        preprocessed_rxn_path.parent.mkdir(parents=True, exist_ok=True)
        pair_db.to_csv(preprocessed_rxn_path, sep="\t", index=False)
        print(f"Saved task3 de-AAM reactions to {preprocessed_rxn_path}")
    return pair_db


def load_task3_pairs(split_file, pair_db_path, enzyme_db_path, preprocessed_dir=None):
    split_file = Path(split_file)
    pairs = pd.read_csv(split_file, sep="\t")
    pairs = pairs[["rxn_id", "enz_id"]].drop_duplicates().reset_index(drop=True)

    pair_db = _load_or_preprocess_pair_db(pair_db_path, preprocessed_dir=preprocessed_dir)
    pairs = pairs.merge(pair_db, on=["rxn_id", "enz_id"], how="left")
    enzyme_db = _load_enzyme_db(enzyme_db_path)
    pairs["sequence"] = pairs["enz_id"].map(enzyme_db)

    stats = {
        "input_pairs": len(pairs),
        "missing_mapped_rxn": int(pairs["mapped_rxn"].isna().sum()),
        "invalid_reaction": int(pairs["reaction"].isna().sum()),
        "missing_sequence": int(pairs["sequence"].isna().sum()),
    }

    pairs = pairs.dropna(subset=["reaction", "sequence"]).reset_index(drop=True)
    stats["usable_pairs"] = len(pairs)
    return pairs, stats


class Task3CREEPDataset(Dataset):
    def __init__(
        self,
        split_file,
        pair_db_path,
        enzyme_db_path,
        protein_tokenizer,
        reaction_tokenizer,
        protein_max_sequence_len,
        reaction_max_sequence_len,
        preprocessed_dir=None,
    ):
        self.protein_tokenizer = protein_tokenizer
        self.reaction_tokenizer = reaction_tokenizer
        self.protein_max_sequence_len = protein_max_sequence_len
        self.reaction_max_sequence_len = reaction_max_sequence_len
        self.pairs, self.stats = load_task3_pairs(
            split_file,
            pair_db_path,
            enzyme_db_path,
            preprocessed_dir=preprocessed_dir,
        )

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        protein_sequence = " ".join(row["sequence"])
        reaction_sequence = row["reaction"]

        protein_input_ids, protein_attention_mask = encode_sequence(
            protein_sequence,
            self.protein_tokenizer,
            self.protein_max_sequence_len,
        )
        reaction_input_ids, reaction_attention_mask = encode_sequence(
            reaction_sequence,
            self.reaction_tokenizer,
            self.reaction_max_sequence_len,
        )

        return {
            "rxn_id": row["rxn_id"],
            "enz_id": row["enz_id"],
            "protein_sequence_input_ids": protein_input_ids,
            "protein_sequence_attention_mask": protein_attention_mask,
            "reaction_sequence_input_ids": reaction_input_ids,
            "reaction_sequence_attention_mask": reaction_attention_mask,
        }

    def __len__(self):
        return len(self.pairs)


class Task3PairBatchSampler(Sampler):
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
        return self.pairs.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

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
