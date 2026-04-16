import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, T5EncoderModel, T5Tokenizer

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
except ImportError:
    RDLogger = None

SCRIPT_PATH = Path(__file__).resolve()

# Support both layouts:
# 1. repository-root/baseline/CARE/...
# 2. standalone CARE/...
if (SCRIPT_PATH.parents[2] / "CREEP").exists():
    CARE_ROOT = SCRIPT_PATH.parents[2]
    REPO_ROOT = CARE_ROOT.parent
else:
    REPO_ROOT = SCRIPT_PATH.parents[4]
    CARE_ROOT = REPO_ROOT / "baseline" / "CARE"

CREEP_ROOT = CARE_ROOT / "CREEP"
for path in (REPO_ROOT, CREEP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from CREEP.datasets.dataset_CREEP import encode_sequence
from CREEP.datasets.dataset_task3 import _load_or_preprocess_pair_db
from CREEP.models import SingleModalityModel
from CREEP.utils.tokenization import SmilesTokenizer


class OrderedSequenceDataset(Dataset):
    def __init__(self, entries, tokenizer, max_sequence_len, modality):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.modality = modality

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item_id, sequence = self.entries[index]
        if self.modality == "protein":
            sequence = " ".join(sequence)

        sequence_input_ids, sequence_attention_mask = encode_sequence(
            sequence,
            self.tokenizer,
            self.max_sequence_len,
        )
        return {
            "item_id": item_id,
            "sequence_input_ids": sequence_input_ids,
            "sequence_attention_mask": sequence_attention_mask,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ssl_emb_dim", type=int, default=256)
    parser.add_argument(
        "--split_type",
        type=str,
        default="all",
        choices=["enzyme_split", "rxn_sub_split", "all"],
    )
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--pretrained_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--pair_db_path",
        type=str,
        default=str(CARE_ROOT / "task3" / "data" / "pair_merged_data" / "all_pair_data.tsv"),
    )
    parser.add_argument(
        "--enzyme_db_path",
        type=str,
        default=str(CARE_ROOT / "task3" / "data" / "pair_merged_data" / "enzyme_db_extended.json"),
    )
    parser.add_argument(
        "--enzyme_candidates_mode",
        type=str,
        default="extended",
        choices=["extended", "original_entries"],
    )
    parser.add_argument(
        "--enzyme_db_metadata_path",
        type=str,
        default=str(CARE_ROOT / "task3" / "data" / "pair_merged_data" / "enzyme_db_extended.metadata.json"),
    )
    parser.add_argument("--protein_backbone_model", type=str, default="ProtT5", choices=["ProtT5"])
    parser.add_argument("--reaction_backbone_model", type=str, default="rxnfp")
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--reaction_max_sequence_len", type=int, default=512)
    parser.add_argument("--batch_size_protein", type=int, default=4)
    parser.add_argument("--batch_size_reaction", type=int, default=256)
    parser.add_argument("--similarity_reaction_chunk_size", type=int, default=64)
    parser.add_argument("--similarity_protein_chunk_size", type=int, default=65536)
    parser.add_argument(
        "--preprocessed_rxn_dir",
        type=str,
        default=str(CARE_ROOT / "runtime" / "task3_preprocessed"),
    )
    parser.add_argument("--max_reactions", type=int, default=None)
    parser.add_argument("--max_enzymes", type=int, default=None)
    parser.add_argument("--use_final_checkpoint", action="store_true")
    parser.add_argument("--keep_embeddings", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_split_file(split_type):
    split_to_file = {
        "enzyme_split": CARE_ROOT / "task3" / "data" / "enzyme_split" / "val_pairs.tsv",
        "rxn_sub_split": CARE_ROOT / "task3" / "data" / "rxn_sub_split" / "val_reactions.tsv",
    }
    return split_to_file[split_type]


def get_checkpoint_path(pretrained_folder, prefix, use_final_checkpoint):
    model_suffix = "model_final.pth" if use_final_checkpoint else "model.pth"
    return Path(pretrained_folder) / f"{prefix}_{model_suffix}"


def load_state_dict(module, checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    module.load_state_dict(state_dict)
    return module


def build_protein_model(args, device):
    protein_tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        do_lower_case=False,
        cache_dir=str(CREEP_ROOT / "data" / "pretrained_ProtT5"),
        local_files_only=True,
    )
    protein_model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        cache_dir=str(CREEP_ROOT / "data" / "pretrained_ProtT5"),
        local_files_only=True,
    )
    protein_model = load_state_dict(
        protein_model,
        get_checkpoint_path(args.pretrained_folder, "protein", args.use_final_checkpoint),
    )

    protein2latent_model = nn.Linear(1024, args.ssl_emb_dim)
    protein2latent_model = load_state_dict(
        protein2latent_model,
        get_checkpoint_path(args.pretrained_folder, "protein2latent", args.use_final_checkpoint),
    )

    model = SingleModalityModel(
        protein_model,
        protein2latent_model,
        args.protein_backbone_model,
        "protein",
    )
    model.eval()
    model.to(device)
    return protein_tokenizer, model


def build_reaction_model(args, device):
    rxnfp_dir = CREEP_ROOT / "data" / "pretrained_rxnfp"
    reaction_tokenizer = SmilesTokenizer(str(rxnfp_dir / "vocab.txt"), do_lower_case=False)
    reaction_model = BertModel.from_pretrained(str(rxnfp_dir), local_files_only=True)
    reaction_model = load_state_dict(
        reaction_model,
        get_checkpoint_path(args.pretrained_folder, "reaction", args.use_final_checkpoint),
    )

    reaction2latent_model = nn.Linear(256, args.ssl_emb_dim)
    reaction2latent_model = load_state_dict(
        reaction2latent_model,
        get_checkpoint_path(args.pretrained_folder, "reaction2latent", args.use_final_checkpoint),
    )

    model = SingleModalityModel(
        reaction_model,
        reaction2latent_model,
        args.reaction_backbone_model,
        "reaction",
    )
    model.eval()
    model.to(device)
    return reaction_tokenizer, model


def load_original_entry_limit(enzyme_db_metadata_path):
    with open(enzyme_db_metadata_path) as f:
        enzyme_db_metadata = json.load(f)

    original_entries = enzyme_db_metadata.get("original_entries")
    if original_entries is None:
        raise ValueError(
            f"`original_entries` is missing from enzyme db metadata: {enzyme_db_metadata_path}"
        )
    if not isinstance(original_entries, int) or original_entries < 0:
        raise ValueError(
            f"`original_entries` must be a non-negative integer in {enzyme_db_metadata_path}"
        )
    return original_entries


def load_ordered_enzyme_entries(
    enzyme_db_path,
    enzyme_candidates_mode="extended",
    enzyme_db_metadata_path=None,
    max_enzymes=None,
):
    with open(enzyme_db_path) as f:
        enzyme_db = json.load(f)

    entries = list(enzyme_db.items())
    original_entries = None
    if enzyme_candidates_mode == "original_entries":
        if enzyme_db_metadata_path is None:
            raise ValueError(
                "`enzyme_db_metadata_path` is required when --enzyme_candidates_mode=original_entries"
            )
        original_entries = load_original_entry_limit(enzyme_db_metadata_path)
        if original_entries > len(entries):
            raise ValueError(
                f"Metadata original_entries={original_entries} exceeds enzyme db size={len(entries)}"
            )
        entries = entries[:original_entries]

    if max_enzymes is not None:
        entries = entries[:max_enzymes]
    return entries, original_entries


def load_ordered_reaction_entries(split_file, pair_db_path, preprocessed_dir=None, max_reactions=None):
    split_df = pd.read_csv(split_file, sep="\t", usecols=["rxn_id"])
    ordered_rxn_ids = split_df["rxn_id"].drop_duplicates().tolist()
    if max_reactions is not None:
        ordered_rxn_ids = ordered_rxn_ids[:max_reactions]

    pair_db = _load_or_preprocess_pair_db(pair_db_path, preprocessed_dir=preprocessed_dir)
    valid_pair_db = pair_db.dropna(subset=["reaction"]).copy()

    reaction_conflicts = valid_pair_db.groupby("rxn_id")["reaction"].nunique()
    conflicting_rxns = reaction_conflicts[reaction_conflicts > 1]
    if not conflicting_rxns.empty:
        conflict_preview = conflicting_rxns.index[:5].tolist()
        raise ValueError(f"Multiple reaction strings found for rxn_id(s): {conflict_preview}")

    reaction_lookup = valid_pair_db.drop_duplicates("rxn_id")[["rxn_id", "reaction"]]
    ordered_reactions = pd.DataFrame({"rxn_id": ordered_rxn_ids}).merge(
        reaction_lookup,
        on="rxn_id",
        how="left",
    )

    missing_rxns = ordered_reactions[ordered_reactions["reaction"].isna()]["rxn_id"].tolist()
    if missing_rxns:
        raise ValueError(f"Missing reaction strings for rxn_id(s): {missing_rxns[:10]}")

    return list(zip(ordered_reactions["rxn_id"].tolist(), ordered_reactions["reaction"].tolist()))


def extract_embeddings(
    entries,
    tokenizer,
    model,
    batch_size,
    num_workers,
    max_sequence_len,
    modality,
    device,
    output_path,
    emb_dim,
    verbose=False,
):
    dataset = OrderedSequenceDataset(entries, tokenizer, max_sequence_len, modality)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), emb_dim),
    )

    iterator = tqdm(dataloader, desc=f"encode_{modality}") if verbose else dataloader
    offset = 0
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for batch in iterator:
            sequence_input_ids = batch["sequence_input_ids"].to(device, non_blocking=True)
            sequence_attention_mask = batch["sequence_attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                repr_tensor = model(sequence_input_ids, sequence_attention_mask)
            repr_tensor = F.normalize(repr_tensor.float(), dim=-1)
            batch_repr = repr_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            embeddings[offset : offset + len(batch_repr)] = batch_repr
            offset += len(batch_repr)

    embeddings.flush()
    return output_path


def compute_similarity_matrix(
    reaction_embedding_path,
    protein_embedding_path,
    output_path,
    reaction_chunk_size,
    protein_chunk_size,
    verbose=False,
):
    reaction_repr = np.load(reaction_embedding_path, mmap_mode="r")
    protein_repr = np.load(protein_embedding_path, mmap_mode="r")
    pred_matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(reaction_repr.shape[0], protein_repr.shape[0]),
    )

    row_ranges = range(0, reaction_repr.shape[0], reaction_chunk_size)
    if verbose:
        row_ranges = tqdm(row_ranges, desc="cosine_similarity")

    for row_start in row_ranges:
        row_end = min(row_start + reaction_chunk_size, reaction_repr.shape[0])
        reaction_block = np.asarray(reaction_repr[row_start:row_end], dtype=np.float32)
        for col_start in range(0, protein_repr.shape[0], protein_chunk_size):
            col_end = min(col_start + protein_chunk_size, protein_repr.shape[0])
            protein_block = np.asarray(protein_repr[col_start:col_end], dtype=np.float32)
            pred_matrix[row_start:row_end, col_start:col_end] = np.dot(
                reaction_block,
                protein_block.T,
            )

    pred_matrix.flush()
    return output_path


def write_single_column_tsv(path, column_name, values):
    with open(path, "w") as f:
        f.write(f"{column_name}\n")
        for value in values:
            f.write(f"{value}\n")


def build_metadata(args, split_type, split_file, row_ids, col_ids, pred_path):
    pred_array = np.load(pred_path, mmap_mode="r")
    return {
        "split_type": split_type,
        "split_file": str(split_file),
        "pair_db_path": str(args.pair_db_path),
        "enzyme_db_path": str(args.enzyme_db_path),
        "enzyme_candidates_mode": args.enzyme_candidates_mode,
        "enzyme_db_metadata_path": str(args.enzyme_db_metadata_path),
        "pretrained_folder": str(args.pretrained_folder),
        "checkpoint_variant": "final" if args.use_final_checkpoint else "best",
        "dtype": str(pred_array.dtype),
        "shape": [int(pred_array.shape[0]), int(pred_array.shape[1])],
        "num_reactions": len(row_ids),
        "num_enzymes": len(col_ids),
        "original_entry_limit": (
            load_original_entry_limit(args.enzyme_db_metadata_path)
            if args.enzyme_candidates_mode == "original_entries"
            else None
        ),
        "max_reactions": args.max_reactions,
        "max_enzymes": args.max_enzymes,
    }


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_suffix(enzyme_candidates_mode):
    return "" if enzyme_candidates_mode == "extended" else f"_{enzyme_candidates_mode}"


def main():
    args = parse_args()
    print("arguments", args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.pretrained_folder) / "predictions"
    ensure_dir(output_dir)

    split_types = ["enzyme_split", "rxn_sub_split"] if args.split_type == "all" else [args.split_type]

    enzyme_entries, _ = load_ordered_enzyme_entries(
        args.enzyme_db_path,
        enzyme_candidates_mode=args.enzyme_candidates_mode,
        enzyme_db_metadata_path=args.enzyme_db_metadata_path,
        max_enzymes=args.max_enzymes,
    )
    enzyme_ids = [enzyme_id for enzyme_id, _ in enzyme_entries]
    output_suffix = get_output_suffix(args.enzyme_candidates_mode)
    protein_embedding_path = output_dir / f"_tmp_protein_embeddings{output_suffix}.npy"

    protein_tokenizer, protein_model = build_protein_model(args, device)
    extract_embeddings(
        entries=enzyme_entries,
        tokenizer=protein_tokenizer,
        model=protein_model,
        batch_size=args.batch_size_protein,
        num_workers=args.num_workers,
        max_sequence_len=args.protein_max_sequence_len,
        modality="protein",
        device=device,
        output_path=protein_embedding_path,
        emb_dim=args.ssl_emb_dim,
        verbose=args.verbose,
    )
    cleanup_model(protein_model)

    reaction_tokenizer, reaction_model = build_reaction_model(args, device)
    for split_type in split_types:
        split_file = Path(args.split_file) if args.split_file else get_split_file(split_type)
        reaction_entries = load_ordered_reaction_entries(
            split_file=split_file,
            pair_db_path=args.pair_db_path,
            preprocessed_dir=args.preprocessed_rxn_dir,
            max_reactions=args.max_reactions,
        )
        reaction_ids = [reaction_id for reaction_id, _ in reaction_entries]
        reaction_embedding_path = output_dir / f"_tmp_{split_type}_reaction_embeddings.npy"
        pred_path = output_dir / f"{split_type}{output_suffix}_preds.npy"
        row_path = output_dir / f"{split_type}{output_suffix}_row_rxn_ids.tsv"
        col_path = output_dir / f"{split_type}{output_suffix}_col_enz_ids.tsv"
        metadata_path = output_dir / f"{split_type}{output_suffix}_metadata.json"

        extract_embeddings(
            entries=reaction_entries,
            tokenizer=reaction_tokenizer,
            model=reaction_model,
            batch_size=args.batch_size_reaction,
            num_workers=args.num_workers,
            max_sequence_len=args.reaction_max_sequence_len,
            modality="reaction",
            device=device,
            output_path=reaction_embedding_path,
            emb_dim=args.ssl_emb_dim,
            verbose=args.verbose,
        )
        compute_similarity_matrix(
            reaction_embedding_path=reaction_embedding_path,
            protein_embedding_path=protein_embedding_path,
            output_path=pred_path,
            reaction_chunk_size=args.similarity_reaction_chunk_size,
            protein_chunk_size=args.similarity_protein_chunk_size,
            verbose=args.verbose,
        )

        write_single_column_tsv(row_path, "rxn_id", reaction_ids)
        write_single_column_tsv(col_path, "enz_id", enzyme_ids)
        with open(metadata_path, "w") as f:
            json.dump(
                build_metadata(args, split_type, split_file, reaction_ids, enzyme_ids, pred_path),
                f,
                indent=2,
            )

        if not args.keep_embeddings and reaction_embedding_path.exists():
            reaction_embedding_path.unlink()

    cleanup_model(reaction_model)
    if not args.keep_embeddings and protein_embedding_path.exists():
        protein_embedding_path.unlink()


if __name__ == "__main__":
    main()
