import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from transformers import BertModel, T5EncoderModel, T5Tokenizer

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

try:
    import wandb
except ImportError:
    wandb = None

from CREEP.datasets import Task3CREEPDataset, Task3PairBatchSampler
from CREEP.models import AEFacilitatorModel, CREEPModel
from CREEP.utils.tokenization import SmilesTokenizer


class Logger:
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_cl(x, y, args):
    if args.normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

    if args.cl_loss == "EBM_NCE":
        criterion = nn.BCEWithLogitsLoss()
        neg_y = torch.cat([y[cycle_index(len(y), i + 1)] for i in range(args.cl_neg_samples)], dim=0)
        neg_x = x.repeat((args.cl_neg_samples, 1))

        pred_pos = torch.sum(x * y, dim=1) / args.temperature
        pred_neg = torch.sum(neg_x * neg_y, dim=1) / args.temperature

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos), device=pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg), device=pred_neg.device))
        cl_loss = (loss_pos + args.cl_neg_samples * loss_neg) / (1 + args.cl_neg_samples)
        cl_acc = (
            torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()
        ) / (len(pred_pos) + len(pred_neg))
        cl_acc = cl_acc.detach().cpu().item()
    elif args.cl_loss == "InfoNCE":
        criterion = nn.CrossEntropyLoss()
        batch_size = x.size(0)
        logits = torch.mm(x, y.transpose(1, 0)) / args.temperature
        labels = torch.arange(batch_size, device=logits.device).long()
        cl_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        cl_acc = pred.eq(labels).sum().detach().cpu().item() / batch_size
    else:
        raise ValueError(f"Unsupported CL loss: {args.cl_loss}")

    return cl_loss, cl_acc


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(dataloader, model, optimizer, scaler, device, args, training):
    if training:
        model.train()
    else:
        model.eval()

    iterator = tqdm(dataloader) if args.verbose else dataloader
    mse_loss = nn.MSELoss()
    amp_enabled = device.type == "cuda"
    start_time = time.time()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_generative_loss = 0.0
    total_acc = 0.0

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        protein_input_ids = batch["protein_sequence_input_ids"]
        protein_attention_mask = batch["protein_sequence_attention_mask"]
        reaction_input_ids = batch["reaction_sequence_input_ids"]
        reaction_attention_mask = batch["reaction_sequence_attention_mask"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                protein_repr, _, reaction_repr, reaction2protein_repr, protein2reaction_repr = model(
                    protein_sequence_input_ids=protein_input_ids,
                    protein_sequence_attention_mask=protein_attention_mask,
                    reaction_sequence_input_ids=reaction_input_ids,
                    reaction_sequence_attention_mask=reaction_attention_mask,
                )

                loss_pr, acc_pr = do_cl(protein_repr, reaction_repr, args)
                loss_rp, acc_rp = do_cl(reaction_repr, protein_repr, args)
                contrastive_loss = (loss_pr + loss_rp) / 2
                contrastive_acc = (acc_pr + acc_rp) / 2

                generative_loss = mse_loss(reaction2protein_repr, protein_repr) + mse_loss(
                    protein2reaction_repr,
                    reaction_repr,
                )
                loss = (
                    args.alpha_contrastive * contrastive_loss
                    + args.alpha_generative * generative_loss
                )

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_generative_loss += generative_loss.item()
        total_acc += contrastive_acc

    num_batches = max(1, len(dataloader))
    metrics = {
        "loss": total_loss / num_batches,
        "contrastive_loss": total_contrastive_loss / num_batches,
        "generative_loss": total_generative_loss / num_batches,
        "contrastive_acc": total_acc / num_batches,
        "epoch_time_sec": time.time() - start_time,
    }
    return metrics


def build_dataloader(dataset, batch_size, num_batches, num_workers, seed, epoch):
    sampler = Task3PairBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        shuffle=True,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def save_model(args, model_parts, save_best):
    if args.output_model_dir is None:
        return

    model_file = "model.pth" if save_best else "model_final.pth"
    for prefix, module in model_parts.items():
        torch.save(module.state_dict(), os.path.join(args.output_model_dir, f"{prefix}_{model_file}"))


def init_wandb(args):
    if wandb is None:
        print("wandb is not installed; skipping wandb logging.")
        return None

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=args.output_model_dir,
        config=vars(args),
        mode=args.wandb_mode,
    )


def dump_stats(output_dir, split_name, stats):
    stats_path = Path(output_dir) / f"{split_name}_dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ssl_emb_dim", type=int, default=256)
    parser.add_argument("--split_type", type=str, default="enzyme_split", choices=["enzyme_split", "rxn_sub_split"])
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--pair_db_path", type=str, default=str(CARE_ROOT / "task3" / "data" / "pair_merged_data" / "all_pair_data.tsv"))
    parser.add_argument("--enzyme_db_path", type=str, default=str(CARE_ROOT / "task3" / "data" / "pair_merged_data" / "enzyme_db_extended.json"))
    parser.add_argument("--protein_backbone_model", type=str, default="ProtT5", choices=["ProtT5"])
    parser.add_argument("--text_backbone_model", type=str, default="SciBERT")
    parser.add_argument("--reaction_backbone_model", type=str, default="rxnfp")
    parser.add_argument("--protein_max_sequence_len", type=int, default=512)
    parser.add_argument("--reaction_max_sequence_len", type=int, default=512)
    parser.add_argument("--protein_lr", type=float, default=1e-5)
    parser.add_argument("--protein_lr_scale", type=float, default=1.0)
    parser.add_argument("--reaction_lr", type=float, default=1e-5)
    parser.add_argument("--reaction_lr_scale", type=float, default=1.0)
    parser.add_argument("--cl_neg_samples", type=int, default=1)
    parser.add_argument("--cl_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--alpha_contrastive", type=float, default=1.0)
    parser.add_argument("--alpha_generative", type=float, default=0.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_batches_per_epoch", type=int, default=5000)
    parser.add_argument("--val_num_batches", type=int, default=500)
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="care-task3-creep")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    if args.train_file is None:
        args.train_file = str(CARE_ROOT / "task3" / "data" / args.split_type / ("train_pairs.tsv" if args.split_type == "enzyme_split" else "train_reactions.tsv"))
    if args.val_file is None:
        args.val_file = str(CARE_ROOT / "task3" / "data" / args.split_type / ("val_pairs.tsv" if args.split_type == "enzyme_split" else "val_reactions.tsv"))
    if args.wandb_run_name is None:
        args.wandb_run_name = f"creep-task3-{args.split_type}"

    os.makedirs(args.output_model_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_model_dir, "log.txt"), "w")
    print("arguments", args)

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

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
    protein_dim = 1024

    text_model = AutoModel.from_pretrained(
        "allenai/scibert_scivocab_uncased",
        cache_dir=str(CREEP_ROOT / "data" / "pretrained_SciBert"),
        local_files_only=True,
    )
    text_dim = 768

    reaction_tokenizer = SmilesTokenizer.from_pretrained(str(CREEP_ROOT / "data" / "pretrained_rxnfp" / "vocab.txt"))
    reaction_model = BertModel.from_pretrained(str(CREEP_ROOT / "data" / "pretrained_rxnfp"))
    reaction_dim = 256

    protein2latent_model = nn.Linear(protein_dim, args.ssl_emb_dim)
    text2latent_model = nn.Linear(text_dim, args.ssl_emb_dim)
    reaction2latent_model = nn.Linear(reaction_dim, args.ssl_emb_dim)
    reaction2protein_facilitator_model = AEFacilitatorModel(args.ssl_emb_dim)
    protein2reaction_facilitator_model = AEFacilitatorModel(args.ssl_emb_dim)

    model = CREEPModel(
        protein_model,
        text_model,
        reaction_model,
        protein2latent_model,
        text2latent_model,
        reaction2latent_model,
        reaction2protein_facilitator_model,
        protein2reaction_facilitator_model,
        args.protein_backbone_model,
        args.text_backbone_model,
        args.reaction_backbone_model,
    )
    model.to(device)

    train_dataset = Task3CREEPDataset(
        split_file=args.train_file,
        pair_db_path=args.pair_db_path,
        enzyme_db_path=args.enzyme_db_path,
        protein_tokenizer=protein_tokenizer,
        reaction_tokenizer=reaction_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        reaction_max_sequence_len=args.reaction_max_sequence_len,
    )
    dump_stats(args.output_model_dir, "train", train_dataset.stats)
    print("train dataset stats", train_dataset.stats)

    val_dataset = None
    if args.val_file:
        val_dataset = Task3CREEPDataset(
            split_file=args.val_file,
            pair_db_path=args.pair_db_path,
            enzyme_db_path=args.enzyme_db_path,
            protein_tokenizer=protein_tokenizer,
            reaction_tokenizer=reaction_tokenizer,
            protein_max_sequence_len=args.protein_max_sequence_len,
            reaction_max_sequence_len=args.reaction_max_sequence_len,
        )
        dump_stats(args.output_model_dir, "val", val_dataset.stats)
        print("val dataset stats", val_dataset.stats)

    model_param_group = [
        {"params": protein_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": reaction_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": protein2latent_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
        {"params": reaction2latent_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": reaction2protein_facilitator_model.parameters(), "lr": args.reaction_lr * args.reaction_lr_scale},
        {"params": protein2reaction_facilitator_model.parameters(), "lr": args.protein_lr * args.protein_lr_scale},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model_parts = {
        "text": text_model,
        "protein": protein_model,
        "reaction": reaction_model,
        "text2latent": text2latent_model,
        "protein2latent": protein2latent_model,
        "reaction2latent": reaction2latent_model,
        "reaction2protein_facilitator": reaction2protein_facilitator_model,
        "protein2reaction_facilitator": protein2reaction_facilitator_model,
    }

    wandb_run = init_wandb(args)
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_batches=args.num_batches_per_epoch,
            num_workers=args.num_workers,
            seed=args.seed,
            epoch=epoch,
        )
        train_metrics = run_epoch(train_loader, model, optimizer, scaler, device, args, training=True)
        print(
            "train "
            f"CL Loss: {train_metrics['contrastive_loss']:.5f}\t"
            f"CL Acc: {train_metrics['contrastive_acc']:.5f}\t"
            f"Generative Loss: {train_metrics['generative_loss']:.5f}\t"
            f"Total Loss: {train_metrics['loss']:.5f}\t"
            f"Time: {train_metrics['epoch_time_sec']:.5f}"
        )

        val_metrics = None
        if val_dataset is not None:
            eval_batches = min(args.val_num_batches, max(1, len(val_dataset.pairs) // args.batch_size))
            val_loader = build_dataloader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=eval_batches,
                num_workers=args.num_workers,
                seed=args.seed + 10_000,
                epoch=epoch,
            )
            val_metrics = run_epoch(val_loader, model, optimizer, scaler, device, args, training=False)
            print(
                "val "
                f"CL Loss: {val_metrics['contrastive_loss']:.5f}\t"
                f"CL Acc: {val_metrics['contrastive_acc']:.5f}\t"
                f"Generative Loss: {val_metrics['generative_loss']:.5f}\t"
                f"Total Loss: {val_metrics['loss']:.5f}\t"
                f"Time: {val_metrics['epoch_time_sec']:.5f}"
            )

        current_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        if current_metric < best_metric:
            best_metric = current_metric
            save_model(args, model_parts, save_best=True)

        save_model(args, model_parts, save_best=False)

        log_payload = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/contrastive_loss": train_metrics["contrastive_loss"],
            "train/generative_loss": train_metrics["generative_loss"],
            "train/contrastive_acc": train_metrics["contrastive_acc"],
            "train/epoch_time_sec": train_metrics["epoch_time_sec"],
            "lr/protein": optimizer.param_groups[0]["lr"],
            "lr/reaction": optimizer.param_groups[1]["lr"],
        }
        if val_metrics is not None:
            log_payload.update(
                {
                    "val/loss": val_metrics["loss"],
                    "val/contrastive_loss": val_metrics["contrastive_loss"],
                    "val/generative_loss": val_metrics["generative_loss"],
                    "val/contrastive_acc": val_metrics["contrastive_acc"],
                    "val/epoch_time_sec": val_metrics["epoch_time_sec"],
                }
            )
        if wandb_run is not None:
            wandb.log(log_payload)

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
