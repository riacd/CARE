import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from Bio import pairwise2
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment # type: ignore

ignore_label = -1

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_pair_ids(rxn_db_dir, enz_db_path, ids_path):
    del rxn_db_dir
    pair_ids = pd.read_csv(
        ids_path,
        sep='\t' if ids_path.endswith('.tsv') else ',',
        usecols=['rxn_id', 'enz_id']
    )
    valid_enz_ids = read_json(enz_db_path).keys()
    pair_ids = pair_ids[pair_ids['enz_id'].isin(valid_enz_ids)]
    return pair_ids

def get_query_ids(train_ids, test_ids, seed=42):
    test_rxn_groups = dict(list(test_ids.groupby('rxn_id', sort=False)))
    train_rxn_groups = dict(list(train_ids.groupby('rxn_id', sort=False)))

    query_ids = []
    for i, (rxn_id, test_rxn_group) in enumerate(test_rxn_groups.items()):
        sample_enzs = test_rxn_group['enz_id']
        if rxn_id in train_rxn_groups:
            train_pos_enzs = train_rxn_groups[rxn_id]['enz_id']
            if len(sample_enzs) == 1:
                sample_enzs = train_pos_enzs
            else:
                sample_enzs = sample_enzs.iloc[:int(len(train_pos_enzs) * 0.25)]
                sample_enzs = pd.concat([train_pos_enzs, sample_enzs])
        elif len(sample_enzs) < 2:
            continue
        else:
            unseen_enzs = sample_enzs[~sample_enzs.isin(train_ids['enz_id'])]
            if len(unseen_enzs) > 0:
                sample_enzs = unseen_enzs
        query_id = sample_enzs.sample(n=1, random_state=seed).iloc[0]
        query_ids.append({'rxn_id': rxn_id, 'enz_id': query_id, 'rxn_idx': i})

    query_ids = pd.DataFrame(query_ids)
    test_ids = test_ids[test_ids['rxn_id'].isin(query_ids['rxn_id'])]
    print(f'Number of test reactions with multiple enzymes: {len(query_ids)}')
    return test_ids, query_ids

def get_eval_labels(eval_rxn_enz_ids, eval_rxn_ids, eval_enz_ids, train_rxn_enz_ids=None):
    rxn_id_to_idx = {rxn_id: idx for idx, rxn_id in enumerate(eval_rxn_ids)}
    rxn_indices = torch.from_numpy(eval_rxn_enz_ids['rxn_id'].map(rxn_id_to_idx).values)
    enz_id_to_idx = {enz_id: idx for idx, enz_id in enumerate(eval_enz_ids)}
    enz_indices = torch.from_numpy(eval_rxn_enz_ids['enz_id'].map(enz_id_to_idx).values)

    labels = torch.zeros((len(rxn_id_to_idx), len(enz_id_to_idx)), dtype=torch.float)
    labels[rxn_indices, enz_indices] = 1

    if train_rxn_enz_ids is not None:
        train_rxn_enz_ids = train_rxn_enz_ids[train_rxn_enz_ids['rxn_id'].isin(rxn_id_to_idx.keys())]
        rxn_indices = torch.from_numpy(train_rxn_enz_ids['rxn_id'].map(rxn_id_to_idx).values)
        enz_indices = torch.from_numpy(train_rxn_enz_ids['enz_id'].map(enz_id_to_idx).values)
        labels[rxn_indices, enz_indices] = ignore_label

    return labels, rxn_id_to_idx, enz_id_to_idx

def retrieval_metrics(preds, labels, k=20, num_pos=None, reduce=True, ignore_index=-100):
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)

    label_mask = labels != ignore_index
    preds = preds.where(label_mask, -1000)
    labels = labels.where(label_mask, 0)

    indices = preds.topk(k, dim=1).indices
    topk_labels = labels.gather(dim=1, index=indices)

    num_matches = topk_labels.sum(1)
    num_pos = labels.sum(1) if num_pos is None else num_pos.to(preds.device)
    success_rate = (num_matches > 0).float()
    precision = num_matches / k
    recall = num_matches / num_pos
    if reduce:
        success_rate = success_rate.mean().item()
        precision = precision.mean().item()
        recall = recall.mean().item()

    return {
        f'sr@{k}': success_rate,
        f'acc@{k}': precision,
        f'recall@{k}': recall
    }

def screening_metrics(preds, labels, alpha=85, fraction=0.02, reduce=True, ignore_index=-100):
    if type(preds) is not torch.Tensor:
        preds = torch.from_numpy(preds)
    if type(labels) is not torch.Tensor:
        labels = torch.from_numpy(labels)
    labels = labels.to(preds.device)

    label_mask = labels != ignore_index
    preds = preds.where(label_mask, -1000)

    preds, indices = preds.sort(dim=1, descending=True)
    labels = labels.gather(dim=1, index=indices)

    bedroc, ef = [], []
    scores = torch.stack([preds.cpu(), labels.cpu()], dim=-1).double().numpy()
    num_cdts = label_mask.sum(1).tolist()
    for i, n in enumerate(num_cdts):
        bedroc.append(CalcBEDROC(scores[i][:n], col=1, alpha=alpha))
        ef.append(CalcEnrichment(scores[i][:n], col=1, fractions=[fraction])[0])
    if reduce:
        bedroc = sum(bedroc) / preds.size(0)
        ef = sum(ef) / preds.size(0)

    return {
        f'bedroc@{alpha}': bedroc,
        f'ef@{fraction}': ef
    }

def pairwise_identity(seq1, seq2):
    best_aln = pairwise2.align.globalxx(seq1, seq2)[0]
    matches = sum(a == b for a, b in zip(best_aln.seqA, best_aln.seqB))
    return matches / len(best_aln.seqA)

def get_max_rxn_sims(train_ids, test_ids):
    all_data = pd.read_pickle('data/pair_merged_data/rxn_uniprot_pair_merged_data_washed_with_drfp.pkl')
    all_drfps = all_data.set_index('rxn_id')['DRFP']
    train_drfps = all_drfps[train_ids['rxn_id'].unique()]
    train_drfps = torch.from_numpy(np.array(train_drfps.tolist())).float()
    test_drfps = all_drfps[test_ids['rxn_id'].unique()]
    test_drfps = torch.from_numpy(np.array(test_drfps.tolist())).float()

    # compute cosine similarity
    train_drfps /= (train_drfps.norm(dim=1, keepdim=True) + 1e-8)
    test_drfps /= (test_drfps.norm(dim=1, keepdim=True) + 1e-8)
    sims = test_drfps @ train_drfps.T
    max_rxn_sims = sims.max(dim=1).values
    return max_rxn_sims

def get_max_pos_sims(test_ids, test_query_ids, enz_db):
    ref_ids = test_query_ids.set_index('rxn_id')['enz_id']
    max_pos_sims = []
    for rxn_id, rxn_group in test_ids.groupby('rxn_id', sort=False):
        ref_id = ref_ids[rxn_id]
        pos_ids = rxn_group['enz_id']
        pos_ids = pos_ids[pos_ids != ref_id]
        max_pos_sim = max([pairwise_identity(enz_db[ref_id], enz_db[pos_id]) for pos_id in pos_ids])
        max_pos_sims.append(max_pos_sim)
    return torch.tensor(max_pos_sims)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn_db_dir', '-rdb', type=str, default='data/all_reactions')
    parser.add_argument('--enz_db_path', '-edb', type=str, default='data/pair_merged_data/enzyme_db.json')
    parser.add_argument('--train_ids_path', '-tid', type=str, default='data/pair_merged_data/rxn_sub_split/train_reactions.tsv')
    parser.add_argument('--test_ids_path', '-sid', type=str, default='data/pair_merged_data/rxn_sub_split/val_reactions.tsv')
    parser.add_argument('--pred_path', '-pp', type=str, default=None)
    parser.add_argument('--custom_labels', '-cl', type=str, default=None)
    parser.add_argument('--ref_enzymes', '-ref', action='store_true')
    parser.add_argument('--query_seed', '-qs', type=int, default=42)
    parser.add_argument('--thresholds', '-t', nargs='+', type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument('--score_path', '-sp', type=str, default='predictions/rxn_sub_split_scores.json')
    parser.add_argument('--overwrite', '-o', action='store_true')
    return parser.parse_args()

# python -m scripts.screening.screen_scores -tid train_ids路径 -sid test_ids路径 -pp preds矩阵路径 -t 1 -sp predictions/temp.json
# 用上面个这个命令来算分
# 使用 Protein_CUDA121 环境，注意检查环境中 python 解释器链接正确，
# -pp /mnt/shared-storage-user/huyutong/CARE/data/inference_preds_bs40/enzyme_split_original_entries_preds.npy \
# -tid /mnt/shared-storage-user/huyutong/CARE/data/enzyme_split/train_pairs.tsv \
# -sid /mnt/shared-storage-user/huyutong/CARE/data/enzyme_split/val_pairs.tsv \
# -t 1 \
# -sp /mnt/shared-storage-user/huyutong/CARE/tmp.json

if __name__ == '__main__':
    args = parse_args()
    train_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.train_ids_path)
    test_ids = get_pair_ids(args.rxn_db_dir, args.enz_db_path, args.test_ids_path)
    enz_db = read_json(args.enz_db_path)

    if args.ref_enzymes:
        test_ids, test_query_ids = get_query_ids(train_ids, test_ids, seed=args.query_seed)

    if args.custom_labels:
        eval_labels = torch.load(args.custom_labels)
        eval_labels = eval_labels.to_dense().float()
        if args.ref_enzymes:
            eval_labels = eval_labels[torch.from_numpy(test_query_ids['rxn_idx'].values)]
            rxn_id_to_idx = {rxn_id: idx for idx, rxn_id in enumerate(test_query_ids['rxn_id'])}
            enz_id_to_idx = {enz_id: idx for idx, enz_id in enumerate(enz_db.keys())}
    else:
        eval_labels, rxn_id_to_idx, enz_id_to_idx = get_eval_labels(
            eval_rxn_enz_ids=test_ids,
            eval_rxn_ids=test_ids['rxn_id'].unique(),
            eval_enz_ids=list(enz_db.keys()),
            train_rxn_enz_ids=train_ids
        )
    
    if args.ref_enzymes:
        test_rxn_indices = torch.from_numpy(test_query_ids['rxn_id'].map(rxn_id_to_idx).values)
        test_query_indices = torch.from_numpy(test_query_ids['enz_id'].map(enz_id_to_idx).values)
        eval_labels[test_rxn_indices, test_query_indices] = ignore_label
    
    sim_path = os.path.join(
        os.path.dirname(args.test_ids_path),
        'max_pos_sims.pkl' if args.ref_enzymes else 'max_rxn_sims.pkl'
    )
    expected_sim_len = len(test_query_ids) if args.ref_enzymes else test_ids['rxn_id'].nunique()
    if os.path.exists(sim_path):
        sims = torch.load(sim_path)
        if len(sims) != expected_sim_len:
            print(f'Cached similarities in {sim_path} have length {len(sims)}; expected {expected_sim_len}. Recomputing...')
            if args.ref_enzymes:
                print('Computing template-positive enzyme similarities...')
                sims = get_max_pos_sims(test_ids, test_query_ids, enz_db)
            else:
                print('Computing train-test reaction similarities...')
                sims = get_max_rxn_sims(train_ids, test_ids)
            torch.save(sims, sim_path)
    else:
        if args.ref_enzymes:
            print('Computing template-positive enzyme similarities...')
            sims = get_max_pos_sims(test_ids, test_query_ids, enz_db)
        else:
            print('Computing train-test reaction similarities...')
            sims = get_max_rxn_sims(train_ids, test_ids)
        torch.save(sims, sim_path)
    
    device = 'cuda' if torch.cuda.is_available() and '_extended' not in args.enz_db_path else 'cpu'
    eval_preds = torch.load(args.pred_path).detach().to(device)
    eval_labels = eval_labels.to(device)
    sims = sims.to(device)
    
    # compute and save scores
    if os.path.exists(args.score_path) and not args.overwrite:
        all_scores = read_json(args.score_path)
    else: 
        all_scores = {'rxn_screen': {}, 'ref_screen': {}}
    
    scores = {}
    for threshold in args.thresholds:
        mask = sims <= threshold + 1e-4 # slightly loosen the threshold
        masked_labels = eval_labels[mask]
        if masked_labels.size(0) == 0:
            print(f'No reaction under similarity threshold {threshold}. Skipping...')
            continue
        print(f'\n======= Computing scores under similarity threshold: {threshold} =======')
        print(f'Number of reactions: {masked_labels.size(0)}')
        masked_preds = eval_preds[mask]
    
        scores[threshold] = {'rxn_num': masked_labels.size(0)}
        for k in (1, 3, 5, 10, 20):
            metrics = retrieval_metrics(masked_preds, masked_labels, k=k, ignore_index=ignore_label)
            print('\t'.join([f'{name}: {value * 100:.2f}%' for name, value in metrics.items()]))
            scores[threshold].update(metrics)

        metrics = screening_metrics(masked_preds, masked_labels, ignore_index=ignore_label)
        for name, value in metrics.items():
            if name.startswith('bedroc'):
                print(f'{name}: {value * 100:.2f}%')
            else:
                print(f'{name}: {value:.2f}')
        scores[threshold].update(metrics)
    
    screen_name = 'ref_screen' if args.ref_enzymes else 'rxn_screen'
    all_scores[screen_name][args.pred_path] = scores
    write_json(all_scores, args.score_path)
