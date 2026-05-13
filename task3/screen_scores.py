import os
import argparse
import torch
import pandas as pd
import numpy as np
from Bio import pairwise2
from rxnzyme.data.datasets.base import ignore_label
from rxnzyme.data.modules.prorxn import get_eval_labels
from rxnzyme.utils import read_json, write_json, retrieval_metrics, screening_metrics
from .rxnzyme_eval import get_query_ids, get_pair_ids

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
# preds 矩阵路径为 ‘baseline/CARE/data/inference_preds_bs40/rxn_sub_split_original_entries_preds.npy’
# train, test id 在目录：baseline/CARE/data/rxn_sub_split

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
