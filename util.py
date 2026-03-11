import sys
import copy
import random
import numpy as np
import multiprocessing
import time
import os
from collections import defaultdict
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

Ks = [5, 10, 20]
cores = 1 

# 1. set_color
def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try: index = color_set.index(color)
    except: index = len(color_set) - 1
    prev_log = '\033[1;3' if highlight else '\033[0;3'
    return prev_log + str(index) + 'm' + log + '\033[0m'

# 2. record_loss
def record_loss(loss_file, loss_left, kl_loss, loss):
    with open(loss_file, 'a') as f:
        line = "{:.4f}\t{:.4f}\t{:.4f}\n".format(loss_left, kl_loss, loss)
        f.write(line) 

# 3. load_file_and_sort
def load_file_and_sort(filename, reverse=False, augdata=None, aug_num=0, M=10):
    data = defaultdict(list)
    max_uind, max_iind = 0, 0
    with open(filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            max_uind = max(max_uind, uind)
            max_iind = max(max_iind, iind)
            t = float(one_interaction[2])
            data[uind].append((iind, t))
    if augdata:
        for u, ilist in augdata.items():
            sorted_interactions = sorted(ilist, key=lambda x:x[1])
            for i in range(min(aug_num, len(sorted_interactions))):
                if len(data[u]) >= M: continue
                data[u].append((sorted_interactions[i]))
    sorted_data = {}
    for u, i_list in data.items():
        sorted_interactions = sorted(i_list, key=lambda x:x[1], reverse=reverse)
        sorted_data[u] = [it[0] for it in sorted_interactions]
    return sorted_data, max_uind, max_iind

# 4. augdata_load
def augdata_load(aug_filename):
    augdata = defaultdict(list)
    with open(aug_filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            augdata[int(one_interaction[0]) + 1].append((int(one_interaction[1]) + 1, float(one_interaction[2])))
    return augdata

# 5. data_load
def data_load(data_name, args):
    reverseornot = args.reversed == 1
    suffix = "_reverse.txt" if reverseornot else ".txt"
    train_file = f"./data/{data_name}/train{suffix}"
    valid_file = f"./data/{data_name}/valid{suffix}"
    test_file = f"./data/{data_name}/test{suffix}"

    original_train, _, _ = load_file_and_sort(train_file)
    augdata = None
    if args.aug_traindata > 0:
        aug_data_signature = './aug_data/{}/lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}_gen_num_'.format(
                               args.dataset, args.lr, args.maxlen, args.hidden_units, args.num_blocks, args.dropout_rate, args.l2_emb, args.num_heads)
        if os.path.exists(aug_data_signature + '20_M_20.txt'):
            augdata = augdata_load(aug_data_signature + '20_M_20.txt')

    if args.aug_traindata > 0:
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot, augdata=augdata, aug_num=args.aug_traindata, M=args.M)
    else:
        user_train, train_usernum, train_itemnum = load_file_and_sort(train_file, reverse=reverseornot)
    user_valid, valid_usernum, valid_itemnum = load_file_and_sort(valid_file, reverse=reverseornot)
    user_test, test_usernum, test_itemnum = load_file_and_sort(test_file, reverse=reverseornot)
    usernum = max([train_usernum, valid_usernum, test_usernum])
    itemnum = max([train_itemnum, valid_itemnum, test_itemnum])
    return [user_train, user_valid, user_test, original_train, usernum, itemnum]

# 6. data_augment
def data_augment(model, dataset, args, sess, gen_num):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    all_users = list(train.keys())
    cumulative_preds = defaultdict(list)
    items_idx_set = set([i for i in range(itemnum)])
    for num_ind in range(gen_num):
        batch_seq, batch_u, batch_item_idx = [], [], []
        for u_ind, u in enumerate(tqdm(all_users, total=len(all_users), ncols=100, desc=set_color(f"Gen {num_ind+1}/{gen_num}", 'green'))):
            u_data = train.get(u, []) + valid.get(u, []) + test.get(u, []) + cumulative_preds.get(u, [])
            if len(u_data) == 0 or len(u_data) >= args.M: continue
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(u_data):
                if idx == -1: break
                seq[idx], idx = i, idx - 1
            rated = set(u_data)
            item_idx = list(items_idx_set - rated)
            batch_seq.append(seq); batch_item_idx.append(item_idx); batch_u.append(u)
            if len(batch_u) >= args.batch_size:
                predictions = model.predict(sess, batch_u, batch_seq)
                for batch_ind in range(len(batch_item_idx)):
                    test_item_idx = batch_item_idx[batch_ind]
                    ranked_items_ind = predictions[batch_ind][test_item_idx].argsort()
                    rankeditem_oneuserids = int(test_item_idx[ranked_items_ind[-1]])
                    cumulative_preds[batch_u[batch_ind]].append(rankeditem_oneuserids) 
                batch_seq, batch_item_idx, batch_u = [], [], []
        # flush remaining batch after iterating all users
        if len(batch_u) > 0:
            predictions = model.predict(sess, batch_u, batch_seq)
            for batch_ind in range(len(batch_item_idx)):
                test_item_idx = batch_item_idx[batch_ind]
                ranked_items_ind = predictions[batch_ind][test_item_idx].argsort()
                rankeditem_oneuserids = int(test_item_idx[ranked_items_ind[-1]])
                cumulative_preds[batch_u[batch_ind]].append(rankeditem_oneuserids)
            batch_seq, batch_item_idx, batch_u = [], [], []
    return cumulative_preds

# 7. eval_one_interaction (修正 recall 呼叫方式)
def eval_one_interaction(x):
    results = init_metrics()
    rankeditems = np.array(x[0]) # 排序後的索引列表
    test_ind = x[1]              # 正樣本索引 (固定為 0)
    scale_pred = x[2]            # 預測分數
    
    # 建立二元相關性向量 r (用於 precision, ndcg, hit, mrr)
    r = np.zeros_like(rankeditems)
    r[rankeditems == test_ind] = 1
    
    # 根據 metrics.py 中的定義計算指標
    for ind_k in range(len(Ks)):
        k = Ks[ind_k]
        results["precision"][ind_k] += precision_at_k(r, k)
        # 修正：依照 metrics.py 的 recall(rank, ground_truth, N) 呼叫
        results["recall"][ind_k] += recall(rankeditems, [test_ind], k)
        results["ndcg"][ind_k] += ndcg_at_k(r, k, 1)
        results["hit_ratio"][ind_k] += hit_at_k(r, k)
    
    # AUC 計算需要完整的 Ground Truth 標籤
    gd_prob = np.zeros_like(scale_pred)
    gd_prob[test_ind] = 1
    results["auc"] += auc(gd_prob, scale_pred)
    results["mrr"] += mrr(r)
    return results

# 8. rank_corrected
def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1: continue
        corrected_r[corrected_rank] = 1
    return corrected_r

# 9. init_metrics
def init_metrics():
    return {"precision": np.zeros(len(Ks)), "recall": np.zeros(len(Ks)), "ndcg": np.zeros(len(Ks)), "hit_ratio": np.zeros(len(Ks)), "auc": 0., "mrr": 0.}

# 10. evaluate
def evaluate(model, dataset, args, sess, testorvalid):
    [train, valid, test, original_train, usernum, itemnum] = copy.deepcopy(dataset)
    results = init_metrics()
    eval_data = test if testorvalid == "test" else valid
    all_predictions_results_output = []
    batch_seq, batch_u, batch_item_idx = [], [], []
    u_ind, items_idx_set = 0, set(range(1, itemnum + 1))
    
    for u, i_list in tqdm(eval_data.items(), desc=set_color(f"Eval {testorvalid}", 'cyan'), ncols=100):
        if len(train[u]) < 1 or len(eval_data[u]) < 1: continue
        rated = set(train[u]) | {0}
        if testorvalid == "test": rated |= set(valid.get(u, []))
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if testorvalid == "test" and u in valid:
            for i in reversed(valid[u]):
                if idx == -1: break
                seq[idx], idx = i, idx - 1
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx], idx = i, idx - 1
        
        pos_item = i_list[0]
        item_idx = [pos_item]
        if args.evalnegsample == -1:
            item_idx += list(items_idx_set - rated - {pos_item})
        else:
            candidates = list(items_idx_set - rated - {pos_item})
            item_idx += list(np.random.choice(candidates, min(len(candidates), args.evalnegsample), replace=False))
        
        batch_seq.append(seq); batch_item_idx.append(item_idx); batch_u.append(u)
        u_ind += 1

        # 優化：Batch 預測與立即處理
        if len(batch_u) >= 10 or u_ind == len(eval_data):
            predictions = model.predict(sess, batch_u, batch_seq)
            for b_i in range(predictions.shape[0]):
                user_id, curr_item_idx = batch_u[b_i], batch_item_idx[b_i]
                user_preds = predictions[b_i][curr_item_idx]
                
                # 排序 (降序)
                ranked_relative_idx = user_preds.argsort()[::-1]
                
                # 指標計算
                one_res = eval_one_interaction((ranked_relative_idx, 0, user_preds))
                for k in ["precision", "recall", "ndcg", "hit_ratio"]: results[k] += one_res[k]
                results["auc"] += one_res["auc"]; results["mrr"] += one_res["mrr"]
                
                # 僅儲存 Top-100 減少輸出負擔
                all_predictions_results_output.append({
                    "u_ind": int(user_id), "u_pos_gd": int(pos_item), 
                    "predicted": [int(curr_item_idx[idx]) for idx in ranked_relative_idx[:100]]
                })
            batch_seq, batch_u, batch_item_idx = [], [], []

    if u_ind > 0:
        for k in results: results[k] /= u_ind
    return results, None, None, None, None, None, all_predictions_results_output