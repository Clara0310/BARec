import numpy as np
from multiprocessing import Process, Queue

def random_neq(l, r, s): #maybe redundant items
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(del_num, user_train, origin_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # === [FIX START] 修復 KeyError ===
    # 建立有效使用者清單，避免抽到不存在的 ID
    valid_users = [u for u in user_train.keys() if len(user_train[u]) > 1]
    # === [FIX END] ===

    def sample():
        # === [FIX START] 從有效清單中抽樣 ===
        user = np.random.choice(valid_users)
        # === [FIX END] ===
        
        # aug/original seq
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1] # label
        idx = maxlen - 1
        ts = set(user_train[user])

        inter_len = len(user_train[user][:-1]) - del_num if (len(user_train[user][:-1]) - del_num) > 0 else 1 
        inter_len_revs = len(user_train[user][1:]) - del_num if (len(user_train[user][1:]) - del_num) > 0 else 1
        inter_len_origin = len(origin_train[user][:-1]) - del_num if (len(origin_train[user][:-1]) - del_num) > 0 else 1

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i

            if inter_len > 0:
                pos[idx] = nxt # label
                if nxt != 0: 
                    neg[idx] = random_neq(1, itemnum + 1, ts)
                inter_len -= 1

            nxt = i
            idx -= 1
            if idx == -1: break
                
        # seq_revs, pos_revs, neg_revs==neg
        seq_revs = np.zeros([maxlen], dtype=np.int32)
        pos_revs = np.zeros([maxlen], dtype=np.int32)  
        nxt_revs = user_train[user][0] # label
        idx_revs = maxlen - 1

        for i in user_train[user][1:]:
            seq_revs[idx_revs] = i

            if inter_len_revs > 0:
                pos_revs[idx_revs] = nxt_revs #label
                inter_len -= 1
                
            nxt_revs = i
            idx_revs -= 1
            if idx_revs == -1: break

        # original seq
        seq_origin = np.zeros([maxlen], dtype=np.int32)
        pos_origin = np.zeros([maxlen], dtype=np.int32)
        neg_origin = np.zeros([maxlen], dtype=np.int32)
        nxt_origin = origin_train[user][-1] # label
        idx_origin = maxlen - 1
        ts_origin = set(origin_train[user])

        for i in reversed(origin_train[user][:-1]):
            seq_origin[idx_origin] = i

            if inter_len_origin > 0:
                pos_origin[idx_origin] = nxt_origin # label
                if nxt_origin != 0: 
                    neg_origin[idx_origin] = random_neq(1, itemnum + 1, ts_origin)
                inter_len_origin -= 1
                
            nxt_origin = i
            idx_origin -= 1
            if idx_origin == -1: break

        return (user, seq, pos, neg, seq_revs, pos_revs, seq_origin, pos_origin, neg_origin)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, del_num, User, Origin, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                    Process(target=sample_function, args=(del_num, 
                                                          User, Origin,
                                                          usernum,
                                                          itemnum,
                                                          batch_size,
                                                          maxlen,
                                                          self.result_queue,
                                                          np.random.randint(2e9)
                                                         )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()