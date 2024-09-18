import scipy.io
import torch
import numpy as np
import os

#######################################################################
# Evaluate function
def evaluate(qf, ql, gf, gl):
    query = qf
    score = np.dot(gf, query)
    # Predict index
    index = np.argsort(score)  # From small to large
    index = index[::-1]

    # Good index and junk index
    query_index = np.argwhere(gl == ql).flatten()
    junk_index1 = np.argwhere(gl == -1).flatten()
    junk_index = np.concatenate((junk_index1, np.array([])))  # Ensure junk_index is a 1D array

    CMC_tmp = compute_mAP(index, query_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if len(good_index) == 0:  # If empty
        cmc[0] = -1
        return ap, cmc

    # Remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # Find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap += d_recall * (old_precision + precision) / 2

    return ap, cmc

######################################################################
# Load results
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = m_result['mquery_f']
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]

# Evaluate single-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    try:
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp
        print(f"Query {i}: Rank@1={CMC_tmp[0]}")
    except Exception as e:
        print(f"Error evaluating query {i}: {e}")

CMC = CMC.float()
CMC = CMC / len(query_label)  # Average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

# Evaluate multi-query if applicable
if multi:
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        try:
            mquery_index1 = np.argwhere(mquery_label == query_label[i]).flatten()
            mq = np.mean(mquery_feature[mquery_index1, :], axis=0)
            ap_tmp, CMC_tmp = evaluate(mq, query_label[i], gallery_feature, gallery_label)
            if CMC_tmp[0] == -1:
                continue
            CMC += CMC_tmp
            ap += ap_tmp
        except Exception as e:
            print(f"Error evaluating multi-query {i}: {e}")
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # Average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
