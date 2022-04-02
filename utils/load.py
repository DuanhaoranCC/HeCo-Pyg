import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import HeteroData


def preprocess_sp_features(features):
    features = features.tocoo()
    row = torch.from_numpy(features.row)
    col = torch.from_numpy(features.col)
    e = torch.stack((row, col))
    v = torch.from_numpy(features.data)
    x = torch.sparse_coo_tensor(e, v, features.shape).to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x

def preprocess_th_features(features):
    x = features.to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x

def nei_to_edge_index(nei, reverse=False):
    edge_indexes = []

    for src, dst in enumerate(nei):
        src = torch.tensor([src], dtype=dst.dtype, device=dst.device)
        src = src.repeat(dst.shape[0])
        if reverse:
            edge_index = torch.stack((dst, src))
        else:
            edge_index = torch.stack((src, dst))

        edge_indexes.append(edge_index)
    
    return torch.cat(edge_indexes, dim=1)

def sp_feat_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def sp_adj_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices

def make_sparse_eye(N):
    e = torch.arange(N, dtype=torch.long)
    e = torch.stack([e, e])
    o = torch.ones(N, dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=(N, N))

def make_sparse_tensor(x):
    row, col = torch.where(x == 1)
    e = torch.stack([row, col])
    o = torch.ones(e.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=x.shape)

def load_dblp():
    path = "./data/dblp/" 
    ratio = [20, 40, 60]

    label = np.load(path + "labels.npy").astype('int32')
    nei_p = np.load(path + "nei_a.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = torch.LongTensor(label)
    nei_p = nei_to_edge_index([torch.LongTensor(i) for i in nei_p], True)
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    apa = sp_adj_to_tensor(apa)
    apcpa = sp_adj_to_tensor(apcpa)
    aptpa = sp_adj_to_tensor(aptpa)
    pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    mask = torch.tensor([False] * feat_a.shape[0])
    data['a'].x = feat_a
    data['a'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['a'][train_mask_l] = train_mask
        data['a'][val_mask_l] = val_mask
        data['a'][test_mask_l] = test_mask
        
    data['p'].x = feat_p
    data[('p', 'a')].edge_index = nei_p
    data[('a', 'p', 'a')].edge_index = apa
    data[('a', 'pcp', 'a')].edge_index = apcpa
    data[('a', 'ptp', 'a')].edge_index = aptpa
    data[('a', 'pos', 'a')].edge_index = pos

    metapath_dict={
        ('a', 'p', 'a'): None,
        ('a', 'pcp', 'a'): None,
        ('a', 'ptp', 'a'): None
    }

    schema_dict = {
        ('p', 'a'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'a'
    data['use_nodes'] = ('p', 'a')

    return data

def load_acm():
    path = "./data/acm/" 
    ratio = [20, 40, 60]
    label = np.load(path + "labels.npy").astype('int32')
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_s = make_sparse_eye(60)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_s = nei_to_edge_index([torch.LongTensor(i) for i in nei_s])
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    feat_s = preprocess_th_features(feat_s)
    pap = sp_adj_to_tensor(pap)
    psp = sp_adj_to_tensor(psp)
    pos = sp_adj_to_tensor(pos)

    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['s'].x = feat_s
    data['p'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

    data[('a', 'p')].edge_index = nei_a
    data[('s', 'p')].edge_index = nei_s
    data[('p', 'a', 'p')].edge_index = pap
    data[('p', 's', 'p')].edge_index = psp
    data[('p', 'pos', 'p')].edge_index = pos
    
    metapath_dict={
        ('p', 'a', 'p'): None,
        ('p', 's', 'p'): None
    }

    schema_dict = {
        ('a', 'p'): None,
        ('s', 'p'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 's')

    return data

def load_aminer():
    ratio = [20, 40, 60]
    path = "./data/aminer/"

    label = np.load(path + "labels.npy").astype('int32')
    nei_a = np.load(path + "nei_pa.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_pr.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = make_sparse_eye(6564)
    feat_a = make_sparse_eye(13329)
    feat_r = make_sparse_eye(35890)
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]


    label = torch.LongTensor(label)
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_r = nei_to_edge_index([torch.LongTensor(i) for i in nei_r])
    feat_p = preprocess_th_features(feat_p)
    feat_a = preprocess_th_features(feat_a)
    feat_r = preprocess_th_features(feat_r)
    pap = sp_adj_to_tensor(pap)
    prp = sp_adj_to_tensor(prp)
    pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]


    data = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['r'].x = feat_r
    data['p'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask
    
    data[('a', 'p')].edge_index = nei_a
    data[('r', 'p')].edge_index = nei_r
    data[('p', 'a', 'p')].edge_index = pap
    data[('p', 'r', 'p')].edge_index = prp
    data[('p', 'pos', 'p')].edge_index = pos

    metapath_dict={
        ('p', 'a', 'p'): None,
        ('p', 'r', 'p'): None
    }

    schema_dict = {
        ('a', 'p'): None,
        ('r', 'p'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 'r')

    return data


def load_freebase():
    ratio = [20, 40, 60]
    path = "./data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    nei_d = np.load(path + "nei_md.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_ma.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_mw.npy", allow_pickle=True)

    feat_m = make_sparse_eye(3492)
    feat_d = make_sparse_eye(2502)
    feat_a = make_sparse_eye(33401)
    feat_w = make_sparse_eye(4459)

    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    nei_d = nei_to_edge_index([torch.LongTensor(i) for i in nei_d])
    nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    nei_w = nei_to_edge_index([torch.LongTensor(i) for i in nei_w])

    feat_m = preprocess_th_features(feat_m)
    feat_d = preprocess_th_features(feat_d)
    feat_a = preprocess_th_features(feat_a)
    feat_w = preprocess_th_features(feat_w)

    mam = sp_adj_to_tensor(mam)
    mdm = sp_adj_to_tensor(mdm)
    mwm = sp_adj_to_tensor(mwm)
    pos = sp_adj_to_tensor(pos)

    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    mask = torch.tensor([False] * feat_m.shape[0])

    data['m'].x = feat_m
    data['d'].x = feat_d
    data['a'].x = feat_a
    data['w'].x = feat_w
    data['m'].y = label

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['m'][train_mask_l] = train_mask
        data['m'][val_mask_l] = val_mask
        data['m'][test_mask_l] = test_mask
    
    data[('d', 'm')].edge_index = nei_d
    data[('a', 'm')].edge_index = nei_a
    data[('w', 'm')].edge_index = nei_w
    data[('m', 'a', 'm')].edge_index = mam
    data[('m', 'd', 'm')].edge_index = mdm
    data[('m', 'w', 'm')].edge_index = mwm

    num_main_nodes = feat_m.shape[0]
    data[('m', 'pos', 'm')].edge_index = pos

    metapath_dict={
        ('m', 'a', 'm'): None,
        ('m', 'd', 'm'): None,
        ('m', 'w', 'm'): None
    }

    schema_dict = {
        ('a', 'm'): None,
        ('d', 'm'): None,
        ('w', 'm'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    data['main_node'] = 'm'
    data['use_nodes'] = ('m', 'a', 'd', 'w')

    return data