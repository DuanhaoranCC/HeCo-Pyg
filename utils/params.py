import argparse

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--use_cuda', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=600)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)
    
    args = parser.parse_args()

    if args.dataset == "acm":
        args.sample_rate = {'a': 7, 's': 1}
    elif args.dataset == "dblp":
        args.sample_rate = {'p': 6}
    elif args.dataset == "aminer":
        args.sample_rate = {'a': 7, 'r': 1}
    elif args.dataset == "freebase":
        args.sample_rate = {'d': 1, 'a': 18, 'w': 2}
    else:
        raise NotImplementedError
    
    return args


