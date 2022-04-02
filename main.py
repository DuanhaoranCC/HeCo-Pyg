import numpy
import torch
from utils.load import load_acm, load_aminer, load_dblp, load_freebase
from utils.params import set_params
from utils.evaluate import evaluate
from module.heco import HeCo
import datetime
import pickle as pkl
import os
import random


args = set_params()
if torch.cuda.is_available() and args.use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_name = args.dataset
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train():
    if args.dataset == "acm":
        load_data = load_acm
    elif args.dataset == "dblp":
        load_data = load_dblp
    elif args.dataset == "aminer":
        load_data = load_aminer
    elif args.dataset == "freebase":
        load_data = load_freebase
    else:
        raise NotImplementedError
    
    print(args)

    data = load_data().to(device)
   
    model = HeCo(
        data, 
        args.hidden_dim, 
        args.feat_drop, 
        args.attn_drop, 
        args.sample_rate, 
        args.tau, 
        args.lam
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
 
    embeds = model.get_embeds(data)
    satio = args.ratio[0]
    print(evaluate(
        embeds, 
        satio,
        data[data.main_node][f'{satio}_train_mask'],
        data[data.main_node][f'{satio}_val_mask'],
        data[data.main_node][f'{satio}_test_mask'],
        data[data.main_node].y,
        device,
        args.dataset,
        args.eva_lr,
        args.eva_wd,
        False
    ))
   
    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(data)
        
        print(f"{epoch}: loss: {loss.item():.4f}")
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+dataset_name+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        
        loss.backward()
        optimizer.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_'+dataset_name+'.pkl'))
    model.eval()
    os.remove('HeCo_'+dataset_name+'.pkl')
    embeds = model.get_embeds(data)
     
    for ratio in args.ratio:
        evaluate(
            embeds, 
            ratio,
            data[data.main_node][f'{ratio}_train_mask'],
            data[data.main_node][f'{ratio}_val_mask'],
            data[data.main_node][f'{ratio}_test_mask'],
            data[data.main_node].y,
            device,
            args.dataset,
            args.eva_lr,
            args.eva_wd
        )

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
