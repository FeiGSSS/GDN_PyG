import argparse
import random

import numpy as np

import torch
import torch.nn.functional as F

from src.dataset import TimeDataset, get_loaders
from src.model import GDN
        
def main():
    parser = argparse.ArgumentParser()
    # dataset configurations
    parser.add_argument("--dataset", type=str, default="swat",
                        help="Dataset name.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=15,
                        help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=5,
                        help="Slide window stride.")
    # model configurations
    parser.add_argument("--hid_dim", type=int, default=64,
                        help="Hidden dimension.")
    parser.add_argument("--out_layer_num", type=int, default=1,
                        help="Number of out layers.")
    parser.add_argument("--out_layer_hid_dim", type=int, default=256,
                        help="Out layer hidden dimension.")
    parser.add_argument("--heads", type=int, default=1,
                        help="Number of heads.")
    parser.add_argument("--topk", type=int, default=20,
                        help="The knn graph topk.")
    # training configurations
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device.")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Random seed.")
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda:1")
    
    # load datasets and dataloaders
    print("Loading datasets...")
    train_dataset = TimeDataset(dataset_name=args.dataset,
                                mode='train',
                                slide_win=args.slide_win,
                                slide_stride=args.slide_stride)
    test_dataset = TimeDataset(dataset_name=args.dataset,
                               mode='test',
                               slide_win=args.slide_win,
                               slide_stride=args.slide_stride)
    
    train_loader, val_loader, test_loader = get_loaders(train_dataset,
                                                        test_dataset,
                                                        args.batch_size,
                                                        val_ratio=args.val_ratio)
    
    # build model
    model = GDN(number_nodes=train_dataset.number_nodes,
                in_dim = train_dataset.input_dim,
                hid_dim = args.hid_dim,
                out_layer_hid_dim = args.out_layer_hid_dim,
                out_layer_num = args.out_layer_num,
                topk = args.topk,
                heads = args.heads)
    model = model.to(device)
    
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    
    # begin training
    
    for epoch in range(args.epochs):
        epoch_loss = []
        model.train()
        for batch in train_loader:
            x, y, label = [item.to(device) for item in batch]
            optimizer.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, y, reduction='mean')
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
        print("Epoch {:<3} Loss: {:.4f}".format(epoch, np.mean(epoch_loss)))



if __name__ == '__main__':
    main()