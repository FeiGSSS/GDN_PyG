import os
import argparse
import random
import datetime
import yaml
import numpy as np

import torch
import torch.nn.functional as F

from src.dataset import TimeDataset, get_loaders
from src.model import GDN
from src.anomaly_detection import test_performence

class TrainingHelper(object):
    def __init__(self, args) -> None:
        self.min_val_loss = 1e+30
        self.early_stop_count = 0
        
        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)
        model_marker = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_checkpoint_path = os.path.join(args.model_checkpoint_path,
                                                  "{}_model_{}.pth".format(args.dataset,
                                                                           model_marker))
        self.early_stop_patience = args.early_stop_patience
        
        self.early_stop: bool = False
        
    
    def check(self, val_loss: float, model: GDN) -> bool:
        # save model if val loss is smaller than min val loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            torch.save(model.state_dict(), self.model_checkpoint_path)
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        
        if self.early_stop_count >= self.early_stop_patience:
            self.early_stop = True

def test(model: GDN,
         loader: torch.utils.data.DataLoader,
         device: torch.device,
         mode: str = 'test'):
    assert mode in ['test', 'val'], "mode must be 'test' or 'val'!"
    model.eval()
    loss_list, prediction_list, ground_truth_list, label_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, label = [item.to(device) for item in batch]
            out = model(x)
            loss = model.loss(out, y) # both have size [batch_size, num_nodes]
            loss_list.append(loss.item())
            prediction_list.append(out.cpu().numpy())
            ground_truth_list.append(y.cpu().numpy())
            label_list.append(label.cpu().numpy())
    loss = np.mean(loss_list)
    prediction = np.concatenate(prediction_list, axis=0) # size [total_time_len, num_nodes]
    ground_truth = np.concatenate(ground_truth_list, axis=0) # size [total_time_len, num_nodes]
    anomaly_label = np.concatenate(label_list, axis=0) # size [total_time_len]
    
    if mode == 'test':
        return loss, (prediction, ground_truth, anomaly_label)
    else:
        return loss
        
def main():
    parser = argparse.ArgumentParser()
    # dataset configurations
    parser.add_argument("--dataset", type=str, default="swat",
                        help="Dataset name.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=5,
                        help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=1,
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
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay.")
    parser.add_argument("--device", type=int, default=0,
                        help="Training cuda device, -1 for cpu.")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stop patience.")
    parser.add_argument("--model_checkpoint_path", type=str, default="./checkpoints",
                        help="Model checkpoint path.")
    # config
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path.")
    args = parser.parse_args()
    
    # load config and update args using config
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    # print all args info in a table-like format
    print("Configurations:")
    for key, value in vars(args).items():
        print("\t{}: {}".format(key, value))
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")
    
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
                                 betas = (0.9, 0.99),
                                 weight_decay=args.weight_decay)
    
    # training helper
    training_helper = TrainingHelper(args)
    
    # begin training and validation
    print("Training...")
    for epoch in range(args.epochs):
        epoch_loss = []
        model.train()
        for batch in train_loader:
            x, y, label = [item.to(device) for item in batch]
            optimizer.zero_grad()
            out = model(x)
            loss = model.loss(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        train_loss = np.mean(epoch_loss)
        
        # validation
        assert val_loader is not None, "val_loader is None!"
        val_loss = test(model, val_loader, device, mode='val')
        
        print("Epoch {:>3} train loss: {:.4f}, val loss: {:.4f}".format(epoch, train_loss, val_loss))
        
        training_helper.check(val_loss, model)
        if training_helper.early_stop:
            print("Early stop at epoch {} with val loss {:.4f}.".format(epoch,
                                                                        training_helper.min_val_loss))
            print("Model saved at {}.".format(training_helper.model_checkpoint_path))
            break
        
    # testing
    print("Testing...")
    assert test_loader is not None, "test_loader is None!"
    model.load_state_dict(torch.load(training_helper.model_checkpoint_path))
    _, val_results = test(model, val_loader, device, mode='test')
    _, test_results = test(model, test_loader, device, mode='test')
    test_performence(test_results, val_results)

if __name__ == '__main__':
    main()