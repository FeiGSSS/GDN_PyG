import os
import argparse
import datetime
import random
import numpy as np
import torch

from src.dataset import TimeDataset, get_loaders
from src.model import GDN
from src.anomaly_detection import test_performence


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
        
        
class TrainingHelper(object):
    """
    Helper class for training.
    1. save model if val loss is smaller than min val loss
    2. early stop if val loss does not decrease for early_stop_patience epochs
    """
    def __init__(self, args):
        self.min_val_loss = 1e+30
        self.early_stop_count = 0
        
        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)
        model_marker = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_checkpoint_path = os.path.join(args.model_checkpoint_path,
                        "{}_model_{}.pth".format(args.dataset, model_marker))
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
    parser.add_argument("--dataset", type=str, default="swat", help="Dataset name.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=5, help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=1, help="Slide window stride.")
    # model configurations
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--out_layer_num", type=int, default=1, help="Number of out layers.")
    parser.add_argument("--out_layer_hid_dim", type=int, default=128, help="Out layer hidden dimension.")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads.")
    parser.add_argument("--topk", type=int, default=15, help="The knn graph topk.")
    # training configurations
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("--device", type=int, default=0, help="Training cuda device, -1 for cpu.")
    parser.add_argument("--random_seed", type=int, default=2023, help="Random seed.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stop patience.")
    parser.add_argument("--model_checkpoint_path", type=str, default="./checkpoints", help="Model checkpoint path.")
    # config
    args = parser.parse_args()
        
    
    # set random seed
    seed_everything(args.random_seed)
    
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda:{}".format(args.device))
    
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
    for epoch in range(1, args.epochs):
        epoch_loss = []
        model.train()
        for batch in train_loader:
            x, y, _ = [item.to(device) for item in batch]
            optimizer.zero_grad()
            out = model(x)
            loss = model.loss(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        train_loss = np.mean(epoch_loss)
        # validation
        val_loss = test(model, val_loader, device, mode='val')
        print("Epoch {:>3}/{} train loss: {:.8f}, val loss: {:.8f}".format(epoch, args.epochs, train_loss, val_loss))
        
        training_helper.check(val_loss, model)
        if training_helper.early_stop:
            print("Early stop at epoch {} with val loss {:.4f}.".format(epoch,
                                                                        training_helper.min_val_loss))
            print("Model saved at {}.".format(training_helper.model_checkpoint_path))
            break
        
    # testing
    print("Testing...")
    model.eval()
    model.load_state_dict(torch.load(training_helper.model_checkpoint_path))
    _, val_results = test(model, val_loader, device, mode='test')
    _, test_results = test(model, test_loader, device, mode='test')
    all_performance = test_performence(test_results, val_results)
    
    return all_performance

if __name__ == '__main__':
    main()
        