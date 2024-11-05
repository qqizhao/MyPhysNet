import torch
from torch.utils.data import DataLoader

from data_process.ubfc_rppg_dataset import MyDataset
from configs.config import PreprocessArgs, TrainArgs
from model.trainer import Trainer
from utils import set_random_seed


if __name__ == '__main__':
    
    # 42 is the magic number
    SEED = 42
    set_random_seed(SEED)
    
    train_args = TrainArgs()
    trainer = Trainer(train_args)
    
    preprocess_args = PreprocessArgs()
    ubfc_dataset = MyDataset(preprocess_args)
    
    train_loader = DataLoader(ubfc_dataset, batch_size=train_args.batch_size, shuffle=True)
    
    if train_args.mode == 'train_test':
        trainer.train(train_loader)
        trainer.test(train_loader)
    if train_args.mode == 'only_test':
        trainer.test(train_loader)
    
    