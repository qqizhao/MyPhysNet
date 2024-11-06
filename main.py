import torch
from torch.utils.data import DataLoader

from data_process.ubfc_rppg_dataset import MyDataset
from configs.config import PreprocessArgs, TrainArgs
from model.trainer import Trainer
from utils import set_random_seed
from data_process.utils import get_raw_data, split_raw_data


def load_datasets(preprocess_args, train_args):
    
    data_dirs = get_raw_data(preprocess_args.path['raw_dataset_path'])
    train_dirs, test_dirs = split_raw_data(data_dirs, preprocess_args.split_ratio)
    
    train_loader = DataLoader(
        MyDataset(preprocess_args, data_dirs=train_dirs, mode='train'),
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = None
    if test_dirs is not None:
        test_loader = DataLoader(
            MyDataset(preprocess_args, data_dirs=test_dirs, mode='test'),
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    return train_loader, test_loader


if __name__ == '__main__':
    try:
        # 42 is the magic number
        SEED = 42
        set_random_seed(SEED)
        
        train_args = TrainArgs()
        trainer = Trainer(train_args)
        logger = trainer.logger
        logger.info(f"SEED: {SEED}")
        
        # get dataloader
        preprocess_args = PreprocessArgs()
        train_loader, test_loader = load_datasets(preprocess_args, train_args)
        
        # run
        if train_args.mode == 'train_test':
            trainer.train(train_loader)
            if test_loader is not None:
                trainer.test(test_loader)
            else:
                trainer.test(train_loader)
        elif train_args.mode == 'only_test':
            if test_loader is not None:
                trainer.test(test_loader)
            else:
                trainer.test(train_loader)
        else:
            logger.error(f"unsupported mode: {train_args.mode}")
            raise
        
    except Exception as e:
        logger.error(f"error: {str(e)}")
        raise
    
    