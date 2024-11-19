import torch
from torch.utils.data import DataLoader

from dataset.ubfc_rppg import UbfcDataset, UbfcDataLoader
from dataset.pure import PureDataset, PureDataLoader
from configs.config import TrainArgs, UBFCArgs, PureArgs
from model.trainer import Trainer
from utils import set_random_seed


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
        ubfc_rppg_args = UBFCArgs()
        pure_args = PureArgs()
        
        # train_loader, test_loader = UbfcDataLoader(ubfc_rppg_args, train_args).get_dataloder()
        train_loader, test_loader = PureDataLoader(pure_args, train_args).get_dataloder()
        # run
        if train_args.mode == 'train_test':
            trainer.train(train_loader)
            if test_loader is not None:
                trainer.test(test_loader)
            else:
                logger.info('Test_loader is None, use train_loader for test.')
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
    
    