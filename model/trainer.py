import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from statistics import mean
import os
import time
import numpy as np
import pickle

from .physnet import PhysNet_3DCNN_ED
from .NegPearsonLoss import Neg_Pearson
from utils import create_exp_dir, create_logger, get_max_exp, merge_clips, get_latest_checkpoint
from evaluate import postprocess,metric

class Trainer():
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = PhysNet_3DCNN_ED()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = Neg_Pearson()
        
        # logs/exp
        self.exp_path = create_exp_dir(args.logs_path)
        self.logger = create_logger(self.exp_path)
        
        self.logs_path = args.logs_path
        self.freq_save_model = args.freq_save_model
        
        self.logger.info('Train args:')
        self.logger.info(vars(args))
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Logs directory: {self.exp_path}")
        
        
    def train(self, train_loader):
        
        self.checkpoints_path = create_exp_dir(self.args.checkpoints_path)
        self.logger.info(f"Checkpoints directory: {self.checkpoints_path}")
        self.logger.info('Training is starting!')
        
        train_writer = SummaryWriter(log_dir=self.exp_path)
        
        self.model.to(self.device)
        self.model.train()
        
        num_train_loader = len(train_loader)
        
        for epoch in range(self.epochs):
            start_time = time.time()
            self.logger.info(f"Epoch: {epoch+1}/{self.epochs}")
            train_loss = []
            
            for i, (data, label, _, _) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data = data.permute(0, 2, 1, 3, 4)
                
                output, _, _, _ = self.model(data)
                # normalize
                output = (output - output.mean(dim=-1, keepdim=True)) / output.std(dim=-1, keepdim=True)
                label = (label - label.mean(dim=-1, keepdim=True)) / label.std(dim=-1, keepdim=True)
                loss = self.criterion(output, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss.append(loss.item())
                self.logger.info(f"Epoch: {epoch+1}/{self.epochs} | Batch: {i+1}/{num_train_loader} | Train loss: {mean(train_loss)}")
            
            train_writer.add_scalar('Train Loss', mean(train_loss), epoch)
            
            end_time = time.time()
            self.logger.info(f"Epoch time: {end_time - start_time:.2f}s")
            
            if (epoch+1) % self.freq_save_model == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_path, f'model_{epoch+1}.pth'))
                self.logger.info(f"Model saved to {os.path.join(self.checkpoints_path, f'model_{epoch+1}.pth')}")
        
        self.logger.info('Training is done!\n')
        train_writer.close()
                
    def test(self,test_loader):
        
        model = PhysNet_3DCNN_ED()
        now_exp_path = get_max_exp(self.args.checkpoints_path)
        latest_checkpoint = get_latest_checkpoint(now_exp_path)
        self.logger.info(f"Loading model from {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint))
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            self.logger.info('Testing is starting!')
            start_time = time.time()
            predictions = dict()
            labels = dict()
            progress_bar = tqdm(range(len(test_loader)))
            
            # subjects: 文件名  chunks: 每个文件的chunk索引
            for idx, (data, label, subjects, chunks) in enumerate(test_loader):
                data, label = data.to(self.device), label.to(self.device)
                data = data.permute(0, 2, 1, 3, 4)  
                output, _, _, _ = model(data)

                # predictions:{file_name: {chunk_idx: [output]}}
                # labels:{file_name: {chunk_idx: [label]}}
                for i in range(len(data)):
                    file_name = subjects[i]                  
                    chunk_idx = chunks[i]    
                    if file_name not in predictions.keys():
                        predictions[file_name] = dict()
                        labels[file_name] = dict()
                    predictions[file_name][chunk_idx] = output[i].detach().cpu().numpy()
                    labels[file_name][chunk_idx] = label[i].detach().cpu().numpy()
               
                progress_bar.update(1)
            progress_bar.close()

            # 合并所有clips并进行后处理
            pred_phys = []
            label_phys = []
            self.logger.info('Merging clips...')
            for file_name in predictions.keys():
                # 合并同一文件的所有chunks
                pred_temp = merge_clips(predictions[file_name])
                label_temp = merge_clips(labels[file_name])
                # 如果使用FFT后处理
                if self.args.post == 'fft':
                    # 对预测值和真实值进行FFT处理,提取生理信号,返回的是视频对应的脉搏值
                    pred_temp = postprocess.fft_physiology(pred_temp, Fs=float(self.args.Fs),
                                                   diff=self.args.diff_flag,
                                                   detrend_flag=self.args.detrend_flag).reshape(-1)
                    label_temp = postprocess.fft_physiology(label_temp, Fs=float(self.args.Fs),
                                                    diff=self.args.diff_flag,
                                                    detrend_flag=self.args.detrend_flag).reshape(-1)
                pred_phys.append(pred_temp)
                label_phys.append(label_temp)
                
            saved_data = dict()
            saved_data['predictions'] = predictions
            saved_data['labels'] = labels
            saved_data['label_type'] = self.args.preprocess_label_type
            saved_data['fs'] = self.args.Fs

            # 创建输出路径
            output_path = os.path.join(self.exp_path, 'test_output_results.pkl')
            with open(output_path, 'wb') as handle:
                pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f'Test results saved to: {output_path}')
            
            self.logger.info('Calculating metrics...')
            metrics = metric.cal_metric(pred_phys, label_phys)
            
            self.logger.info(f"Test result -> MAE: {metrics[0]:.4f}, MSE: {metrics[1]:.4f}, RMSE: {metrics[2]:.4f}, Pearson: {metrics[3]:.4f}")
            end_time = time.time()
            self.logger.info(f"Test time: {end_time - start_time:.2f}s")
            self.logger.info('Test is done!\n')

        