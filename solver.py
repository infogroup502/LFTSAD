import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.LFTSAD import LFTSAD
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
import pandas as pd
warnings.filterwarnings('ignore')





def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.min_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.sw_max_mean = self.sw_max_mean
        self.sw_loss = self.sw_loss
        self.p_seq = self.p_seq
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')


    def build_model(self):
        self.model = LFTSAD(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c,
                                d_model=self.d_model, patch_size=self.patch_size, channel=self.input_c,
                                patch_seq=self.patch_seq,seq_size=self.seq_size)

        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        



    def train(self):

        time_now = time.time()

        train_steps = len(self.train_loader) #3866

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device) #(128,100,51)
                series, prior, series_seq, prior_seq = self.model(input)

                loss = 0.0
                for u in range(len(prior)):
                    if (self.sw_loss == 0):
                        loss += (self.p_seq * self.criterion(series_seq[u], prior_seq[u]) + (1 - self.p_seq) * self.criterion(
                            series[u], prior[u]))
                    else:
                        loss += (self.p_seq * self.criterion(series_seq[u], prior_seq[u]) + (1 - self.p_seq) * self.criterion(
                            series[u], prior[u]))


                loss = loss / len(prior)
                # loss = revin_layer(loss, 'norm')
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            # vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior, series_seq, prior_seq = self.model(input)

            loss = 0
            for u in range(len(prior)):

                if (self.sw_loss == 0):
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])
                else:
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])

            if (self.sw_max_mean == 0):
                loss = torch.mean(loss, dim=-1)
            else:
                loss, _ = torch.max(loss, dim=-1)

            metric = torch.softmax(loss, dim=-1)
            # metric = loss
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        if (len(attens_energy) == 0):
            print("win_size * batchsize的乘积过大，请适当调小")
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        test_labels = []
        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, series_seq, prior_seq = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            test_labels.append(labels)
            loss = 0
            for u in range(len(prior)):

                if (self.sw_loss == 0):
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])
                else:
                    loss += (self.p_seq * self.criterion_keep(series_seq[u], prior_seq[u])
                             + (1 - self.p_seq) * self.criterion_keep(series[u], prior[u]))
                    # loss += self.criterion_keep(series[u], prior[u])

            if (self.sw_max_mean == 0):
                loss = torch.mean(loss, dim=-1)
            else:
                loss, _ = torch.max(loss, dim=-1)
            metric = torch.softmax(loss, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)



        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio", self.anormly_ratio)
        print("Threshold :", thresh)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))
        results_df = pd.DataFrame({
            'Timestamp': np.arange(len(gt)),  # 如果有实际的时间戳数据，替换为实际的时间戳
            'Actual_Label': gt,
            'Predicted_Label': pred,
            'Energy_Score': test_energy
        })
        #
     

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score


