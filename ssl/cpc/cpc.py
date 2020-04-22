import torch
import time
import os
import numpy as np
import torch.nn.functional as F

from model import CPCModel
from config import Args
from loss_similarity import CPCLoss
from dataset_wrapper import DataSetWrapper


class CPC_train(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.criterion = CPCLoss(args)
        self.dataset = DataSetWrapper(args, num_workers=0, s=1)

    def _step(self, model, x, y):
        y = model.encode_fixed(y)
        y = F.normalize(y, dim=2)

        y_hat = model(x)
        y_hat = F.normalize(y_hat, dim=2)

        loss_rnn = self.criterion(y_hat, y)

        x_hat = model.encode(x)
        x_hat = F.normalize(x_hat, dim=2)

        loss_contrast = self.criterion(x_hat, -y)
        return loss_rnn, loss_contrast

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = self.args.mdl_dir
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
            with open(self.args.log_dir + '/config.txt', 'w') as f:
                f.write("Loaded pre-trained model with success.")

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
            with open(self.args.log_dir + '/config.txt', 'w') as f:
                f.write("Pre-trained weights not found. Training from scratch.")

        return model

    def train(self):

        with open(self.args.log_dir + '/config.txt', 'w') as f:
            f.write(str(self.args))

        train_loader = self.dataset.get_data_loader()
        model = CPCModel(self.args)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        start_time = time.time()

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.args.epochs):
            for iteration, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                loss_rnn, loss_contrast = self._step(model, x, y)

                print(loss_rnn.item(), loss_contrast.item())

                if n_iter % self.args.log_every_n_steps == 0:
                    info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(n_iter, epoch_counter,
                                                                                              iteration,
                                                                                              len(train_loader),
                                                                                              time.time() - start_time)

                    info += 'Train RNN Loss: {:.4f}, Train Contrast Loss: {:.4f},'.format(loss_rnn.item(), loss_contrast.item())
                    print(info)
                    break
                break
            break
