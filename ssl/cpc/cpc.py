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

        loss = self.criterion(y_hat, y)

        return loss

    def _load_pre_trained_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.args.mdl_dir, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
            with open(self.args.log_dir + '/config.txt', 'a') as f:
                f.write("Loaded pre-trained model with success.")

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
            with open(self.args.log_dir + '/config.txt', 'a') as f:
                f.write("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (x, y) in valid_loader:
                loss = self._step(model, x, y)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loader()
        model = CPCModel(self.args)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        start_time = time.time()

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        model.train()

        for epoch_counter in range(self.args.epochs):
            for iteration, (x, y) in enumerate(train_loader):

                optimizer.zero_grad()
                loss = self._step(model, x, y)

                if n_iter % self.args.log_every_n_steps == 0:
                    info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(n_iter, epoch_counter,
                                                                                              iteration,
                                                                                              len(train_loader),
                                                                                              time.time() - start_time)

                    info += 'Train Loss: {:.4f},'.format(loss.item())
                    with open(self.args.log_dir + '/config.txt', 'a') as f:
                        f.write(info)
                    print(info)

                loss.backward()
                optimizer.step()
                n_iter += 1

                if n_iter % self.args.encoder_update_every_n_steps == 0:
                    model.update_encoder_k()

                if epoch_counter % self.args.eval_every_n_epochs == 0:
                    valid_loss = self._validate(model, valid_loader)

                    info = "\n====> Cur_iter: [{}]: Valid Epoch[{}]({}/{}): time: {:4.4f}: ".format(n_iter,
                                                                                                    valid_n_iter,
                                                                                                    len(valid_loader),
                                                                                                    len(valid_loader),
                                                                                                    time.time() - start_time)

                    info += 'Valid Loss: {:.4f}'.format(valid_loss.item())
                    with open(self.args.log_dir + '/config.txt', 'a') as f:
                        f.write(info)
                    print(info)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        info = "\n====> Saving Best Model."
                        with open(self.args.log_dir + '/config.txt', 'a') as f:
                            f.write(info)
                        torch.save(model.state_dict(), os.path.join(self.args.mdl_dir, 'model.pth'))

                    valid_n_iter += 1

                if epoch_counter >= 10:
                    scheduler.step()

                    info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(n_iter, epoch_counter,
                                                                                              iteration,
                                                                                              len(train_loader),
                                                                                              time.time() - start_time)

                    info += 'Change Learning Rate: {:.4f},'.format(scheduler.get_lr()[0])
                    with open(self.args.log_dir + '/config.txt', 'a') as f:
                        f.write(info)
                    print(info)
