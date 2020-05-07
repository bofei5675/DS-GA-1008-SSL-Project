import sys
sys.path.append("..")

from pix2vox import pix2vox
from self_supervised.model import CPCModel
from self_supervised.config import Args


class DockSLL():

    def __init__(self, ssl_model_addr='../model_weights/best-model-cpc-3e-05.pth', args=None):

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.ssl_model = CPCModel(Args())
        if self.use_cuda:
            state_dict1 = torch.load(ssl_model_addr)
            self.ssl_model.cuda()
            self.ssl_model.load_state_dict(state_dict1)
        self.ssl_model.eval()
        if args.model_config == 'center_net':
            self.sl_model = pix2vox(args, args.pre_train, False, True)
        else:
            self.sl_model = pix2vox(args, args.pre_train, args.det, args.seg)
        if self.use_cuda:
            self.sl_model.cuda()
        self.sl_model.train()

    def copy_ssl_encoder(self):

        for param_q, param_k in zip(self.ssl_model.encoder_q.parameters(), self.sl_model.encoder.parameters()):
            try:
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = True # not freezing
            except:
                print('copied ssl encoder to the lowest layer, weight freezed')
                break

    def get_model_from_ssl(self):
        self.copy_ssl_encoder()
        return self.sl_model

import torch
import torch.nn as nn
import torch.nn.functional as F

# pix2vox = DockSLL().get_model_from_ssl()

def ssl_loader(args,
               ssl_model='../model_weights/best-model-cpc-3e-05.pth'):
    return DockSLL(ssl_model, args).get_model_from_ssl()

#ssl_loader()