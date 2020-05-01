import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parallel import data_parallel, DistributedDataParallel


class EncoderResNet(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "resnet101": models.resnet101(pretrained=False),
                            "resnet152": models.resnet152(pretrained=False),
                            "resnext50_wide": models.resnext50_32x4d(pretrained=False),
                            "resnext101_wide": models.resnext101_32x8d(pretrained=False),
                            }

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.Sequential(*list(resnet.children())[-2:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        self.weights_init()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        batch, seq, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        f = self.features(x)
        h = self.pool(f)
        h = h.squeeze().view(batch, seq, -1)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x).view(batch, seq, -1)
        return f, x

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            xavier(m.bias.data)


class Seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, seq_len=3,
                 encoder_model='GRU', decoder_model='GRU'):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.rnn_encoder_dict = {"RNN": nn.RNN(input_dim, hidden_dim, num_layers=num_layers),
                                 "LSTM": nn.LSTM(input_dim, hidden_dim, num_layers=num_layers),
                                 "GRU": nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
                                 }

        self.rnn_decoder_dict = {"RNN": nn.RNN(hidden_dim, output_dim, num_layers=num_layers),
                                 "LSTM": nn.LSTM(hidden_dim, output_dim, num_layers=num_layers),
                                 "GRU": nn.GRU(hidden_dim, output_dim, num_layers=num_layers)
                                 }

        self.rnn_lr = nn.Linear(hidden_dim * num_layers, output_dim * num_layers)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.seq_len = seq_len

        self.decoder_model = decoder_model

        self.encoder = self._get_encodermodel(encoder_model)
        self.decoder = self._get_decodermodel(decoder_model)

        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

        self.weights_init()

    def _get_encodermodel(self, model_name):
        model = self.rnn_encoder_dict[model_name]
        print("RNN model:", model_name)
        return model

    def _get_decodermodel(self, model_name):
        model = self.rnn_decoder_dict[model_name]
        print("RNN model:", model_name)
        return model

    def initHidden(self, batch_size):
        if self.decoder_model == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)

    def initInput(self, batch_size):
        return torch.zeros(self.seq_len, batch_size, self.hidden_dim, device=self.device)

    def weights_init(m):
        if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            xavier_normal(m.weight.data)
            xavier_uniform(m.bias.data)

    def forward(self, encoder_inputs):
        batch_size = encoder_inputs.size(1)
        encoder_hidden = self.initHidden(batch_size)

        encoder_outputs = torch.zeros(
            self.seq_len, batch_size, self.hidden_dim, device=self.device)

        input_length = encoder_inputs.size(0)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                encoder_inputs[ei].unsqueeze(0), encoder_hidden)
            encoder_outputs[ei] = encoder_output

        decoder_inputs = self.initInput(batch_size)

        if self.hidden_dim != self.output_dim:
            if self.decoder_model == 'LSTM':
                decoder_hidden = (
                    self.rnn_lr(encoder_hidden[0].transpose_(0, 1).reshape(batch_size, -1))
                        .reshape(batch_size, -1, self.output_dim).transpose_(0, 1),
                    self.rnn_lr(encoder_hidden[1].transpose_(0, 1).reshape(batch_size, -1))
                        .reshape(batch_size, -1, self.output_dim).transpose_(0, 1))

            else:
                decoder_hidden = self.rnn_lr(encoder_hidden.transpose_(0, 1).reshape(batch_size, -1)).contiguous()
                decoder_hidden = decoder_hidden.reshape(batch_size, -1, self.output_dim).transpose_(0, 1).contiguous()
        else:
            decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            self.seq_len, batch_size, self.output_dim, device=self.device)

        for di in range(input_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_inputs[di].unsqueeze(0), decoder_hidden)
            decoder_outputs[di] = decoder_output

        return decoder_outputs


class CPCModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder_q = EncoderResNet(args.resnet_model, args.embed_size)
        self.encoder_k = EncoderResNet(args.resnet_model, args.embed_size)

        self.seq2seq = Seq2seq(args.embed_size, args.rnn_hidden_size, args.output_size, num_layers=args.rnn_n_layers,
                               seq_len=args.rnn_seq_len,
                               encoder_model=args.encoder_model, decoder_model=args.decoder_model)

        self.device = self._get_device()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.encoder_q.to(self.device)
        self.encoder_k.to(self.device)
        self.seq2seq.to(self.device)

        self.m = 0.99

    def encode(self, inputs):
        _, x = data_parallel(self.encoder_q, inputs)
        return x

    def get_feature(self, inputs):
        f, _ = data_parallel(self.encoder_q, inputs)
        return f

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on: {device}, {torch.cuda.get_device_name()}")
        return device

    def encode_fixed(self, inputs):
        _, x = data_parallel(self.encoder_k, inputs)
        return x

    def forward(self, inputs):
        f, x = data_parallel(self.encoder_q, inputs.to(self.device))
        x = x.transpose_(1, 0)
        x = self.seq2seq(x)
        outputs = x.transpose_(1, 0)
        return outputs

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
