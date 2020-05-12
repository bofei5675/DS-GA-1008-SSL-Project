# Bird’s-eye View Prediction: Contrastive Predictive Coding pre-train Pio2Vox with YOLOv3
Code for DS-GA-1008 Self-Supervised Learning(SSL) Final Project for Bofei Zhang, Can Cui, Yuanxi Sun

Report: [Bird-eye View Prediction - Contrastive Predictive Coding pre-train Pio2Vox with YOLOv3](https://drive.google.com/file/d/1h16G4X8eopV4VoFVUfZZr2hF1wYa9UxK/view?usp=sharing)

Rank (out of 50 teams): 
* Detection: 9
* Segmentation: 2
* Overall: 4

# Instruction

## Step 1. CPC pretrain Pix2Vox

In this step we utilized unlabelled data and CPC to pretrian a ResNet-50 encoder. 
Command:
folder `./self_supervised/config.py` contains configurations of the model, when adjusted and simply run
```
python ./self_supervised/run.py
```
Important details about `config.py`:

* `resnet_model`, the encoder model of feature extractor, by default `resnet50` and lists are {`resnet18`, `resnet50`. `resnet101`, `resnet152`}
* `encoder_model`, the encoder model of seq2seq, by default `LSTM`, and lists are {`LSTM`, `GRU`, `RNN`}
* `embed_size`, the embedding size of feature extractor
* `rnn_hidden_size`, the hidden size of seq2seq
* `output_size`, the output size of the seq2seq
* `rnn_n_layers`, the number of layers of seq2seq (both encoder and decoder)
* `rnn_seq_len`, the sequence length of seq2seq
    

## Step 2. Fine Tune or train from scratch

Folder `./car_detection` contains all code train a Pix2Vox model for detection/segmentation task. To run this on your own environment, please first configure `./car_detection/data.py` for the dataloader. To run training, you will need to run `main.py` with following command line arguments:
* `-mc`, `--model-conifg`, setup the model configuration. The best model uses configuration `pix2vox`
* `-bs`, `--batch-size`, batch size of training
* `-dm`, yes or no, if yes, it only adds one scene into train/validation set, which is good for debugging
* `-det`, `-seg`, yes or no. Train detection or train segmentation model. Note, you can train them in a multi-tasking manner, but in practice, it does not converges.
* `-ssl`, yes or no, if you want to used pre-train weights. You have to configure the path of pre-train weights in `./car_detection/load_ssl.py` before set it to yes
* `-pt`, yes or no, if you want to use ImageNet pre-train weights for ResNet encoder
* `-a`, `-g`, float, weights parameter for Focal Loss and weighted cross entropy

To run the training, simply do 
```bash
# segmentation model from scratch
python main.py -mc pix2vox -bs 2 -pt no -det no -seg yes -ssl no

# detection model from scratch
python main.py -mc pix2vox -bs 2 -pt no -det yes -seg no -ssl no
```
After training, you can use `detect_pix.py` for the testing and figure generation.

# References
1. We refer https://github.com/eriklindernoren/PyTorch-YOLOv3 for YOLOv3 implementation
