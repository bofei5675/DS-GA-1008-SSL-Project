# Bird’s-eye View Prediction: Contrastive Predictive Coding pre-train Pio2Voxwith YOLOv3
Code for DS-GA-1008 Self-Supervised Learning(SSL) Final Project for Bofei Zhang, Can Cui, Yuanxi Sun

Rank (out of 50 teams): 
* Detection: 9
* Segmentation: 2
* Overall: 4

# Instruction

## Step 1. CPC pretrain Pix2Vox

In this step we utilized unlabelled data and CPC to pretrian a ResNet-50 encoder. 
Command:
```

```

## Step 2. Fine Tune or train from scratch

Folder `./car_detection` contains all code train a Pix2Vox model for detection/segmentation task. To run training, you will need to run `main.py` with following command line arguments
* `-mc`, `--model-conifg`, setup the model configuration. The best model uses configuration `pix2vox`
* `-bs`, `--batch-size`, batch size of training
* `-dm`, yes or no, if yes, it only adds one scene into train/validation set, which is good for debugging
* `-det`, `-seg`, yes or no. Train detection or train segmentation model. Note, you can train them in a multi-tasking manner, but in practice, it does not converges.
* `-ssl`, yes or no, if you want to used pre-train weights. You have to configure the path of pre-train weights in `./car_detection/load_ssl.py` before set it to yes
* `-pt`, yes or no, if you want to use ImageNet pre-train weights for ResNet encoder
* `-a`, `-g`, float, weights parameter for Focal Loss and weighted cross entropy

To run the training, simply do 
```bash
# segmentation model
python main.py -mc pix2vox -bs 2 -pt no -det no -seg yes -ssl no

# detection model
python main.py -mc pix2vox -bs 2 -pt no -det no -seg yes -ssl no
```
After training, you can use `detect_pix.py` for the testing and figure generation.

# References
1. We refer https://github.com/eriklindernoren/PyTorch-YOLOv3 for YOLOv3 implementation
