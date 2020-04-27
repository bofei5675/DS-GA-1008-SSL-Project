class Args:
    resnet_model = 'resnet50'
    encoder_model = 'LSTM'
    decoder_model = 'LSTM'
    embed_size = 128
    rnn_hidden_size = 64
    output_size = 128
    rnn_n_layers = 2
    rnn_seq_len = 3
    valid_size = 0.05

    temperature = 0.5
    encoder_update_every_n_steps = 5
    weight_decay = 10e-6

    # adjust_batch_size
    batch_size = 10
    device = 'cuda'
    epochs = 80

    data_dir = '../data'
    log_dir = '../runs/logs'
    mdl_dir = '../runs/bestmodels'
    scripts_dir = '../runs/scripts'
    eval_every_n_epochs = 1
    log_every_n_steps = 30
    pre_train=False
