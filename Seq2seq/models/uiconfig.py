class UIconfig:
    seq_len = 100
    batch_size = 32
    seed = 391275
    ignore_index = 1  #padding - dont calculate attention
    lr_main = 0.01
    drop_out = 0.1
    epochs = 2

    #transformer params
    nhead = 4
    num_encoder_layers = 2
    d_model = 64