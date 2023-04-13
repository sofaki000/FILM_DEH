# enc_in: Integer, the number of input features for the encoder.
# dec_in: Integer, the number of input features for the decoder.
# d_model: Integer, the hidden size of the model.
# embed: Integer, the embedding size.
# freq: String, the frequency of the input time series data. It could be one of the following: 'h' for hourly, 'd' for daily, 'b' for business days, 'w' for weekly, 'm' for monthly, or 'q' for quarterly.
# dropout: Float, the dropout rate for regularization.
# factor: Integer, the factor used in the ProbSparse Attention layer.
# n_heads: Integer, the number of attention heads.
# e_layers: Integer, the number of encoder layers.
# d_layers: Integer, the number of decoder layers.
# d_ff: Integer, the hidden size of the feed-forward layer.
# distil: Boolean, whether to use distillation in the encoder or not.
# activation: String, the activation function used in the model. It could be one of the following: 'relu', 'gelu', 'leaky_relu', 'tanh', or 'sigmoid'.
# c_out: Integer, the number of output channels

def getConfig():
    configs = {
        'distil':False,
        'enc_in': 1,  # Number of input features for encoder
        'dec_in': 1,  # Number of input features for decoder
        'd_model': 256,  # Dimension of the model
        'embed': 'fixed',  # Type of data embedding ('fixed', 'learnable')
        'freq': 'h',  # Frequency of the time series data ('t', 'h', 'd', 'b', 'w', 'm')
        'dropout': 0.1,  # Dropout rate
        'e_layers': 2,  # Number of encoder layers
        'd_layers': 1,  # Number of decoder layers
        'n_heads': 4,  # Number of heads in multi-head attention
        'd_ff': 4,  # Dimension of the feed-forward layer
        'activation': 'gelu',  # Activation function ('relu', 'gelu')
        'factor': 5,  # Factor for ProbSparse attention
        'output_attention': False,  # Whether to output attention weights
        'pred_len': 24,  # Length of the predicted output sequence
        'c_out': 1  # Number of output features
    }

    return configs