import torch
import torch.nn as nn

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        _, (hidden, _) = self.lstm(input)
        return hidden

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, _ = self.lstm(input, hidden)
        output = self.fc(output)
        return output

# Define the Encoder-Decoder class
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size, output_size)

    def forward(self, input):
        hidden = self.encoder(input)
        output = self.decoder(input, hidden)
        return output