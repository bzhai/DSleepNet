# -*-coding:utf-8-*-
import torch.nn
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=32, output_size=3, num_layers=1, bidirectional=False, seq_len=101):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        # self.hidden = None
        self.seq_len = seq_len
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.linear = torch.nn.Linear(self.hidden_size*self.seq_len, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, input, hidden=None):
        # if self.hidden is None:
        #     output, self.hidden = self.lstm(input)
        # else:
        #     output, self.hidden = self.lstm(input, self.hidden)
        output, self.hidden = self.lstm(input)
        output = self.linear(output.reshape(-1, self.hidden_size * self.seq_len))
        # output = self.softmax(output)
        return output

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if not is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    lstm = LSTM()
    batch_size = 256
    input1 = torch.rand(batch_size, 101, 9)
    output1 = lstm(input1.to(device))
    print("output shape %s", output1.shape)