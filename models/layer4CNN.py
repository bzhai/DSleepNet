import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist


class qzd(nn.Module):
    def __init__(self, in_channels=9, num_classes=3, fc_dim=7):
        super(qzd, self).__init__()

        self.n_feature = in_channels
        self.fc_dim = fc_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True, padding=(0, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(self.fc_dim * 128, num_classes))
        # self.fc12 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
    def forward(self, x):
        x_img = x.float()
        if x_img.shape[-1] == self.n_feature:
            x_img = x_img.view(-1, x_img.shape[2], x_img.shape[1])
        x_img = torch.unsqueeze(x_img, 2)

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
        y_prob = self.fc11(out)
        return y_prob