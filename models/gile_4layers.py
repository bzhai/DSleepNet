import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(px, self).__init__()
        self.la_step = args.la_step
        self.n_feature = args.n_feature

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, args.fc_dim, bias=False),
                                 nn.BatchNorm1d(args.fc_dim), nn.ReLU())

        self.un1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )

        self.un2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )

        self.un3 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )

        self.un4 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=self.n_feature, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv1[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv2[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv3[0].weight)
        torch.nn.init.xavier_uniform_(self.deconv4[0].weight)

    def forward(self, zd, zx, zy, idxs, sizes):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)
        h = h.view(-1, 128, 1, self.la_step)  # ??

        out_1 = self.un1(h, idxs[3], output_size=sizes[3])
        out_11 = self.deconv1(out_1)

        out_2 = self.un2(out_11, idxs[2], output_size=sizes[2])
        out_22 = self.deconv2(out_2)

        out_3 = self.un3(out_22, idxs[1], output_size=sizes[1])
        out_33 = self.deconv3(out_3)

        out_4 = self.un4(out_33, idxs[0], output_size=sizes[0])
        out_44 = self.deconv4(out_4)
        out = out_44.permute(0, 2, 3, 1)  # (256, 9, 1, 101) -> (256, 1, 101, 9)
        return out


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(pzd, self).__init__()
        self.d_dim = d_dim
        self.device = args.device
        # self.now_target_domain_int = int(args.target_domain[-1]) - 1
        # embedding part
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        d_onehot = torch.zeros(d.shape[0], self.d_dim)
        for idx, val in enumerate(d):
            d_onehot[idx][val.item()] = 1
        d_onehot = d_onehot.to(self.device)
        hidden = self.fc1(d_onehot)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(pzy, self).__init__()

        self.y_dim = y_dim
        self.device = args.device

        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        y_onehot = torch.zeros(y.shape[0], self.y_dim)
        for idx, val in enumerate(y):
            y_onehot[idx][val.item()] = 1

        y_onehot = y_onehot.to(self.device)

        hidden = self.fc1(y_onehot)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzd, self).__init__()

        self.n_feature = args.n_feature

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

        self.fc11 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        x_img = x.float()  # (256, 1, 101, 9)
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])  # input is N, C, H, W  (256, 1, 101, 9) -> (256, 9, 1, 101)
        # x_img = x_img.reshape(-1, x_img.shape[3], 1, x_img.shape[2])  # input is N, C, H, W

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3]) + 1e-6
        size0 = x_img.size()
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()

        zd_loc = self.fc11(out) + 1e-7
        zd_scale = self.fc12(out) + 1e-7

        return zd_loc, zd_scale, [idx1, idx2, idx3, idx4], [size0, size1, size2, size3]


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzx, self).__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=2, return_indices=True, ceil_mode=True)

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

        self.fc11 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
        size0 = x_img.size()
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()

        zx_loc = self.fc11(out)
        zx_scale = self.fc12(out) + 1e-7

        return zx_loc, zx_scale, [idx1, idx2, idx3, idx4], [size0, size1, size2, size3]


class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(qzy, self).__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=2, return_indices=True, ceil_mode=True)

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

        self.fc11 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(args.fc_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])
        # x_img = x_img.reshape(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])  # [64, 512]
        size0 = x_img.size()
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()

        zy_loc = self.fc11(out)
        zy_scale = self.fc12(out) + 1e-7

        return zy_loc, zy_scale, [idx1, idx2, idx3, idx4], [size0, size1, size2, size3]


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)
        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y
