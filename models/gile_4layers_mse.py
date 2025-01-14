import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from models.gile_4layers import px, pzy, qzy, qzd, qzx, qd, qy


# Auxiliary tasks qd qy
class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super(pzd, self).__init__()
        self.d_dim = d_dim
        self.device = args.device
        # self.now_target_domain_int = int(args.target_domain[-1]) - 1
        # embedding part
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=True), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        # d_onehot = torch.zeros(d.shape[0], self.d_dim)
        # for idx, val in enumerate(d):
        #     d_onehot[idx][val.item()] = 1
        # d_onehot = d_onehot.to(self.device)
        if (len(d.shape)<=1):
            d = torch.unsqueeze(d, 1)
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale


class GSNMSE3DIS(nn.Module):
    """ this is the baseline model I got from MakeSenseSleep HP tuning results"""
    def __init__(self, args):
        super(GSNMSE3DIS, self).__init__()
        self.zd_dim = args.d_AE
        self.zx_dim = 0
        self.zy_dim = args.d_AE
        self.d_dim = len(args.dis_type) - len(args.mask_att)
        self.x_dim = args.x_dim
        self.y_dim = args.n_class

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.cuda()

    def forward(self, d, x, y):
        x = torch.unsqueeze(x, 1)

        d = d.float()
        y = y.long()

        # Encode
        zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale, _, _ = self.qzx(x)
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q, idxs_y, sizes_y)

        zd_p_loc, zd_p_scale = self.pzd(d)  # prior for d KL(qzd||pzd)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)  # prior for y KL(qzy||pzy)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)  # prior for d KL(qzd||pzd)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)  # prior for y KL(qzy||pzy)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):
        d = d.float()
        y = y.long()

        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        x = torch.unsqueeze(x, 1)
        MSE_x = F.mse_loss(x_recon, x.float())  # reconstruction loss

        zd_p_minus_zd_q = self.beta_d * (torch.mean(pzd.log_prob(zd_q) - qzd.log_prob(zd_q)))  # ? KL domain specific
        if self.zx_dim != 0:
            KL_zx = self.beta_x * (torch.mean(pzx.log_prob(zx_q) - qzx.log_prob(zx_q)))
        else:
            KL_zx = 0

        zy_p_minus_zy_q = self.beta_y * (torch.mean(pzy.log_prob(zy_q) - qzy.log_prob(zy_q)))  # ? KL domain agnostic
        # if d.shape[-1] == 5 and len(set(d.cpu().numpy()[:, 0])) == 1:
        #     MSE_d = F.mse_loss(torch.squeeze(d_hat[:, 2:]), d[:, 1:], reduction='mean')
        #     CE_d = F.cross_entropy(d_hat[:, :2], d[:, 0], reduction="sum")
        #     MSE_d +=CE_d
        # else:
        if len(d_hat.shape) > 1 and len(d.shape)==1:
            d_hat = torch.squeeze(d_hat)
        MSE_d = self.aux_loss_multiplier_d * F.mse_loss(d_hat, d)
        CE_y = self.aux_loss_multiplier_y * F.cross_entropy(y_hat, y, reduction='sum')

        return (MSE_x \
               - zd_p_minus_zd_q \
               - KL_zx \
               - zy_p_minus_zy_q) \
               + MSE_d \
               + CE_y, \
               CE_y, y_hat, MSE_d, -zd_p_minus_zd_q, -zy_p_minus_zy_q, MSE_x
    def loss_function_false(self, args, d, x, y=None):
        """
        Independence Excitation loss
        @param args:
        @param d:
        @param x:
        @param y:
        @return:
        """
        d = d.long()
        y = y.long()

        pred_d, pred_y, _, _, pred_d_false, pred_y_false = self.classifier(x)
        if len(pred_d.shape) > 1 and len(d.shape) == 1:
            pred_d = torch.squeeze(pred_d)
        if len(pred_d_false.shape) > 1 and len(d.shape) == 1:
            pred_d_false = torch.squeeze(pred_d_false)
        # if d.shape[-1] == 5 and len(set(d.cpu().numpy()[:, 0])) == 1:
        #     loss_classify_true = args.weight_true * (
        #             F.mse_loss(pred_d[:, 2:], d[:, 1:], reduction='mean') +
        #             F.cross_entropy(pred_d[:, :2], d[:, 0], reduction='mean') +
        #             F.cross_entropy(pred_y, y, reduction='sum'))
        #     loss_classify_false = args.weight_false * (
        #             F.mse_loss(pred_d_false[:, 2:], d[:, 1:], reduction='mean') +
        #             F.mse_loss(pred_d_false[:, :2], d[:, 0], reduction='mean') +
        #             F.cross_entropy(pred_y_false, y, reduction='sum'))
        # else:
        loss_classify_true = F.mse_loss(pred_d, d) + F.cross_entropy(pred_y, y, reduction='sum')
        loss_classify_true = args.weight_true * loss_classify_true
        loss_classify_false = F.mse_loss(pred_d_false, d) + F.cross_entropy(pred_y_false, y, reduction='sum')
        loss_classify_false = args.weight_false * loss_classify_false
        loss = loss_classify_true - loss_classify_false

        loss.requires_grad = True

        return loss

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():

            x = torch.unsqueeze(x, 1)

            zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
            zd = zd_q_loc
            # alpha_d = F.softmax(self.qd(zd), dim=1)
            d = self.qd(zd)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability


            zy_q_loc, zy_q_scale, _, _ = self.qzy.forward(x)
            zy = zy_q_loc
            alpha_y = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            _, ind = torch.topk(alpha_y, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha_y.size())
            y = y.scatter_(1, ind, 1.0)
            # this was the classification
            # alpha_y2d = F.softmax(self.qd(zy), dim=1)
            d_false = self.qd(zy)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability


            alpha_d2y = F.softmax(self.qy(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            _, ind = torch.topk(alpha_d2y, 1)

            # convert the digit(s) to one-hot tensor(s)
            y_false = x.new_zeros(alpha_d2y.size())
            y_false = y_false.scatter_(1, ind, 1.0)

        return d, y, alpha_d2y, alpha_y, d_false, y_false

    def get_features(self, x):
        x = torch.unsqueeze(x, 1)
        zy_q_loc, zy_q_scale, _, _ = self.qzy(x)
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        return zy_q, zd_q

