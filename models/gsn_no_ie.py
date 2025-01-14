import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from models.gile_4layers import px, pzy, qzy, qzd, qzx, qd, qy
from models.gile_4layers_mse import pzd

class GSN_NO_IE(nn.Module):
    """ GSN with no independent excitation"""
    def __init__(self, args):
        super(GSN_NO_IE, self).__init__()
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

        #  Encode
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

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q


    def loss_function(self, d, x, y=None):
        d = d.float()
        y = y.long()

        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        x = torch.unsqueeze(x, 1)
        CE_x = F.mse_loss(x_recon, x.float())  # reconstruction loss

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))  # ? KL domain specific
        if self.zx_dim != 0:
            KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
        else:
            KL_zx = 0

        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))  # ? KL domain ago
        MSE_d = F.mse_loss(d_hat, d, reduction='sum')
        CE_y = F.cross_entropy(y_hat, y, reduction='sum')
        # ?
        return CE_x \
               - self.beta_d * zd_p_minus_zd_q \
               - self.beta_x * KL_zx \
               - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * MSE_d \
               + self.aux_loss_multiplier_y * CE_y, \
               CE_y, y_hat, MSE_d


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

        return d, y, d_false, alpha_y, d_false, y_false

    def get_features(self, x):
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()
        return zy_q