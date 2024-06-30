import torch.nn as nn
from resnet import *
import torch
from sagan import *
from causal_model import *
import math
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class Encoder(nn.Module):
    r'''ResNet Encoder

    Args:
        latent_dim: latent dimension
        arch: network architecture. Choices: resnet - resnet50, resnet18
        dist: encoder distribution. Choices: deterministic, gaussian, implicit
        fc_size: number of nodes in each fc layer
        noise_dim: dimension of input noise when an implicit encoder is used
    '''

    def __init__(self, latent_dim=64, in_channels=3, hidden_dims=None):
        super().__init__()  # 调用父类的初始化函数

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, x):
        '''
        :param x: input image
        :param avepool: whether to return the average pooling feature (used for downstream tasks)
        :return:
        '''
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


# encoder = Encoder()
# # print((encoder))
# x = torch.rand(16, 3, 64, 64)
# z_mu, z_a = encoder(x)
# print(z_mu, z_a)


class Decoder(nn.Module):
    r'''Big generator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        dist: generator distribution. Choices: deterministic, gaussian, implicit
        g_std: scaling the standard deviation of the gaussian generator. Default: 1
    '''

    def __init__(self, latent_dim=64, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.att = Self_Attn(hidden_dims[-1])

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())



    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.att(result)
        result = self.final_layer(result)
        return result


# decoder = Decoder()
# z = torch.rand(16, 64)
# x = decoder(z)
# print(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class WVAE(nn.Module):

    def __init__(self, latent_dim=64, conv_dim=32, image_size=64,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128, dec_dist='implicit',
                 prior='gaussian', num_label=None, A=None, reconstruction_loss='mse', use_mss=True, alpha=1, beta=4,
                 gamma=1, edges = None, in_channels=3, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.prior_dist = prior
        self.num_label = num_label
        self.reconstruction_loss = reconstruction_loss
        self.use_mss = use_mss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.edges = edges




        self.encoder = Encoder(latent_dim, in_channels, hidden_dims)
        self.decoder = Decoder(latent_dim, hidden_dims)
        self.GAT = GAT(1, 1, 1, 2)
        if 'scm' in prior:
            self.prior = SCM(num_label, A, scm_type=prior)


    def traverse(self, eps, gap=8, n=10):
        dim = self.num_label if self.num_label is not None else self.latent_dim  # 如果self.num_label不是None，就把它赋值给dim，否则就把self.latent_dim赋值给dim
        sample = torch.zeros((n * (dim+1), 3, self.image_size, self.image_size))  # 生成一个全零张量，并赋值给sample
        eps = eps.expand(n, self.latent_dim)  # 把eps扩展到(n, self.latent_dim)的维度，并赋值给eps
        if self.prior_dist == 'gaussian' or self.prior_dist == 'uniform':  # 如果self.prior_dist是'gaussian'或者'uniform'
            z = eps  # 把eps赋值给z
        else:  # 否则
            label_z = self.prior(eps[:, :dim])  # 调 用self.prior，得到eps的前dim个维度经过因果层后的输出，并赋值给label_z
            label_z = self.gae(label_z)
            other_z = eps[:, dim:]  # 得到eps的剩余维度，并赋值给other_z
            z = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z
        for idx in range(dim):  # 循环dim次
            traversals = torch.linspace(-gap, gap, steps=n)  # 生成一个在[-gap, gap]区间上均匀分布的张量，并赋值给traversals
            z_new = z.clone()  # 复制z，并赋值给z_new
            z_new[:, idx] = traversals  # 把traversals赋值给z_new的第idx个维度
            # z_new_p = self.gae(z_new[:, :dim])
            # z_new = torch.cat([z_new_p, z_new[:, dim:]], dim=1)
            # z_new[:, idx] = traversals
            with torch.no_grad():  # 不计算梯度
                sample[n * idx:(n * (idx + 1)), :, :, :] = self.decoder(z_new)  # 调用self.decoder，得到生成的图像，并赋值给sample的相应位置
        sample[n * (dim+1) - n:n * (dim+1), :, :, :] = self.decoder(z)
        return sample  # 返回sample

    def forward(self, x=None, z=None, recon=False, gen=False, infer_mean=True):
        # recon_mean is used for gaussian decoder which we do not use here.
        # Training Mode
        if x is not None and z is None:  # 如果x和z都不是None
            if self.enc_dist == 'gaussian':  # 如果self.enc_dist是'gaussian'
                z_mu, z_logvar = self.encoder(x)  # 调用self.encoder，得到隐变量的均值和对数方差，并赋值给z_mu和z_logvar
                z_logvar = torch.ones(z_logvar.shape, device=x.device)
            else:  # deterministic or implicit
                z = self.encoder(x)  # 调用self.encoder，得到隐变量，并赋值给z_fake


            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                # in prior
                label_z_q = self.prior(z_mu[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
                label_z_q = self.gae(label_z_q)
                other_z_q = z_mu[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
                z_mu_fake = torch.cat([label_z_q, other_z_q], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z


            z_logvar_fake = torch.ones(z_logvar.shape, device=x.device)
            z = reparameterize(z_mu_fake, z_logvar_fake)

            x_fake = self.decoder(z)  # 调用self.decoder，得到生成的图像，并赋值给x_fake

            if recon is True:
                return x_fake

            return x_fake, z, z_mu_fake, z_logvar_fake, z_mu, z_logvar

            # if recon == True:
            #     return x_fake

            # if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
            #     if self.enc_dist == 'gaussian' and infer_mean:  # 如果self.enc_dist是'gaussian'并且infer_mean为真
            #         return z_fake, x_fake, z, z_mu, z_logvar  # 返回z_fake, x_fake, z, z_mu
            #     else:  # 否则
            #         return z_fake, x_fake, z, None, z_logvar  # 返回z_fake, x_fake, z, None
            # return z_fake, x_fake, z_mu, z_logvar  # 返回z_fake, x_fake, z_mu



        # Generation Mode
        elif x is None and z is not None:  # 如果x是None而z不是None
            if 'scm' in self.prior_dist:  # 如果self.prior_dist中包含'scm'
                label_z = self.prior(z[:, :self.num_label])  # 调用self.prior，得到z的前self.num_label个维度经过因果层后的输出，并赋值给label_z
                label_z = self.gae(label_z)
                other_z = z[:, self.num_label:]  # 得到z的剩余维度，并赋值给other_z
                z = torch.cat([label_z, other_z], dim=1)  # 把label_z和other_z在第二个维度上拼接起来，并赋值给z

            x_fake = self.decoder(z)
            return  x_fake

            # if gen == True:
            #     return x_fake

            # if self.enc_dist == 'gaussian':  # 如果self.enc_dist是'gaussian'
            #     z_mu, z_logvar = self.encoder(x_fake)  # 调用self.encoder，得到隐变量的均值和对数方差，并赋值给z_mu和z_logvar
            #     z_fake = reparameterize(z_mu, torch.exp(0.5 * z_logvar))
            # else:  # deterministic or implicit
            #     z_fake = self.encoder(x_fake)  # 调用self.encoder，得到隐变量，并赋值给z_fake
            #
            # return z_fake, z_mu, z_logvar, x_fake, z  # 返回调用self.decoder后的结果



    def _compute_log_gauss_density(self, z, mu, log_var):
        """element-wise computation"""
        return -0.5 * (
                torch.log(torch.tensor([2 * np.pi]).to(z.device))
                + log_var
                + (z - mu) ** 2 * torch.exp(-log_var)
        )
    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """Compute importance weigth matrix for MSS
        Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """

        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def getfeature(self, x, discriminator):

        feature_layers = ['block2', 'block3', 'block4', 'block5']
        features = []
        result = x
        for (key, module) in discriminator.module.discriminator._modules.items():
            if key == 'snlinear1':
                feature_x = torch.sum(result, dim=[2, 3])
                sx = torch.squeeze(module(feature_x))
                break
            result = module(result)
            if (key in feature_layers):
                features.append(result)

        features.append(feature_x)
        return features

    def loss_function(self, recon_x, x, discriminator):

        if self.reconstruction_loss == "mse":

            # recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='sum') / x.shape[0]


            feature_loss = 0.0
            recon_feature = self.getfeature(recon_x, discriminator)
            x_feature = self.getfeature(x, discriminator)
            for (r, i) in zip(recon_feature, x_feature):
                feature_loss += F.mse_loss(r, i, reduction='sum') / x.shape[0]

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)


        return (self.alpha * feature_loss)

    def gae(self, x):
        x = x.t()
        processed_columns = []
        for i in range(x.size(1)):
            column = x[:, i].unsqueeze(1)
            processed_columns.append(self.GAT(column, self.edges))

        result = torch.cat(processed_columns, dim=1)
        return result.t()
        # return (
        #     (
        #             recon_loss
        #             + self.alpha * mutual_info_loss
        #             + self.beta * TC_loss
        #             + self.gamma * dimension_wise_KL
        #     ).mean(dim=0),
        #     recon_loss.mean(dim=0),
        #     (
        #             self.alpha * mutual_info_loss
        #             + self.beta * TC_loss
        #             + self.gamma * dimension_wise_KL
        #     ).mean(dim=0),
        # )


class BigJointDiscriminator(nn.Module):
    r'''Big joint discriminator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        fc_size: number of nodes in each fc layers
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, fc_size=1024):
        super().__init__()
        self.discriminator = Discriminator(conv_dim, image_size, in_channels=3, out_feature=True) # 创建一个判别器，并把它赋值给self.discriminator
        self.discriminator_z = Discriminator_MLP(latent_dim, fc_size) # 创建一个多层感知器的判别器，并把它赋值给self.discriminator_z
        self.discriminator_j = Discriminator_MLP(conv_dim * 16 + fc_size, fc_size) # 创建一个多层感知器的判别器，并把它赋值给self.discriminator_j

    def forward(self, x=None, z=None):
        if x is not None and z is not None:
            sx, feature_x = self.discriminator(x)
            sz, feature_z = self.discriminator_z(z)
            sxz, _ = self.discriminator_j(torch.cat((feature_x, feature_z), dim=1))
            return (sx + sz + sxz) / 3
        elif x is not None and z is None:
            sx, feature_x = self.discriminator(x)
            # sx_f, _ = self.discriminator_j(feature_x)
            return sx, feature_x
        elif x is None and z is not None:
            sz, feature_z = self.discriminator_z(z)
            # sx_f, _ = self.discriminator_j(feature_x)
            return sz, feature_z

