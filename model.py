import torch
from torch import nn
from torch.nn import functional as F


class Causal_HMM(nn.Module):
    def __init__(self, args, z_size, v_size, s_size, A_size, B_size, batch_size, layer_count=3, channels=3):
        super(Causal_HMM, self).__init__()

        d = args.image_size
        self.d = d
        self.z_size = z_size
        self.s_size = s_size
        self.v_size = v_size
        self.batch_size = batch_size
        self.args = args

        self.h_dim = z_size + v_size + s_size

        self.dim = int(d / (2 ** layer_count))

        # image encoder
        self.layer_count = layer_count

        inputs = channels
        mul = 1
        for i in range(self.layer_count):
            setattr(self, "post_conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "post_conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        # image decoder
        self.d1 = nn.Linear(self.h_dim, inputs * self.dim * self.dim)

        # decoder for A
        self.fc_decode_A = nn.Linear(v_size, A_size)

        # prior encoder for B_t, h_t
        b_feature_dim = args.fc_dim
        self.fc_b_layer = nn.Linear(B_size, b_feature_dim)

        self.prior_gru_z = nn.GRUCell(self.z_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_z.bias_ih.data.fill_(0)
        self.prior_gru_z.bias_hh.data.fill_(0)

        self.prior_gru_s = nn.GRUCell(self.s_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_s.bias_ih.data.fill_(0)
        self.prior_gru_s.bias_hh.data.fill_(0)

        self.prior_gru_v = nn.GRUCell(self.v_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_v.bias_ih.data.fill_(0)
        self.prior_gru_v.bias_hh.data.fill_(0)

        # prior encoder
        self.fc_z_mu = nn.Linear(self.args.lstm_out, self.z_size)
        self.fc_z_logvar = nn.Linear(self.args.lstm_out, self.z_size)

        self.fc_s_mu = nn.Linear(self.args.lstm_out, self.s_size)
        self.fc_s_logvar = nn.Linear(self.args.lstm_out, self.s_size)

        self.fc_v_mu = nn.Linear(self.args.lstm_out, self.v_size)
        self.fc_v_logvar = nn.Linear(self.args.lstm_out, self.v_size)

        # posterior encoder
        x_feature_dim = self.d_max
        a_feature_dim = args.fc_dim

        self.post_encode_A = nn.Linear(A_size, a_feature_dim)
        self.post_encode_B = nn.Linear(B_size, b_feature_dim)

        self.post_z_fc = nn.Linear(x_feature_dim + b_feature_dim + self.z_size, self.args.fc_dim)
        self.post_z1 = nn.Linear(self.args.fc_dim, self.z_size)
        self.post_z2 = nn.Linear(self.args.fc_dim, self.z_size)

        self.post_v_fc = nn.Linear(x_feature_dim + a_feature_dim + b_feature_dim + self.v_size, self.args.fc_dim)
        self.post_v1 = nn.Linear(self.args.fc_dim, self.v_size)
        self.post_v2 = nn.Linear(self.args.fc_dim, self.v_size)

        self.post_s_fc = nn.Linear(x_feature_dim + a_feature_dim + b_feature_dim + self.s_size, self.args.fc_dim)
        self.post_s1 = nn.Linear(self.args.fc_dim, self.s_size)
        self.post_s2 = nn.Linear(self.args.fc_dim, self.s_size)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))

            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))


    def prior_encode(self, B_t_last, z_t_last, s_t_last, v_t_last):

        b = self.fc_b_layer(B_t_last)

        z_gru_input = torch.cat((b, z_t_last), 1)
        s_gru_input = torch.cat((b, s_t_last), 1)
        v_gru_input = torch.cat((b, v_t_last), 1)

        z_t = self.prior_gru_z(z_gru_input)
        s_t = self.prior_gru_s(s_gru_input)
        v_t = self.prior_gru_v(v_gru_input)

        mu_z = self.fc_z_mu(z_t)
        logvar_z = self.fc_z_logvar(z_t)

        mu_s = self.fc_s_mu(s_t)
        logvar_s = self.fc_s_logvar(s_t)

        mu_v = self.fc_v_mu(v_t)
        logvar_v = self.fc_v_logvar(v_t)

        return mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v


    def post_encode(self, x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last):

        for i in range(self.layer_count):
            x_t = F.relu(getattr(self, "post_conv%d_bn" % (i + 1))(getattr(self, "post_conv%d" % (i + 1))(x_t)))

        x_t = torch.nn.functional.adaptive_avg_pool2d(x_t, (1, 1))

        x_t = x_t.view(x_t.shape[0], self.d_max)
        feature_A = self.post_encode_A(A_t)
        feature_B = self.post_encode_B(B_t_last)

        f_z = torch.cat((x_t, feature_B, z_t_last), 1)
        f_z = self.post_z_fc(f_z)
        mu_z = self.post_z1(f_z)
        logvar_z = self.post_z2(f_z)

        f_s = torch.cat((x_t, feature_A, feature_B, s_t_last), 1)
        f_s = self.post_s_fc(f_s)
        mu_s = self.post_s1(f_s)
        logvar_s = self.post_s2(f_s)

        f_v = torch.cat((x_t, feature_A, feature_B, v_t_last), 1)
        f_v = self.post_v_fc(f_v)
        mu_v = self.post_v1(f_v)
        logvar_v = self.post_v2(f_v)

        return mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v

    def reparameterize(self, mu, logvar, test):

        if not test:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_x(self, x):
        x = x.view(x.shape[0], self.h_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.dim, self.dim)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))

        return x

    def decode_A(self, v_t):
        A = self.fc_decode_A(v_t)
        return A

    def forward(self, x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last, test=False):

        mu_z_prior, logvar_z_prior,\
        mu_s_prior, logvar_s_prior, mu_v_prior, logvar_v_prior = self.prior_encode(B_t_last, z_t_last, s_t_last, v_t_last)
        mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v = self.post_encode(x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last)

        mu_h = torch.cat((mu_z, mu_s, mu_v), 1)
        logvar_h = torch.cat((logvar_z, logvar_s, logvar_v), 1)

        mu_prior = torch.cat((mu_z_prior, mu_s_prior, mu_v_prior), 1)
        logvar_prior = torch.cat((logvar_z_prior, logvar_s_prior, logvar_v_prior), 1)

        h_t = self.reparameterize(mu_h, logvar_h, test)
        v_t = self.reparameterize(mu_v, logvar_v, test)

        vae_rec_x_t = self.decode_x(h_t.view(-1, self.h_dim, 1, 1))
        vae_rec_A = self.decode_A(v_t)

        return vae_rec_x_t, vae_rec_A, mu_h, logvar_h, \
               mu_prior, logvar_prior, mu_z, mu_s, mu_v, logvar_v, mu_v_prior, logvar_v_prior

    def _init_papameters(self, args):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRUCell):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal(m.weight_hh.data)
                    nn.init.kaiming_normal(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight_hh.data)
                    nn.init.xavier_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)


class Disease_Classifier(nn.Module):
    def __init__(self, args, in_dim):
        super(Disease_Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, args.cls_fc_dim)
        self.fc2 = nn.Linear(args.cls_fc_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


