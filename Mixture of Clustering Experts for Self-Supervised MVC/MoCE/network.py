import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.decoder(x)



class GatingNet(nn.Module):
    def __init__(self, clusters, input_dim, num_experts, topk=3):
        super(GatingNet, self).__init__()
        self.topk = topk
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
        )
        self.NoiseNet = NoiseEncoder(clusters, num_experts, 0.3)

    def forward(self, z, label):
        noise = self.NoiseNet(label)    # [B E}
        logits = self.net(z)  # [B E]
        probs = F.softmax(logits + noise, dim=-1)


        return probs


class cross_expert(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(cross_expert, self).__init__()

        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, z_v):
        out = self.out(z_v)
        return out


class expert(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_clusters, n_views, nu=0.2):
        super(expert, self).__init__()

        self.prototypes = nn.Parameter(torch.randn(n_clusters, input_dim))  # [C, D]
        nn.init.xavier_uniform_(self.prototypes)
        self.n_views = n_views
        self.n_clusters = n_clusters
        self.nu = nu
        self.scale = nn.Parameter(torch.tensor(input_dim ** 0.5))

        self.models = nn.ModuleList()
        for i in range(n_clusters):
            self.models.append(cross_expert(input_dim, hidden_dim))

    def forward(self, z):

        B, D = z.shape
        C = self.n_clusters
        M = self.n_clusters

        expert_outs = []
        for c in range(C):
            z_out = self.models[c](F.normalize(z, dim=1))
            expert_outs.append(z_out)
        expert_outs = torch.stack(expert_outs, dim=-1)  # [B D M]
        proto = F.normalize(self.prototypes, dim=1)

        q_list = []
        for m in range(M):
            z_m = expert_outs[:, :, m]  # [B, D]

            # consine
            cos_sim = z_m @ proto.T
            q = F.softmax(cos_sim / self.nu, dim=1)  # [B, C]
            q_list.append(q)
        q_tensor = torch.stack(q_list, dim=-1)  # [B, C, M]

        return q_tensor




class NoiseEncoder(nn.Module):
    def __init__(self, cluster, noise_dim, semi_noise_scale=0.1):
        super(NoiseEncoder, self).__init__()
        self.semi_noise_scale = semi_noise_scale
        self.fc = nn.Sequential(
            nn.Linear(cluster, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * noise_dim)
        )

    def forward(self, pseudo_labels):
        mu_logvar = self.fc(pseudo_labels)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z_l = mu + eps * std

        semi_noise = torch.randn_like(z_l) * self.semi_noise_scale
        z_l = z_l + semi_noise

        return z_l


class Network(nn.Module):
    def __init__(self, input_dim, feature_dim, view_shape, alpha, clusters, device):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.view_shape = view_shape
        self.alpha = alpha
        self.clusters = clusters
        self.device = device
        self.feature_dim = feature_dim
        self.encoders = []
        self.decoders = []
        self.cluexpert = []
        self.num_experts = clusters
        for v in range(self.view_shape):
            self.encoders.append(Encoder(input_dim[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_dim[v], feature_dim).to(device))


        self.gating = GatingNet(clusters, feature_dim, self.num_experts).to(device)
        self.experts = expert(feature_dim, 128, clusters, view_shape)

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x, label):
        expert_crossv = []

        encoders = [encoder(x_v) for encoder, x_v in zip(self.encoders, x)]
        # decoders = [decoder(y_v) for decoder, y_v in zip(self.decoders, encoders)]

        # fuse
        q_list = []
        gate_list = []
        for v in range(self.view_shape):
            q_v = self.experts(encoders[v])  # [B, C, M]
            gate = self.gating(encoders[v], label) # [B E]
            q_v = torch.einsum('bcm,bm->bc', q_v, gate)  # [B, C]
            q_list.append(q_v)
            gate_list.append(gate)

        q_list = torch.stack(q_list, dim=0) # [V, B, C]


        prototypes = self.experts.prototypes  # C D

        fused_features = [torch.matmul(q_fv, prototypes) for q_fv in q_list]
        decoders = [decoder(y_v) for decoder, y_v in zip(self.decoders, fused_features)]

        return q_list, decoders, encoders, prototypes, gate_list


