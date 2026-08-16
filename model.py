import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Graph convolution layer for functional connectivity networks."""

    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GCNLayer, self).__init__()

        self.add_self = add_self
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.001 else None

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        y = torch.matmul(adj, x)

        if self.add_self:
            y += x

        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias

        return y


class MLPClassifier(nn.Module):
    """Evidence head applied to GCN node representations."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation=nn.ReLU):
        super(MLPClassifier, self).__init__()

        layers = []
        last_size = input_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            last_size = hidden

        layers.append(nn.Linear(last_size, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        x = x.mean(dim=2).float()
        return x


def kl_divergence(alpha, c):
    """KL divergence between a Dirichlet distribution and a uniform prior."""
    beta = torch.ones((1, c), device=alpha.device, dtype=alpha.dtype)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def evidence_loss(p, alpha, c, global_step, annealing_step):
    """Annealed evidential classification loss used by SADLE."""
    S = torch.sum(alpha, dim=1, keepdim=True)

    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * kl_divergence(alp, c)
    return torch.mean((A + B))


class SADLE(nn.Module):
    """Subject-Adaptive Dual-Level Evidential fusion network."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dims = {
            'pc': args.hidden_dim_pc,
            'hofc': args.hidden_dim_hofc,
            'sr': args.hidden_dim_sr,
            'mi': args.hidden_dim_mi
        }
        self.num_features = args.num_features
        self.window_size = args.window_size
        self.step_size = args.step_size
        self.n_classes = args.n_classes
        self.alphas = args.alphas
        self.n_bins = args.n_bins
        self.gcn_encoders = nn.ModuleDict({
            'pc': GCNLayer(self.num_features, self.hidden_dims['pc'],
                              dropout=args.dropout_gcn_pc),
            'hofc': GCNLayer(self.num_features, self.hidden_dims['hofc'],
                                dropout=args.dropout_gcn_hofc),
            'sr': GCNLayer(self.num_features, self.hidden_dims['sr'],
                              dropout=args.dropout_gcn_sr),
            'mi': GCNLayer(self.num_features, self.hidden_dims['mi'],
                              dropout=args.dropout_gcn_mi),
        })

        self.mlp_classifiers = nn.ModuleDict({
            'pc': MLPClassifier(self.hidden_dims['pc'], args.mlp_hidden_pc,
                                self.n_classes, dropout=args.dropout_mlp_pc),
            'hofc': MLPClassifier(self.hidden_dims['hofc'], args.mlp_hidden_hofc,
                                  self.n_classes, dropout=args.dropout_mlp_hofc),
            'sr': MLPClassifier(self.hidden_dims['sr'], args.mlp_hidden_sr,
                                self.n_classes, dropout=args.dropout_mlp_sr),
            'mi': MLPClassifier(self.hidden_dims['mi'], args.mlp_hidden_mi,
                                self.n_classes, dropout=args.dropout_mlp_mi),
        })

        self.softplus = nn.Softplus()

    def sliding_window(self, data, window_size, step_size):
        """Split one subject's time series into overlapping temporal windows."""
        if data.shape[0] < window_size:
            raise ValueError(
                f"Expected at least {window_size} time points, got {data.shape[0]}."
            )
        return data.unfold(0, window_size, step_size).permute(0, 2, 1)

    def compute_pc_graph(self, data, compute_hofc=False, eps=1e-6):
        """Compute Pearson-correlation and optional high-order FC graphs."""
        batch_size, W, N = data.shape
        data = data - data.mean(dim=1, keepdim=True)
        cov = torch.matmul(data.transpose(1, 2), data) / (W - 1)
        std = torch.std(data, dim=1, unbiased=False)
        outer_std = torch.bmm(std.unsqueeze(2), std.unsqueeze(1)) + eps
        pc_full = cov / outer_std

        triu_indices = torch.triu_indices(N, N, offset=1, device=data.device)
        pc_graph = torch.zeros_like(pc_full)
        pc_graph[:, triu_indices[0], triu_indices[1]] = pc_full[:, triu_indices[0], triu_indices[1]]
        pc_graph = pc_graph + pc_graph.transpose(1, 2)

        if compute_hofc:
            row_mean = pc_graph.mean(dim=2, keepdim=True)
            row_std = pc_graph.std(dim=2, keepdim=True) + eps
            pc_norm = (pc_graph - row_mean) / row_std
            hofc_full = torch.bmm(pc_norm, pc_norm.transpose(1, 2))
            hofc_graph = torch.zeros_like(hofc_full)
            hofc_graph[:, triu_indices[0], triu_indices[1]] = hofc_full[:, triu_indices[0], triu_indices[1]]
            hofc_graph = hofc_graph + hofc_graph.transpose(1, 2)
            return pc_graph, hofc_graph
        else:
            return pc_graph

    def compute_mi_graph(self, data, eps=1e-10):
        """Compute the pairwise mutual-information graph."""
        B, W, N = data.shape
        device = data.device

        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        bins = torch.linspace(0, 1, self.n_bins + 1, device=device).view(1, 1, 1, -1)

        norm_data = (data - min_val) / (max_val - min_val + eps)
        norm_data = norm_data.unsqueeze(-1).expand(-1, -1, -1, self.n_bins + 1)
        bin_idx = torch.sum(norm_data > bins, dim=-1) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        mi_graphs = torch.zeros(B, N, N, device=device)
        bin_idx_onehot = F.one_hot(bin_idx, num_classes=self.n_bins).float()

        for i in range(N):
            x = bin_idx_onehot[:, :, i, :]
            y = bin_idx_onehot[:, :, i + 1:, :]

            joint = torch.einsum('bwi,bwjk->bikj', x, y)
            joint_prob = joint / W

            x_prob = joint_prob.sum(dim=2)
            y_prob = joint_prob.sum(dim=1)

            log_ratio = torch.log(joint_prob / (x_prob.unsqueeze(2) * y_prob.unsqueeze(1) + eps) + eps)
            mi = (joint_prob * log_ratio).sum(dim=(1, 2))

            mi_graphs[:, i, i + 1:] = mi
            mi_graphs[:, i + 1:, i] = mi

        return mi_graphs

    def compute_sr_graph(self, data):
        """Compute the regularized sparse-representation graph."""
        B, W, N = data.shape
        device = data.device

        norm_data = data - data.mean(dim=1, keepdim=True)
        norm_data = norm_data / (norm_data.norm(dim=1, keepdim=True) + 1e-8)

        sr_graph = torch.zeros(B, N, N, device=device)
        X = norm_data.transpose(1, 2)
        Gram = torch.bmm(X, X.transpose(1, 2))

        for i in range(N):
            targets = Gram[:, :, i].unsqueeze(-1)
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[i] = False
            D = X[:, mask, :]
            DtD = torch.bmm(D, D.transpose(1, 2))

            reg_matrix = self.alphas * torch.eye(N - 1, device=device).unsqueeze(0)
            coefficients = torch.linalg.solve(DtD + reg_matrix,
                                              torch.bmm(D, X[:, i:i + 1, :].transpose(1, 2)))

            sr_graph[:, i, mask] = coefficients.squeeze(-1)

        sr_graph = (sr_graph + sr_graph.transpose(1, 2)) / 2
        return sr_graph

    def dynamic_evidence_fusion(self, alpha_list):
        """Fuse Dirichlet evidence using conflict-aware subjective logic."""
        K = len(alpha_list)
        if K < 2:
            raise ValueError("Evidence fusion requires at least two opinions.")

        for i in range(K):
            if alpha_list[i].dim() == 1:
                alpha_list[i] = alpha_list[i].unsqueeze(0)

        batch_size, n_classes = alpha_list[0].shape
        S, E, b, u = {}, {}, {}, {}

        for k in range(K):
            S[k] = torch.sum(alpha_list[k], dim=-1, keepdim=True)
            E[k] = alpha_list[k] - 1
            b[k] = E[k] / S[k].expand(E[k].shape)
            u[k] = n_classes / S[k]

        b_fused = b[0]
        u_fused = u[0]

        for k in range(1, K):
            bb = torch.bmm(b_fused.view(batch_size, n_classes, 1),
                           b[k].view(batch_size, 1, n_classes))
            bu = b_fused * u[k].expand(b_fused.shape) + b[k] * u_fused.expand(b[k].shape)
            K_factor = torch.sum(bb, dim=(1, 2)) - torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)

            denominator = (1 - K_factor).clamp_min(torch.finfo(b_fused.dtype).eps).unsqueeze(1)
            b_fused = (b_fused * b[k] + bu) / denominator
            u_fused = (u_fused * u[k]) / denominator

        S_fused = n_classes / u_fused
        e_fused = b_fused * S_fused.expand(b_fused.shape)
        alpha_fused = e_fused + 1

        return alpha_fused

    def forward(self, x):
        """Return view-level, window-level, and subject-level Dirichlet parameters.

        Args:
            x: Resting-state fMRI time series with shape
                ``[batch_size, time_points, num_regions]``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D input tensor, got shape {tuple(x.shape)}.")
        batch_size, time_points, num_features = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} brain regions, got {num_features}."
            )
        if time_points < self.window_size:
            raise ValueError(
                f"Expected at least {self.window_size} time points, got {time_points}."
            )

        # 1. Temporal window construction.
        windowed_data = x.unfold(1, self.window_size, self.step_size).permute(0, 1, 3, 2)
        B, T, W, num_features = windowed_data.shape

        # 2. Multi-view functional connectivity construction.
        data_reshape = windowed_data.reshape(B * T, W, num_features)
        pc_graphs, hofc_graphs = self.compute_pc_graph(data_reshape, compute_hofc=True)
        mi_graphs = self.compute_mi_graph(data_reshape)
        sr_graphs = self.compute_sr_graph(data_reshape)

        view_graphs = {
            'pc': pc_graphs.view(B, T, num_features, num_features),
            'hofc': hofc_graphs.view(B, T, num_features, num_features),
            'mi': mi_graphs.view(B, T, num_features, num_features),
            'sr': sr_graphs.view(B, T, num_features, num_features)
        }

        # 3. Metric-specific GCN encoding.
        view_features = {view: [] for view in ['pc', 'hofc', 'mi', 'sr']}

        for t in range(T):
            for view in ['pc', 'hofc', 'mi', 'sr']:
                adj = view_graphs[view][:, t, :, :]
                node_features = view_graphs['pc'][:, t, :, :] if view != 'pc' else adj
                encoded = self.gcn_encoders[view](node_features, adj)
                view_features[view].append(encoded)

        for view in view_features:
            view_features[view] = torch.stack(view_features[view], dim=1)

        # 4. View-specific evidence estimation.
        view_outputs = {}
        view_evidence = {}
        view_alphas = {}

        for view in ['pc', 'hofc', 'mi', 'sr']:
            view_outputs[view] = self.mlp_classifiers[view](view_features[view])
            view_evidence[view] = self.softplus(view_outputs[view])
            view_alphas[view] = view_evidence[view] + 1

        # 5. Metric-view evidence fusion within each temporal window.
        fused_alphas_list = []
        for t in range(T):
            alpha_list_t = [view_alphas[view][:, t, :] for view in ['pc', 'hofc', 'mi', 'sr']]
            fused_alpha_t = self.dynamic_evidence_fusion(alpha_list_t)
            fused_alphas_list.append(fused_alpha_t)

        fused_alphas = torch.stack(fused_alphas_list, dim=1)

        # 6. Time-view evidence fusion for the subject-level prediction.
        alpha_list_final = [fused_alphas[:, t, :] for t in range(T)]
        final_fused_alpha = self.dynamic_evidence_fusion(alpha_list_final)

        return (view_alphas['pc'], view_alphas['hofc'], view_alphas['sr'], view_alphas['mi'],
                fused_alphas, final_fused_alpha)
