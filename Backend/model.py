import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LocalContextConv(nn.Module):
    def __init__(self, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        n = len(kernel_sizes)
        branch_dims = [d_model // n] * n
        branch_dims[-1] = d_model - (d_model // n) * (n - 1)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, bd, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k, bd in zip(kernel_sizes, branch_dims)
        ])
        self.proj = nn.Linear(sum(branch_dims), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)
        conv_outs = [conv(x_t) for conv in self.convs]
        x_t = torch.cat(conv_outs, dim=1)
        x_t = x_t.transpose(1, 2)
        x_t = self.proj(x_t)
        return self.norm(residual + self.dropout(x_t))


class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        return (forward_score - gold_score).mean()

    def _score_sentence(self, emissions, tags, mask):
        B, T, C = emissions.shape

        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit + trans) * mask[:, t]

        seq_lens = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_lens.clamp(min=0).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_alg(self, emissions, mask):
        B, T, C = emissions.shape
        mask_bool = mask.bool()

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]

        for t in range(1, T):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask_bool[:, t].unsqueeze(1), next_score, score)

        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions, mask):
        B, T, C = emissions.shape
        mask_bool = mask.bool()

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []

        for t in range(1, T):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask_bool[:, t].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions.unsqueeze(0)
        _, best_last = score.max(dim=1)

        seq_lens = mask.long().sum(dim=1)
        best_tags = torch.zeros(B, T, dtype=torch.long, device=emissions.device)

        for b in range(B):
            last = seq_lens[b].item() - 1
            best_tags[b, last] = best_last[b]
            for t in range(last - 1, -1, -1):
                best_tags[b, t] = history[t][b, best_tags[b, t + 1]]

        return best_tags


class FocalLoss(nn.Module):
    """Focal loss to focus on hard/rare examples."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
    
    def forward(self, logits, targets, mask):
        # logits: [B, T, C], targets: [B, T], mask: [B, T]
        B, T, C = logits.shape
        
        logits_flat = logits.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        valid = mask_flat == 1
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = logits_flat[valid]
        valid_targets = targets_flat[valid]
        
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(valid_logits.device)
            alpha_t = alpha.gather(0, valid_targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class FingeringTransformer(nn.Module):
    def __init__(
        self,
        input_dim=15,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.3,
        num_fingers=6,
        class_weights=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_fingers = num_fingers

        # Store class weights for focal loss
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.local_conv = LocalContextConv(d_model, kernel_sizes=[3, 5, 7], dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.emission_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_fingers),
        )

        self.crf = CRFLayer(num_fingers)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'crf' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_emissions(self, note_features, src_key_padding_mask=None):
        x = self.input_proj(note_features)
        x = self.pos_encoder(x)
        x = self.local_conv(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.emission_proj(x)

    def forward(self, note_features, fingers=None, mask=None, src_key_padding_mask=None, **kwargs):
        emissions = self.get_emissions(note_features, src_key_padding_mask)
        
        if fingers is not None and mask is not None:
            # Combined loss: CRF + Focal
            crf_loss = self.crf(emissions, fingers, mask)
            focal_loss = self.focal_loss(emissions, fingers, mask)
            
            # Weight focal loss higher to fight class imbalance
            total_loss = crf_loss + 0.5 * focal_loss
            return emissions, total_loss
        
        return emissions

    def generate(self, note_features, src_key_padding_mask=None, mask=None, **kwargs):
        emissions = self.get_emissions(note_features, src_key_padding_mask)
        if mask is None:
            mask = torch.ones(note_features.shape[:2], device=note_features.device)
        return self.crf.decode(emissions, mask)


def compute_accuracy(preds, targets, mask):
    valid = mask == 1
    if valid.sum() == 0:
        return 0.0
    correct = (preds == targets) & valid
    return (correct.sum().float() / valid.sum().float()).item()


def compute_per_finger_accuracy(preds, targets, mask):
    valid = mask == 1
    per_finger = {}
    for f in range(1, 6):
        finger_mask = (targets == f) & valid
        if finger_mask.sum() == 0:
            per_finger[f] = 0.0
        else:
            correct = (preds == f) & finger_mask
            per_finger[f] = (correct.sum().float() / finger_mask.sum().float()).item()
    return per_finger