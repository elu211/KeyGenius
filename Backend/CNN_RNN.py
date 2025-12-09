import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, cnn_params={}, rnn_params={}):
        super().__init__()

        # --- CNN parameters ---
        input_channels = cnn_params.get("input_channels", 3)
        self.max_sequence = cnn_params.get("max_sequence", 350)
        self.num_hand_classes = cnn_params.get("num_hand_classes", 10)
        self.num_note_classes = cnn_params.get("num_note_classes", 128)
        self.note_weight = cnn_params.get("note_weight", 1.0)
        self.time_stamp_weight = cnn_params.get("time_stamp_weight", 0.01)
        self.hand_weight = cnn_params.get("hand_weight", 1.0)
        self.rnn_weight = cnn_params.get("rnn_weight", 1.0)

        # --- CNN layers ---
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # CNN classifiers
        self.hand_classifier = nn.Linear(256*7*7, self.max_sequence * self.num_hand_classes)
        self.note_classifier = nn.Linear(256*7*7, self.max_sequence * self.num_note_classes)
        self.time_regressor = nn.Linear(256*7*7, self.max_sequence)

        # --- RNN parameters ---
        rnn_input_size = 3  # For sequences: hand, note, time
        self.rnn_hidden_size = rnn_params.get("hidden_size", 64)
        rnn_output_size = rnn_params.get("output_size", self.num_note_classes)
        rnn_layers = rnn_params.get("num_layers", 3)

        self.input_linear = nn.Linear(rnn_input_size, self.rnn_hidden_size)
        self.rnn = nn.LSTM(
            input_size=self.rnn_hidden_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(self.rnn_hidden_size*2, rnn_output_size)

    def forward(self, x, y=None):
        """Forward pass for CNN only (image input)"""
        x = x.to(self.cnn[0].weight.device)
        if y is not None:
            y = y.to(self.cnn[0].weight.device)

        # --- CNN feature extraction ---
        features = self.cnn(x)  # [B, 256, 7, 7]
        flattened = features.view(features.size(0), -1)

        # CNN predictions
        hand_logits = self.hand_classifier(flattened).view(-1, self.max_sequence, self.num_hand_classes)
        note_logits = self.note_classifier(flattened).view(-1, self.max_sequence, self.num_note_classes)
        time_preds = self.time_regressor(flattened).view(-1, self.max_sequence)

        # Compute CNN loss if labels are provided
        loss = None
        if y is not None:
            loss_hand = F.cross_entropy(
                hand_logits.reshape(-1, self.num_hand_classes),
                y[:,0,:].reshape(-1).long(),
                ignore_index=999
            ) * self.hand_weight

            loss_note = F.cross_entropy(
                note_logits.reshape(-1, self.num_note_classes),
                y[:,1,:].reshape(-1).long(),
                ignore_index=999
            ) * self.note_weight

            # Handle time loss with valid mask
            valid_mask = (y[:,2,:] != 0).float()
            if valid_mask.sum() > 0:
                loss_time = (F.mse_loss(time_preds, y[:,2,:], reduction='none') * valid_mask).sum() / valid_mask.sum() * self.time_stamp_weight
            else:
                loss_time = torch.tensor(0.0).to(x.device)

            loss = loss_hand + loss_note + loss_time

        logits = {
            "cnn_hand": hand_logits,
            "cnn_note": note_logits,
            "cnn_time": time_preds
        }

        return loss, logits

    def forward_rnn(self, seq_inputs):
        """Forward pass for RNN only (sequence input)
        seq_inputs: [B, seq_len, 3] (hand, note, time)
        Returns: [B, seq_len, output_size]
        """
        seq_inputs = seq_inputs.float().to(next(self.parameters()).device)
        rnn_in = self.input_linear(seq_inputs)  # [B, seq_len, hidden]
        lstm_out, _ = self.rnn(rnn_in)
        rnn_logits = self.classifier(lstm_out)
        return rnn_logits