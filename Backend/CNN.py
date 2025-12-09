import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, input_channels=3, max_sequence=350, num_hand_classes=10, 
                num_note_classes=128, note_weight=1.0, time_stamp_weight=0.01, 
                hand_weight=1.0):
       super().__init__()

       self.cnn = nn.Sequential(
           nn.Conv2d(input_channels, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
           nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
           nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((7, 7))
       )

       self.flatten = nn.Flatten()

       self.hand_classifier = nn.Linear(256 * 7 * 7, max_sequence * num_hand_classes)
       self.note_classifier = nn.Linear(256 * 7 * 7, max_sequence * num_note_classes)
       self.time_regressor = nn.Linear(256 * 7 * 7, max_sequence)  
       
       self.max_sequence = max_sequence
       self.num_hand_classes = num_hand_classes
       self.num_note_classes = num_note_classes

       self.note_weight = note_weight
       self.time_stamp_weight = time_stamp_weight
       self.hand_weight = hand_weight

   def forward(self, x, y=None):
       x = x.to(self.cnn[0].weight.device)
       if y is not None:
           y = y.to(self.cnn[0].weight.device)
           
       features = self.cnn(x)
       #features = self.flatten(features)
       
       hand_logits = self.hand_classifier(features).view(-1, self.max_sequence, self.num_hand_classes)
       note_logits = self.note_classifier(features).view(-1, self.max_sequence, self.num_note_classes)
       time_preds = self.time_regressor(features).view(-1, self.max_sequence)
       
       logits = (hand_logits, note_logits, time_preds)
       
       loss = None
       if y is not None:
           
           y_hand = y[:, 0, :].long() 
           y_note = y[:, 1, :].long() 
           y_time = y[:, 2, :].float() 
           
           loss_hand = nn.CrossEntropyLoss(ignore_index=999)(
               hand_logits.reshape(-1, self.num_hand_classes), 
               y_hand.reshape(-1)
           ) * self.hand_weight
           
           loss_note = nn.CrossEntropyLoss(ignore_index=999)(
               note_logits.reshape(-1, self.num_note_classes), 
               y_note.reshape(-1)
           ) * self.note_weight

           valid_mask = (y_time != 0).float() 
           loss_time = (nn.MSELoss(reduction='none')(time_preds, y_time) * valid_mask).sum() / valid_mask.sum() * self.time_stamp_weight

           loss = loss_hand + loss_note + loss_time

       return loss, logits