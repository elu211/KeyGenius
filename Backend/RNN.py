import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

   def __init__(self, input_size, output_size, h_layer):
       super(RNN, self).__init__()

       if isinstance(input_size, tuple):
           flattened_size = input_size[0] * input_size[1]
           actual_input_size = input_size[1]
       else:
           flattened_size = input_size
           actual_input_size = input_size
       
       self.input_linear = nn.Linear(actual_input_size, h_layer)
       self.rnn = nn.LSTM(input_size=h_layer, hidden_size=h_layer, 
                         num_layers=7, bidirectional=True, batch_first=True)
       self.classifier = nn.Linear(h_layer * 2, output_size)
       self.output_size = output_size

   
   def forward(self, x, y = None):
       x = x.to(self.input_linear.weight.device)
       if y is not None:
           y = y.to(self.input_linear.weight.device)
           
       batch_size = x.size(0)
       seq_len = x.size(1)
       
       x_processed = self.input_linear(x)
       
       lstm_out, (h_n, c_n) = self.rnn(x_processed)
       
       logits = self.classifier(lstm_out)
       
       loss = None
       if y is not None:
           loss = F.cross_entropy(
               logits.reshape(-1, self.output_size),
               y.reshape(-1).long(),
               ignore_index=999
           )
       
       return loss, logits