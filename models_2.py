import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from .attention import SelfAttention

class BiLSTMBaseline(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, v_dim, vocab_size, 
                       pretrained_vec, use_gpu, label_size,
                       dropout=0.5):
      
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = 1
        
        self.dropout = dropout
      
        # Embeddings (frozen to GLoVe, non-trainable)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_vec)
        self.embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim, v_dim)
        self.dropout_layer = F.dropout
        self.MLP = MLP(hidden_dim*2, label_size, dropout)
        #elf.hidden_layer = nn.Linear(hidden_dim*2, hidden_dim*2)
        #elf.decoder = nn.Linear(hidden_dim*2, label_size)
        #elf.softmax = torch.softmax
        #elf.tanh = torch.tanh
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, text_input):
        
        #Reset init hidden
        self.batch_size = text_input.size(1)
        self.hidden = self.init_hidden()
        # Fetch embedding from pretrained_vec
        
        x = self.embeddings(text_input)
        h , _ = self.lstm(x, self.hidden)
        
        z = h[-1]
        h, attention = self.attention(h, z.repeat(h.size()[0], 1, 1))
        H = h.sum(0)
        H = self.dropout_layer(H, p=self.dropout, training = self.training)
        #H = self.tanh(self.hidden_layer(H))
        #H = self.dropout_layer(H, p=self.dropout, training = self.training)
        #out = self.softmax(self.decoder(H), 1)
        out = self.MLP(H) 
        return out

class BiLSTMSourceNet(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, v_dim, vocab_size, 
                       pretrained_vec, use_gpu, label_size,
                       dropout=0.5, song_flag = True):
      
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = 1
        
        self.dropout = dropout
      
        # Embeddings (frozen to GLoVe, non-trainable)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_vec)
        self.embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim, v_dim)
        self.dropout_layer = F.dropout
        self.hidden_layer = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.decoder = nn.Linear(hidden_dim*2, label_size)
        self.softmax = torch.softmax
        self.song_flag = song_flag
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, text_input):
        
        #Reset init hidden
        self.batch_size = text_input.size(1)
        self.hidden = self.init_hidden()
        # Fetch embedding from pretrained_vec
        
        if not self.song_flag:
        
            with torch.no_grad():
        
                x = self.embeddings(text_input)
                h , _ = self.lstm(x, self.hidden)

                z = h[-1]
                h, attention = self.attention(h, z.repeat(h.size()[0], 1, 1))
                H = h.sum(0)
                H = self.dropout_layer(H, p = self.dropout, training = self.training)
                H = self.hidden_layer(H)
                H = self.dropout_layer(H, p = self.dropout, training = self.training)
                out = self.softmax(self.decoder(H), 1)
           
        else:
            
            x = self.embeddings(text_input)
            h , _ = self.lstm(x, self.hidden)

            z = h[-1]
            h, attention = self.attention(h, z.repeat(h.size()[0], 1, 1))
            H = h.sum(0)
            H = self.dropout_layer(H, p = self.dropout, training = self.training)
            H = self.hidden_layer(H)
            H = self.dropout_layer(H, p = self.dropout, training = self.training)
            out = self.softmax(self.decoder(H), 1)
          
        return attention, out
    
    
class BiLSTMTargetNet(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, v_dim, vocab_size, 
                       pretrained_vec, src_attention, use_gpu, label_size,
                       dropout=0.5):
      
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = 1
        
        self.dropout = dropout
      
        # Embeddings (frozen to GLoVe, non-trainable)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_vec)
        self.embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.src_attention = src_attention
        
        self.attention = DualAttention(hidden_dim, v_dim, src_attention)
        
        self.dropout_layer = F.dropout
        self.hidden_layer = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.decoder = nn.Linear(hidden_dim*2, label_size)
        self.softmax = torch.softmax
        
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, text_input):
        
        #Reset init hidden
        self.batch_size = text_input.size(1)
        self.hidden = self.init_hidden()
        # Fetch embedding from pretrained_vec
        
        x = self.embeddings(text_input)
        h , _ = self.lstm(x, self.hidden)
        
        z = h[-1]
        h, attention = self.attention(h, z.repeat(h.size()[0], 1, 1))
        H = h.sum(0)
        H = self.dropout_layer(H, p=self.dropout, training = self.training)
        H = self.hidden_layer(H)
        H = self.dropout_layer(H, p=self.dropout, training = self.training)
        out = self.softmax(self.decoder(H), 1)
          
        return out
    
class MLP(nn.Module):

    def __init__(self, hidden_dim, label_size,
                       dropout=0.5):
      
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.dropout_layer = F.dropout
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, label_size)
        self.softmax = torch.softmax
        self.tanh = torch.tanh

    def forward(self, input):
        
        H = self.tanh(self.hidden_layer(input))
        H = self.dropout_layer(H, p=self.dropout, training = self.training)
        out = self.softmax(self.decoder(H), 1)
          
        return out
    
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, v_dim):
        super().__init__()
        self.w_z = nn.Linear(hidden_dim*2, v_dim)
        self.w_w = nn.Linear(hidden_dim*2, v_dim)
        self.v = nn.Linear(v_dim, 1)
        self.tanh = torch.tanh
        self.softmax = torch.softmax
        self.hidden_dim = hidden_dim
    def forward(self, hidden, z):
        sum_w = self.w_w(hidden) + self.w_z(z)
        u = self.v(self.tanh(sum_w))
        attention = self.softmax(u, 0)
        h_att = torch.mul(attention.repeat(1,1,2*self.hidden_dim), hidden)
        return h_att, attention.squeeze()
        
  
class DualAttention(nn.Module):
    def __init__(self, hidden_dim, v_dim, alpha_s):
        super().__init__()
        self.w_z = nn.Linear(hidden_dim*2, v_dim)
        self.w_w = nn.Linear(hidden_dim*2, v_dim)
        self.v = nn.Linear(v_dim, 1)
        self.w_a = nn.Parameter(torch.tensor(0))
        self.tanh = torch.tanh
        self.alpha_s = alpha_s
        self.softmax = torch.softmax
    def forward(self, hidden, z):
        sum_w = self.w_w(hidden) + self.w_z(z) + w_a*self.alpha_s
        u = self.v(self.tanh(sum_v))
        attention = self.softmax(u, 0)
        h_att = torch.mul(attention.unsqueeze(2).repeat(1,1,2*hidden_dim), h)
        return attention
  