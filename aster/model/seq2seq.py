import torch,random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.ResNet import ResNet
from model.STN import STN
import ipdb

PAD = 1
SOS = 0
EOS = 2
MAX_LENGTH = 40

class BiRNN(nn.Module):
    def __init__(self,input_size,hidden_size=256,num_layers=2):
        super(BiRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,hidden_size)

    def forward(self,x):    # x.size [batch,t,feature]
        N,seq_len,_ = x.size()
        # Set initial states
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).cuda()

        # Forward propagate LSTM
        out,_ = self.lstm(x,(h0,c0))        # [batch,seq_len,hidden_size*num_directions]
        out = out.contiguous()
        out = out.view(-1,out.size(2))      # reshape to [batch*seq_length,hidden_size*2]
        out = self.fc(out).view(N,seq_len,-1)
        return out # [b,t,c]



class Encoder(nn.Module):
    def __init__(self,lstm_input_size=512,hidden_size=256,num_layers=1,bidirectional=True,use_stn=False):
        super(Encoder,self).__init__()
        if use_stn:
            print('Create model with STN')
            # self.features = nn.Sequential(STN(output_img_size=[32,100],num_control_points=20,margins=[0.1,0.1]),
            #                               ResNet())
            self.stn = STN(output_img_size=[32,100], num_control_points=20, margins=[0.1, 0.1])
        else:
            print('Create model without STN')
        self.use_stn = use_stn
        self.features = ResNet()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self,x):
        recitified = None
        if self.use_stn:
            x = self.stn(x)
            recitified = x
        # print(x.size())
        x = self.features(x)    # [B,C,H,W]
        # print(x.size())
        x = x.view(x.size(0),x.size(1),-1)
        # print(x.size())
        x = x.permute(0,2,1)    #[batch,t,channels]
        # print(x.size())
        x, state = self.lstm(x)
        # print("x:", x.size())
        # print(len(state))
        # print("oooooooooo")
        return x, recitified

class BahdanauAttentionMechanism(nn.Module):

    def __init__(self, query_dim, values_dim, attention_dim):
        super(BahdanauAttentionMechanism, self).__init__()
        self.fc_query = nn.Linear(query_dim, attention_dim)
        self.fc_values = nn.Linear(values_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, values):
        """
        Args:
            query (s_i): [batch_size, query_dim]
            values: [batch_size, T, values_dim]
        Returns:
            context: (c_i), [batch_size, values_dim]
            attention: (a_i), [batch_size, T]
        """
        # print("values:", values.size())
        keys = self.fc_values(values)
        # print("keys:", keys.size())
        query = self.fc_query(query)
        # print("query:", query.size())
        query = query.unsqueeze(1).expand_as(keys)
        # print("query:", query.size())
        e_i = self.v(self.tanh(query + keys)).squeeze(-1)
        # print("e_i:", e_i.size())
        a_i = self.softmax(e_i)
        # print("a_i:", a_i.size())
        c_i = torch.bmm(a_i.unsqueeze(1), values)
        # print("c_i:", c_i.size())
        return c_i.squeeze(1), a_i


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_dim, attention_dim, y_dim, encoder_output_dim):
        super(AttentionDecoder, self).__init__()
        self.lstm_cell = nn.LSTMCell(y_dim + encoder_output_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, y_dim)
        self.attention_mechanism = BahdanauAttentionMechanism(
                query_dim=hidden_dim, values_dim=encoder_output_dim, attention_dim=attention_dim)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.y_dim = y_dim
        self.encoder_output_dim = encoder_output_dim
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim).cuda(),
                torch.zeros(batch_size, self.hidden_dim).cuda())

    def one_hot_embedding(self, input_tokens):
        batch_size = input_tokens.size(0)
        one_hot = torch.cuda.FloatTensor(batch_size, self.y_dim).zero_()
        return one_hot.scatter_(-1, input_tokens.view(batch_size, 1), 1)

    def forward(self, pre_y_token, pre_state_hc, encoder_output):
        # print("pre_state_hc[0]:", pre_state_hc[0].size())
        # print("encoder_output:", encoder_output.size())
        c, a = self.attention_mechanism(pre_state_hc[0], encoder_output)
        # print("pre_y_token:", pre_y_token.size())
        pre_y = self.one_hot_embedding(pre_y_token)
        # print("pre_y:", pre_y.size())
        new_state_h, new_state_c = self.lstm_cell(
                torch.cat((pre_y, c), -1), pre_state_hc)
        out = self.fc_out(new_state_h)
        # print(out.size())
        return out, (new_state_h, new_state_c), a
