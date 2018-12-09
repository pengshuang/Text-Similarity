import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .attention import Attention


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1,dropout_p=0.3,use_cuda=False):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size, n_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size*2, input_size)
        self.attention = Attention(hidden_size)
        self.use_cuda = use_cuda
        
    def init_hidden(self,inputs):
        hidden = Variable(torch.zeros(self.n_layers,inputs.size(0),self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden
    
    
    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_normal(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_normal(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
    
    
    def greedy_decode(self,inputs,context,max_length,encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        
        decode=[]
        i=0
        # Apply GRU to the output so far
        decoded = inputs
        while decoded.data.tolist()[0]!=3:
            _, hidden = self.gru(torch.cat((embedded,context),2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden,context.transpose(0,1)),2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            
            # compute next context vector using attention
            context = self.attention(hidden, encoder_outputs,None)
            i+=1
        #  column-wise concat, reshape!!
        scores = torch.cat(decode)
        return scores.max(1)[1]
    
    
    def beam_search_decode(self,inputs,context,max_length,encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        _, hidden = self.gru(torch.cat((embedded,context),2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
        concated = torch.cat((hidden,context.transpose(0,1)),2) # y_t = g(h_t,y_{t-1},c)
        score = self.linear(concated.squeeze(0))

        beam = Beam([score,hidden],3) # beam init
        nodes = beam.get_next_nodes()
        i = 1

        while i<max_length:
            siblings=[]
            for node in nodes:
                embedded = self.embedding(node[0])# y_{t-1}
                hidden = node[1]
                context = self.attention(hidden, encoder_outputs,None)
                _, hidden = self.gru(torch.cat((embedded,context),2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
                concated = torch.cat((hidden,context.transpose(0,1)),2) # y_t = g(h_t,y_{t-1},c)
                score = self.linear(concated.squeeze(0))

                siblings.append([score,hidden])
            i+=1
            nodes = beam.select_k(siblings)
            # compute next context vector using attention

        return beam.get_best_seq()
    
    
    def forward(self,inputs,context,max_length,encoder_outputs,encoder_lengths=None):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        context : B,1,D (FloatTensor, Last encoder hidden state)
        encoder_outputs : B,T,D
        encoder_lengths : B,T # list
        max_length : int, max length to decode
        """
        # Get the embedding of the current input word
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        embedded = self.dropout(embedded)
        
        decode=[]
        # Apply GRU to the output so far
        for i in range(max_length): # because of <s> , </s>

            _, hidden = self.gru(torch.cat((embedded,context),2), hidden) # h_t = f(h_{t-1},y_{t-1},c)
            concated = torch.cat((hidden,context.transpose(0,1)),2) # y_t = g(h_t,y_{t-1},c)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score,1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1) # y_{t-1}
            embedded = self.dropout(embedded)
            
            # compute next context vector using attention
            context = self.attention(hidden.transpose(0,1), encoder_outputs, encoder_lengths)
            
        #  column-wise concat, reshape!!
        scores = torch.cat(decode,1)
        return scores.view(inputs.size(0)*max_length,-1)


class Beam:
    def __init__(self,root,num_beam):
        """
        root : (score, hidden)
        """
        self.num_beam = num_beam
        
        score = F.log_softmax(root[0],1)
        s,i = score.topk(num_beam)
        s = s.data.tolist()[0]
        i = i.data.tolist()[0]
        i = [[ii] for ii in i]
        hiddens = [root[1] for _ in range(num_beam)]
        self.beams = list(zip(s,i,hiddens))
        self.beams = sorted(self.beams,key= lambda x:x[0], reverse=True)
        
    def select_k(self,siblings):
        """
        siblings : [score,hidden]
        """
        candits=[]
        for p_index,sibling in enumerate(siblings):
            parents = self.beams[p_index] # (cummulated score, list of sequence)
            score = F.log_softmax(sibling[0],1)
            s,i = score.topk(self.num_beam)
            scores = s.data.tolist()[0]
            #scores = [scores[i-1]+i for i in range(1,self.num_beam+1)] # penalize siblings
            indices = i.data.tolist()[0]

            candits.extend([(parents[0]+scores[i],parents[1]+[indices[i]],sibling[1]) for i in range(len(scores))])
        
        candits = sorted(candits,key= lambda x:x[0], reverse=True)

        self.beams = candits[:self.num_beam]
        
        # last_input, hidden
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1,-1),b[2]] for b in self.beams]
        
    def get_best_seq(self):
        return self.beams[0][1]
    
    def get_next_nodes(self):
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1,-1),b[2]] for b in self.beams]
    