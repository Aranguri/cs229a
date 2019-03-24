# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_templates = 10
        self.templates_init = 'orthogonal'

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        elif self.model == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.i = 0

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))

        # we want to modify hidden
        # the temple for paltas' husbands: tempaltes
        # hidden   n_layers x batch_size x hidden_size
        # templa   hidden_size x n_templates
        # print(hidden.size())
        templates = torch.empty(self.hidden_size, self.n_templates)
        # print(templates.size())

        if self.templates_init == 'orthogonal':
            torch.nn.init.orthogonal_(templates)
        elif self.templates_init == 'normal':
            torch.nn.init.normal_(templates)
        elif self.templates_init == 'xavier':
            torch.nn.init.xavier_normal_(templates)

        # templates = templates.cuda()

        def templify(vector, template):
            vector_probs = torch.einsum('ijk,kl->ijl', (vector, template))
            softmax = torch.nn.Softmax(dim=2)
            vector_probs = softmax(vector_probs)
            vector = torch.einsum('ijl,kl->ijk', (vector_probs, template))
            return vector

        hidden = templify(hidden, templates)
        self.templates = templates

        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
