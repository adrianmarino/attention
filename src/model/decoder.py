import torch
from pytorch_common.error import Assertions
from torch import nn

from model.attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, enc_hidden_state_dim, dec_hidden_state_dim, dropout, attention):
        super().__init__()
        Assertions.positive_int(404200, vocab_dim, 'Decoder vocabulary dimension')
        Assertions.positive_int(404201, embedding_dim, 'Decoder embedding dimension')
        Assertions.positive_int(404202, enc_hidden_state_dim, 'Encoder hidden state dimension')
        Assertions.positive_int(404203, dec_hidden_state_dim, 'Decoder hidden state dimension')
        Assertions.positive_float(404204, dropout, 'Decoder dropout probability')
        Assertions.is_class(404205, attention, 'Attention Module', Attention)

        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.rnn = nn.GRU(embedding_dim + enc_hidden_state_dim * 2, dec_hidden_state_dim)
        self.linear = nn.Linear(enc_hidden_state_dim * 2 + dec_hidden_state_dim + embedding_dim, vocab_dim)

        self.enc_hidden_state_dim = enc_hidden_state_dim
        self.dec_hidden_state_dim = dec_hidden_state_dim

    def forward(self, input, prev_hidden, encoder_outputs):
        """
        Args:
            - input           = [batch size] (A target work at once for each phrase)
            - prev_hidden     = [batch size, previous decoder hidden state dim]
            - encoder_outputs = [src phrase len, batch size, encoder hidden state dim * 2 (forward + backward)]
        Return:
            - prediction = [1, batch size, vocab_dim]
            - rnn_hidden = [batch size, decoder hidden state dim]
        """
        Assertions.has_shape(404206, input, (-1,), 'Decoder input')
        Assertions.has_shape(404207, prev_hidden, (-1, self.dec_hidden_state_dim), 'Decoder previous hidden state')
        Assertions.has_shape(404208, encoder_outputs, (-1, -1, self.enc_hidden_state_dim * 2),
                             'Decoder encoder outputs')

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded_input = self.dropout(self.embedding(input))
        # embedded_input = [1, batch size, embedding dim]

        a = self.attention(prev_hidden, encoder_outputs).unsqueeze(1)
        # a = [batch size, src phrase len]
        # a = [batch size, 1, src phrase len]

        # [1, src phrase len] x [src phrase len, encoder hidden state dim * 2] = [1, encoder hidden state dim * 2]
        # a x H => w
        weighted = torch.bmm(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        # weighted = [batch size, 1, encoder hidden state dim * 2]
        # weighted = [1, batch size, encoder hidden state dim * 2]

        rnn_input = torch.cat((weighted, embedded_input), dim=-1)  # -1 last dim
        # rnn_input = [1, batch size, embedding dim + encoder hidden state dim * 2]

        rnn_output, rnn_hidden = self.rnn(rnn_input)
        # rnn_output = [1, batch size, decoder hidden state dim]
        # rnn_hidden = [1, batch size, decoder hidden state dim]

        linear_input = torch.cat((rnn_output, embedded_input, weighted), dim=-1)
        # linear_input = [1, batch size, decoder hidden state dim + embedding dim + encoder hidden state dim * 2]

        prediction = self.linear(linear_input)
        # prediction = [1, batch size, vocab_dim]

        return prediction, rnn_hidden.squeeze(0)
