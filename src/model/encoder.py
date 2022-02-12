import torch
from torch import nn

from error.checker import Checker


class Encoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, dropout, enc_hidden_state_dim, dec_hidden_state_dim):
        super().__init__()

        Checker.positive_int(404100, vocab_dim, 'Encoder vocabulary dimension')
        Checker.positive_int(404101, embedding_dim, 'Encoder embedding dimension')
        Checker.positive_float(404102, dropout, 'Encoder dropout probability')
        Checker.positive_int(404103, enc_hidden_state_dim, 'Encoder hidden state dimension')
        Checker.positive_int(404104, dec_hidden_state_dim, 'Decoder hidden state dimension')

        self.__embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.__dropout = nn.Dropout(dropout)
        self.__rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=enc_hidden_state_dim,
            bidirectional=True
        )
        self.__linear = nn.Linear(
            in_features=enc_hidden_state_dim * 2,
            out_features=dec_hidden_state_dim
        )
        self.__tan_h = nn.Tanh()

    def forward(self, phases_batch):
        """
        Args:
         - phases_batch: [max_words_len, batch_size]
        """

        # Step 1:
        #
        # phases_batch = [max_words_len, batch_size]
        embedded = self.__dropout(self.__embedding(phases_batch))
        # embedded = [max_words_len, batch_size, embedding_dim]

        # Step 2:
        #
        outputs, last_hidden_states = self.__rnn(embedded)
        # outputs = [max_words_len, batch_size, enc_hidden_state_dim * 2]
        #   - Contain both right and left concatenated encoder hidden states.
        #   - One for each input word.
        #
        # last_hidden_states = [2, batch_size, enc_hidden_state_dim]
        #   - One for each direction.
        #   - All layer hidden states stacked like this: forward_1, backward_1, forward_2, backward_2, ...

        # Step 3:
        #
        last_backward_hidden_state = last_hidden_states[-1, :, :]
        last_forward_hidden_state = last_hidden_states[-2, :, :]
        # last_backward_hidden_state = [batch_size, enc_hidden_state_dim]
        # last_forward_hidden_state = [batch_size, enc_hidden_state_dim]

        # Step 4:
        #
        enc_hidden_state = torch.cat((last_forward_hidden_state, last_backward_hidden_state), dim=1)
        # enc_hidden_state = [batch_size, enc_hidden_state_dim * 2]

        # Step 5:
        #
        dec_hidden_state = self.__tan_h(self.__linear(enc_hidden_state))
        # dec_hidden_state = [batch_size, dec_hidden_state]

        return outputs, dec_hidden_state
