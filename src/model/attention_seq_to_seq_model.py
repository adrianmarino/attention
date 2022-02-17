import torch
from pytorch_common.error import Assertions
from pytorch_common.modules import CommonMixin
from torch import nn

from model.attention import Attention
from model.decoder import Decoder
from model.encoder import Encoder


class AttentionSeqToSeqModel(nn.Module, CommonMixin):
    def __init__(self,
                 source_vocab_dim,
                 target_vocab_dim,
                 enc_embedding_dim,
                 dec_embedding_dim,
                 enc_dropout,
                 dec_dropout,
                 enc_hidden_state_dim,
                 dec_hidden_state_dim
                 ):
        super().__init__()
        self.encoder = Encoder(
            source_vocab_dim,
            enc_embedding_dim,
            enc_dropout,
            enc_hidden_state_dim,
            dec_hidden_state_dim
        )
        self.decoder = Decoder(
            target_vocab_dim,
            dec_embedding_dim,
            enc_hidden_state_dim,
            dec_hidden_state_dim,
            dec_dropout,
            Attention(enc_hidden_state_dim, dec_hidden_state_dim)
        )
        self.target_vocab_dim = target_vocab_dim

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
        return self

    def forward(self, source_seq, target_seq, teacher_forcing_ratio=0.5):
        """
        Args:
            - src_seq = [src phrase len, batch size]
            - trg_seq = [target phrase len, batch size]
            - teacher_forcing_ratio(scalar): Percentage of times the model use ground true token
                                             instead of previous predicted token.

        Return: [target phrase len, batch size, dec vocab dim]
        """
        Assertions.is_tensor(404401, source_seq, 'Source sequence')
        Assertions.is_tensor(404402, target_seq, 'Target sequence')
        Assertions.positive_float(404403, teacher_forcing_ratio, 'Teacher forcing ratio')

        batch_size = source_seq.shape[1]
        target_seq_len = target_seq.shape[0]

        predictions = torch.zeros(target_seq_len, batch_size, self.target_vocab_dim).to(self.device)
        # predictions = [target phrase len, batch size, target vocab dim]

        outputs, dec_hidden_state = self.encoder(source_seq)
        # outputs = [max_words_len, batch_size, dec_hidden_state_dim]
        # dec_hidden_state = [batch_size, dec_hidden_state_dim

        target_token_batch = target_seq[0, :]
        # target_token_batch = [batch size]

        for token_index in range(target_seq_len):
            dec_prediction, dec_hidden_state = self.decoder(target_token_batch, dec_hidden_state, outputs)
            # dec_prediction = [1, batch size, vocab_dim]
            # dec_hidden_state = [batch size, decoder hidden state dim]

            predictions[token_index] = dec_prediction.squeeze(0)

            if torch.rand(1).item() < teacher_forcing_ratio:
                target_token_batch = target_seq[token_index, :]
            else:
                target_token_batch = torch.argmax(dec_prediction, dim=-1).squeeze(0)
                # [1, batch size] -> squeeze -> [batch size]

        return predictions
