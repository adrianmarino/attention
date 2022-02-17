import torch
import torch.nn.functional as F
from pytorch_common.error import Assertions
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        Assertions.positive_int(404103, encoder_hidden_dim, 'Encoder hidden state dimension')
        Assertions.positive_int(404104, decoder_hidden_dim, 'Decoder hidden state dimension')

        self.__energy_dense = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.__energy_activation = nn.Tanh()
        self.__v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def __concat_prev_dec_hid_with_each_each_output(self, prev_dec_hidden, encoder_outputs):
        """
        Concatena el estado oculto anterior del decoder con cada
        salida del encoder...
        Args:
            - prev_dec_hidden = [batch size, decoder hidden state dim]
            - encoder_outputs = [src_phrase_len, batch size, encoder hidden state dim * 2 (forward + backward)]
        Return:
             [batch size, src phrase len, decoder hidden state dim + encoder hidden state dim * 2]
        """
        src_phrase_len, batch_size = encoder_outputs.size()[:2]

        # Repeat prev_dec_hidden by src word
        hidden = torch.cat([prev_dec_hidden] * src_phrase_len, dim=0)
        hidden = hidden.view(src_phrase_len, batch_size, -1)
        # hidden = [src phrase len, batch size, previous decoder hidden state dim]

        return torch.cat((hidden, encoder_outputs), dim=-1).permute(1, 0, 2)  # -1 == last dim.

    def __energy(self, prev_dec_hidden, encoder_outputs):
        """
        Calculate energy.
        Args:
            - prev_dec_hidden = [batch size, decoder hidden state dim]
            - encoder_outputs = [src_phrase_len, batch size, encoder hidden state dim * 2 (forward + backward)]
        Return:
            [batch size, src phrase len, decoder hidden state dim]
        """
        prev_dec_hidden_enc_outs = self.__concat_prev_dec_hid_with_each_each_output(prev_dec_hidden, encoder_outputs)
        # prev_dec_hidden_enc_outs =
        #   [batch size, src phrase len, previous decoder hidden state dim + encoder hidden state dim * 2]

        return self.__energy_activation(self.__energy_dense(prev_dec_hidden_enc_outs))

    def __attention(self, energy):
        """
        Calculate attention vector.
        Args:
            -  energy = [batch size, src phrase len, decoder hidden state dim]
        Return: [batch size, src phrase len]
        """
        attention = self.__v(energy).squeeze(-1)  # -1 == last dim
        # __v = [batch size, src phrase len, 1]
        # attention = [batch size, src phrase len]

        return F.softmax(attention, dim=1)

    def forward(self, prev_dec_hidden, encoder_outputs):
        """
        Args:
            - prev_dec_hidden = [batch size, previous decoder hidden state dim]
            - encoder_outputs = [src_phrase_len, batch size, encoder hidden state dim * 2 (forward + backward)]
        Return: [batch size, src phrase len]
        """
        Assertions.is_tensor(404301, prev_dec_hidden, 'Previous decoder hidden state')
        Assertions.is_tensor(404302, encoder_outputs, 'Encoder outputs')

        energy = self.__energy(prev_dec_hidden, encoder_outputs)
        return self.__attention(energy)
