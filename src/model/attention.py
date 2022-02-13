import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
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
        src_phrase_len = encoder_outputs.size()[0]
        batch_size = encoder_outputs.size()[1]

        # Repeat prev_dec_hidden by src word
        hidden = torch.cat([prev_dec_hidden] * src_phrase_len, dim=0)
        hidden = hidden.view(src_phrase_len, batch_size, -1)
        # hidden = [batch size, src phrase len, previous decoder hidden state dim]

        enc_outputs = encoder_outputs.permute(1, 0, 2)
        # enc_outputs = [batch size, src phrase len, encoder hidden state dim * 2]

        return torch.cat((enc_outputs, hidden), dim=-1)  # -1 == last dim.

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
        energy = self.__energy(prev_dec_hidden, encoder_outputs)
        return self._attention(energy)
