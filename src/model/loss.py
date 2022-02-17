from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_fn, target_vocab_dim):
        super().__init__()
        self.loss_fn = loss_fn
        self.target_vocab_dim = target_vocab_dim

    def forward(self, predictions, targets):
        # Remove start token...
        targets = targets[1:]
        predictions = predictions[1:]

        targets = targets.view(-1)
        predictions = predictions.view(-1, self.target_vocab_dim)
        # target_seq = [(target phrase len - 1) * batch size]
        # predictions = [(target phrase len - 1) * batch size, target vocab dim]

        return self.loss_fn(predictions, targets)
