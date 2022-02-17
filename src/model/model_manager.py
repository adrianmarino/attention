import torch
from pytorch_common.callbacks import CallbackManager
from pytorch_common.callbacks.output import Logger
from torch.nn.utils import clip_grad_norm_


class ModelManager:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(
            self,
            train_iterator,
            valid_iterator,
            epochs,
            teacher_forcing_ratio=0.5,
            clip=1,
            verbose=1,
            callbacks=[Logger()]
    ):
        callback_manager = CallbackManager(epochs, self.optimizer, self.loss_fn, self.model, callbacks, verbose)

        for epoch in range(epochs):
            callback_manager.on_epoch_start(epoch)

            train_loss = self.train(train_iterator, teacher_forcing_ratio, clip)
            val_loss = self.validation(valid_iterator)

            callback_manager.on_epoch_end(train_loss, val_loss)

            if callback_manager.break_training():
                break

        return callback_manager.ctx

    def train(self, iterator, teacher_forcing_ratio=0.5, clip=1):
        self.model.train()
        loss = 0
        for batch in iterator:
            self.optimizer.zero_grad()

            batch_loss = self.loss_fn(self.predict(batch, teacher_forcing_ratio), batch.target)

            batch_loss.backward()
            clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            loss += batch_loss.item()

        return loss / len(iterator)

    def predict(self, batch, teacher_forcing_ratio=0.0):
        # source_seq = [src phrase len, batch size]
        # target_seq = [target phrase len, batch size]
        predictions = self.model(batch.source, batch.target, teacher_forcing_ratio)
        # predictions = [target phrase len, batch size, dec vocab dim]

        return predictions

    def validation(self, iterator):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for batch in iterator:
                loss += self.loss_fn(self.predict(batch), batch.target).item()

        return loss / len(iterator)
