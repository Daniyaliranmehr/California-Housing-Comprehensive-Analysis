import torch
from rich.console import Console

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """
    Define the trainer with a specific loss function and optimizer
    """
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.train_history = []
        self.valid_history = []

    def train(self, train_loader):
        self.model.train()
        loss_train = AverageMeter()

        for inputs, targets in train_loader:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs.squueze(), targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_train.update(loss.item())

        return loss_train.avg
    
    def validate(self, valid_loader):
        self.model.eval()
        loss_valid = AverageMeter()

        with torch.no_grad():
            for inputs, targets in valid_loader:
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squueze(), targets)

                loss_valid.update(loss.item())

        return loss_valid.avg
    
    def fit(self, train_loader, valid_loader, epochs=100, log_every=10):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.validate(valid_loader)

            self.train_history.append(train_loss)
            self.valid_history.append(valid_loss)

            console = Console()

            if epoch % log_every == 0:
                console.print(f"[bold green]Epoch {epoch}[/bold green]")
                console.print(f"Train Loss: {train_loss:.4f}")
                console.print(f"Valid Loss: {valid_loss:.4f}")
                console.print("")