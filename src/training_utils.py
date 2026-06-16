import torch
from rich.console import Console
import torch.nn as nn
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchmetrics import R2Score

import matplotlib.pyplot as plt

from rich.live import Live
from rich.panel import Panel


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


def Adaptive_loss(outputs, targets, c=1., alpha=1.):
    if alpha == 2:  # Quadratic
        loss = 0.5 * ((outputs - targets) / c) ** 2
        return torch.mean(loss)

    elif alpha == 0:  # Cauchy
        loss = torch.log((0.5 * ((outputs - targets) / c) ** 2) + 1)
        return torch.mean(loss)
    
    elif alpha == -torch.inf:  # Welsch
        loss = 1 - torch.exp(-0.5 * ((outputs - targets) / c) ** 2)
        return torch.mean(loss)
    
    else:  # Otherwise (Charbonnier & Geman-McClure)
         loss = (abs(alpha - 2) / alpha) * ((((outputs - targets) / c) ** 2 / abs(alpha - 2) + 1) ** (alpha / 2) - 1)
         return torch.mean(loss)


def log_cosh_loss(outputs, targets):
    error = outputs - targets
    return torch.mean(torch.log(torch.cosh(error + 1e-12)))


class Model(nn.Module):
    """
    Define the Model Architecture
    """
    def __init__(self, in_features, h1, h2, out_features):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_features)
        )

    def forward(self, x):
        return self.model(x)


class Trainer:
    """
    Trainer class for training and validating models.
    """
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.r2_metric = R2Score()

        self.console = Console()

        self.loss_train_history = []
        self.loss_valid_history = []

        self.r2_train_history = []
        self.r2_valid_history = []


    def train_epoch(self, train_loader):
        self.model.train()

        loss_meter = AverageMeter()
        self.r2_metric.reset()

        for inputs, targets in train_loader:
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())
            self.r2_metric.update(outputs, targets)

        r2 = self.r2_metric.compute().item()

        return loss_meter.avg, r2

    def validate_epoch(self, valid_loader):
        self.model.eval()

        loss_meter = AverageMeter()
        self.r2_metric.reset()

        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets)

                loss_meter.update(loss.item())
                self.r2_metric.update(outputs, targets)

        r2 = self.r2_metric.compute().item()

        return loss_meter.avg, r2

    def fit(self, train_loader, valid_loader, epochs=100, log_every=10, name=""):

        best_valid_r2 = float("-inf")
        best_valid_loss = float("inf")

        best_r2_epoch = 0
        best_loss_epoch = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(" • "),

            TextColumn("[bold yellow]Best R²:[/bold yellow] {task.fields[best_r2]:.4f}"),
            TextColumn(" Epoch [cyan]{task.fields[best_r2_epoch]}[/cyan]"),

            TextColumn(" | "),

            TextColumn("[bold green]Best Loss:[/bold green] {task.fields[best_loss]:.4f}"),
            TextColumn(" Epoch [cyan]{task.fields[best_loss_epoch]}[/cyan]"),
            console=self.console,
        ) as progress:

            task = progress.add_task(
            "[cyan]Training Model...",
            total=epochs,

            best_r2=0.0,
            best_r2_epoch=0,

            best_loss=float("inf"),
            best_loss_epoch=0,)

            for epoch in range(1, epochs + 1):

                train_loss, train_r2 = self.train_epoch(train_loader)
                valid_loss, valid_r2 = self.validate_epoch(valid_loader)

                self.loss_train_history.append(train_loss)
                self.loss_valid_history.append(valid_loss)

                self.r2_train_history.append(train_r2)
                self.r2_valid_history.append(valid_r2)

                # Best R²
                if valid_r2 > best_valid_r2:
                    best_valid_r2 = valid_r2
                    best_r2_epoch = epoch

                    torch.save(self.model.state_dict(), f"../saved_values/{name}_best_model.pth")

                # Best Loss
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_loss_epoch = epoch

                progress.update(
                    task,
                    advance=1,

                    best_r2=best_valid_r2,
                    best_r2_epoch=best_r2_epoch,

                    best_loss=best_valid_loss,
                    best_loss_epoch=best_loss_epoch)

                if epoch == 1 or epoch % log_every == 0 or epoch == epochs:

                    table = Table(
                        title=f"Epoch {epoch}/{epochs}",
                        show_header=True,
                        show_lines=False,
                    )

                    table.add_column("Metric", justify="left")
                    table.add_column("Train", justify="right")
                    table.add_column("Validation", justify="right")

                    table.add_row(
                        "Loss",
                        f"{train_loss:.4f}",
                        f"{valid_loss:.4f}",
                    )

                    table.add_row(
                        "R²",
                        f"{train_r2:.4f}",
                        f"{valid_r2:.4f}",
                    )

                    self.console.print(table)

        return {
            "train_loss": self.loss_train_history,
            "valid_loss": self.loss_valid_history,

            "train_r2": self.r2_train_history,
            "valid_r2": self.r2_valid_history,

            "best_valid_r2": best_valid_r2,
            "best_r2_epoch": best_r2_epoch,

            "best_valid_loss": best_valid_loss,
            "best_loss_epoch": best_loss_epoch}


def plot_training_history(history, name, figsize=(10, 5)):
    """
    Plot training and validation Loss and R² curves.

    Args:
        history (dict): dictionary returned by Trainer.fit()
            {
                "train_loss": [...],
                "valid_loss": [...],
                "valid_r2": [...]
            }
        figsize (tuple): figure size
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=figsize)

    
    ax1.plot(epochs, history["train_loss"], '-', color='tab:blue', linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["valid_loss"], '-', color='tab:orange', linewidth=2, label="Validation Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_title(f"Training History {name}")

    
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_r2"], '-', color='tab:green', linewidth=2, label="Train R²")
    ax2.plot(epochs, history["valid_r2"], '-', color='tab:red', linewidth=2, label="Validation R²")

    ax2.set_ylabel("R²")
    ax2.tick_params(axis='y', labelcolor='black')

    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.10, 1), borderaxespad=0, frameon=True)

    plt.tight_layout()

    plt.savefig(f"../assets/{name}_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
