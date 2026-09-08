"""Figure construction for Q1(c). Shared by the offline script and the wandb run."""

import matplotlib
matplotlib.use("Agg")  # headless: safe on Colab/Kaggle and inside a training run
import matplotlib.pyplot as plt


def curves_figure(history: dict):
    """Side-by-side loss and top-1 accuracy curves from a train.py history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["test_loss"], label="test")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["test_acc"], label="test")
    ax_acc.set(xlabel="epoch", ylabel="top-1 accuracy (%)", title="Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def save_curves(history: dict, out_path: str) -> str:
    fig = curves_figure(history)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
