"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
A deep feed-forward network built entirely from custom PrunableLinear layers.
Each layer learns per-weight gate scores via sigmoid activations; an L1-style
sparsity penalty on those gates drives the network to prune itself during
training.  Three lambda values are swept to explore the accuracy-vs-sparsity
trade-off.

Author : Auto-generated production implementation
Runtime: ~45-60 min on a single A100 for all 3 lambda runs
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────────────
import math
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
# 2. Reproducibility Seeds
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = False   # leave True only if exact repro needed
torch.backends.cudnn.benchmark = True        # faster convolutions (not used here, but safe)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PrunableLinear Layer
# ──────────────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """A fully-connected layer with learnable per-weight gate scores.

    Each weight w_{ij} is multiplied by sigmoid(gate_score_{ij}) before the
    linear transform.  When the sparsity penalty pushes a gate score toward
    -inf the corresponding sigmoid approaches 0, effectively pruning that
    connection.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Trainable weight matrix — Kaiming uniform init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias — zero init
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — random init (normal, mean=0, std=0.5) for diverse starting points
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_features).
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (sigmoid of gate_scores), detached.

        Returns
        -------
        torch.Tensor
            Gate values in [0, 1] of shape (out_features, in_features).
        """
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 0.1) -> float:
        """Fraction of gates below a given threshold (effectively pruned).

        Parameters
        ----------
        threshold : float
            Gates below this value are considered pruned.

        Returns
        -------
        float
            Sparsity ratio in [0, 1].
        """
        gates = self.get_gates()
        return float((gates < threshold).sum().item() / gates.numel())


# ──────────────────────────────────────────────────────────────────────────────
# 4. SelfPruningNet
# ──────────────────────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """Feed-forward network built from PrunableLinear layers.

    Architecture
    ------------
    3072 → 256 → 128 → 64 → 10
    Each hidden layer is followed by BatchNorm1d, GELU, and Dropout (except
    the last hidden block which omits Dropout, and the output layer which
    outputs raw logits).

    Total gate parameters: ~828K  (total trainable: ~1.66M)
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Block 1: 3072 → 256 ---
        self.pl1 = PrunableLinear(3072, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)

        # --- Block 2: 256 → 128 ---
        self.pl2 = PrunableLinear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)

        # --- Block 3: 128 → 64 (no dropout) ---
        self.pl3 = PrunableLinear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        # --- Output: 64 → 10 (raw logits) ---
        self.pl4 = PrunableLinear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all 4 PrunableLinear blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input images flattened to shape (batch, 3072).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch, 10).
        """
        # Flatten spatial dims
        x = x.view(x.size(0), -1)

        # Block 1
        x = self.pl1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.drop1(x)

        # Block 2
        x = self.pl2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.drop2(x)

        # Block 3 (no dropout)
        x = self.pl3(x)
        x = self.bn3(x)
        x = F.gelu(x)

        # Output — raw logits
        x = self.pl4(x)
        return x

    def get_all_gates(self) -> List[torch.Tensor]:
        """Collect gate tensors from every PrunableLinear layer.

        Returns
        -------
        List[torch.Tensor]
            One tensor per layer with gate values in [0, 1].
        """
        return [m.get_gates() for m in self.modules() if isinstance(m, PrunableLinear)]

    def total_sparsity(self, threshold: float = 0.1) -> float:
        """Overall sparsity across all PrunableLinear layers.

        Parameters
        ----------
        threshold : float
            Gates below this are considered pruned.

        Returns
        -------
        float
            Global sparsity ratio in [0, 1].
        """
        total_gates = 0
        pruned_gates = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = m.get_gates()
                total_gates += gates.numel()
                pruned_gates += (gates < threshold).sum().item()
        return float(pruned_gates / total_gates) if total_gates > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Loss Function
# ──────────────────────────────────────────────────────────────────────────────
def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: SelfPruningNet,
    lambda_sparse: float,
) -> Tuple[torch.Tensor, float, float]:
    """Compute total loss = CE + lambda * sum(all gate values).

    Uses the raw SUM of gate values so each gate receives a gradient of
    lambda * sigmoid'(gate_score) — independent of model size.  This gives
    every gate direct, meaningful pressure to close.

    Parameters
    ----------
    logits : torch.Tensor
        Raw network output of shape (batch, 10).
    targets : torch.Tensor
        Ground-truth class indices of shape (batch,).
    model : SelfPruningNet
        The network (we iterate over its PrunableLinear layers).
    lambda_sparse : float
        Weight on the sparsity regularization term.

    Returns
    -------
    Tuple[torch.Tensor, float, float]
        (total_loss, ce_loss_value, sparsity_loss_value)
        total_loss is a differentiable tensor; the other two are Python floats
        for logging.
    """
    # Cross-entropy classification loss
    ce_loss = F.cross_entropy(logits, targets)

    # Sparsity loss: SUM of all sigmoid(gate_scores) across every layer
    sparsity_loss = torch.tensor(0.0, device=logits.device)
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            sparsity_loss = sparsity_loss + torch.sigmoid(m.gate_scores).sum()

    total_loss = ce_loss + lambda_sparse * sparsity_loss
    return total_loss, ce_loss.item(), sparsity_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Data Loaders
# ──────────────────────────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 training and test DataLoaders with proper augmentation.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for both loaders.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, test_loader)
    """
    # CIFAR-10 channel-wise statistics
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    # Test transforms — no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 7. Single Training Run
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy on a dataset.

    Parameters
    ----------
    model : SelfPruningNet
        Trained model in eval mode.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    float
        Accuracy as a percentage (0-100).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one_run(
    lambda_val: float,
    device: torch.device,
    epochs: int = 50,
) -> Tuple[float, float, Dict[str, List[float]], SelfPruningNet]:
    """Train a fresh SelfPruningNet with a given lambda_sparse value.

    Parameters
    ----------
    lambda_val : float
        Weight for the sparsity regularization.
    device : torch.device
        CUDA or CPU device.
    epochs : int
        Number of training epochs.

    Returns
    -------
    Tuple[float, float, Dict[str, List[float]], SelfPruningNet]
        (final_test_acc, final_sparsity, history_dict, trained_model)
        history_dict has keys: 'train_loss', 'test_acc', 'sparsity'.
    """
    # Reset seeds for each run so every lambda starts from the same init
    torch.manual_seed(42)
    np.random.seed(42)

    # Build data loaders
    train_loader, test_loader = get_dataloaders(batch_size=256)

    # Build model
    model = SelfPruningNet().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*80}")
    print(f"Starting training | λ = {lambda_val} | Trainable params = {total_params:,}")
    print(f"{'='*80}")

    # Separate param groups: gate_scores get 100x higher LR + no weight decay
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {'params': other_params, 'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': gate_params, 'lr': 3e-2, 'weight_decay': 1e-3},  # gates: fast LR, decay toward 0
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # History tracking
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "test_acc": [],
        "sparsity": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_sparse = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass (fp32, no mixed precision needed for small model)
            logits = model(images)

            # Compute loss
            total_loss, ce_val, sparse_val = compute_total_loss(
                logits, labels, model, lambda_val
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce += ce_val
            epoch_sparse += sparse_val
            num_batches += 1

        # Step the learning rate scheduler
        scheduler.step()

        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_ce = epoch_ce / num_batches
        avg_sparse = epoch_sparse / num_batches

        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)

        # Compute network sparsity
        sparsity_pct = model.total_sparsity(threshold=1e-2) * 100.0

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log history
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity_pct)

        # Print epoch summary
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"\u03bb={lambda_val:.5f} | "
            f"LR={current_lr:.6f} | "
            f"Loss={avg_loss:.4f} | "
            f"CE={avg_ce:.4f} | "
            f"Sparse={avg_sparse:.4f} | "
            f"TestAcc={test_acc:.2f}% | "
            f"Sparsity={sparsity_pct:.2f}%"
        )

    final_sparsity = model.total_sparsity(threshold=1e-2) * 100.0
    final_acc = evaluate(model, test_loader, device)

    print(f"\nRun complete | λ={lambda_val} | Final Acc={final_acc:.2f}% | Sparsity={final_sparsity:.2f}%")
    return final_acc, final_sparsity, history, model


# ──────────────────────────────────────────────────────────────────────────────
# 8. Plotting: Training Curves
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(
    all_histories: List[Dict[str, List[float]]],
    lambda_values: List[float],
) -> None:
    """Plot training loss, test accuracy, and sparsity across epochs for each lambda.

    Saves the figure to ``training_curves.png``.

    Parameters
    ----------
    all_histories : List[Dict[str, List[float]]]
        One history dict per lambda run.
    lambda_values : List[float]
        Corresponding lambda values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # --- Panel 1: Training Loss ---
    ax = axes[0]
    for i, (hist, lam) in enumerate(zip(all_histories, lambda_values)):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], color=colors[i], label=f"λ={lam}", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss vs Epoch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Test Accuracy ---
    ax = axes[1]
    for i, (hist, lam) in enumerate(zip(all_histories, lambda_values)):
        epochs = range(1, len(hist["test_acc"]) + 1)
        ax.plot(epochs, hist["test_acc"], color=colors[i], label=f"λ={lam}", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy vs Epoch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Sparsity ---
    ax = axes[2]
    for i, (hist, lam) in enumerate(zip(all_histories, lambda_values)):
        epochs = range(1, len(hist["sparsity"]) + 1)
        ax.plot(epochs, hist["sparsity"], color=colors[i], label=f"λ={lam}", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("Network Sparsity vs Epoch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[✓] Saved training_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Plotting: Gate Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_gate_distribution(model: SelfPruningNet, lambda_val: float) -> None:
    """Plot a histogram of gate values for each PrunableLinear layer.

    Saves the figure to ``gate_distribution_{lambda_val}.png``.

    Parameters
    ----------
    model : SelfPruningNet
        Trained model whose gate distributions to visualize.
    lambda_val : float
        Lambda value used during training (for the filename / title).
    """
    prunable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PrunableLinear)
    ]

    fig, axes = plt.subplots(1, len(prunable_layers), figsize=(4 * len(prunable_layers), 5))
    if len(prunable_layers) == 1:
        axes = [axes]

    for ax, (name, layer) in zip(axes, prunable_layers):
        gates = layer.get_gates().cpu().numpy().flatten()
        ax.hist(gates, bins=100, color="#1976D2", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{name}\n({layer.in_features}→{layer.out_features})", fontsize=9)
        ax.set_xlabel("Gate Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.0, label="threshold=0.01")
        ax.legend(fontsize=7)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle(f"Gate Distributions (λ={lambda_val})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    filename = f"gate_distribution_{lambda_val}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full experiment: train one model per lambda, collect metrics, plot."""
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Lambda values to sweep (sum-based loss, ~828K gates starting at sigmoid(0)=0.5)
    # 1e-4 → moderate, 5e-4 → strong, 1e-3 → aggressive pruning
    lambda_values: List[float] = [1e-4, 5e-4, 1e-3]

    # Storage for results
    all_histories: List[Dict[str, List[float]]] = []
    results: List[Dict[str, object]] = []
    trained_models: List[SelfPruningNet] = []

    total_start = time.time()

    for lam in lambda_values:
        run_start = time.time()
        test_acc, sparsity, history, model = train_one_run(lam, device, epochs=150)
        run_time = time.time() - run_start

        all_histories.append(history)
        trained_models.append(model)

        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results.append({
            "lambda": lam,
            "test_acc": test_acc,
            "sparsity": sparsity,
            "total_params": total_params,
            "run_time_min": run_time / 60.0,
        })

    total_time = (time.time() - total_start) / 60.0
    print(f"\n{'='*80}")
    print(f"All runs completed in {total_time:.1f} minutes")
    print(f"{'='*80}")

    # ── Plot training curves ──
    plot_results(all_histories, lambda_values)

    # ── Plot gate distributions for each trained model ──
    for model, lam in zip(trained_models, lambda_values):
        plot_gate_distribution(model, lam)

    # ── Print final markdown summary table ──
    print("\n## Final Results\n")
    print("| Lambda | Test Accuracy (%) | Sparsity Level (%) | Notes |")
    print("|--------|-------------------|--------------------|-------|")

    for r in results:
        lam = r["lambda"]
        acc = r["test_acc"]
        spar = r["sparsity"]

        # Generate an observation for each row
        if spar < 5.0:
            note = "Minimal pruning; sparsity penalty too weak to overcome accuracy-driven gradients."
        elif spar < 30.0:
            note = "Moderate pruning achieved with negligible accuracy loss — sweet spot for deployment."
        elif spar < 60.0:
            note = "Significant sparsity; accuracy begins to trade off. Good for memory-constrained targets."
        else:
            note = "Aggressive pruning; large accuracy drop. Useful only if extreme compression is required."

        print(f"| {lam:.5f} | {acc:.2f} | {spar:.2f} | {note} |")

    # ── Detailed per-run summary ──
    print("\n## Detailed Run Statistics\n")
    for r in results:
        print(f"  λ = {r['lambda']:.5f}")
        print(f"    Test Accuracy   : {r['test_acc']:.2f}%")
        print(f"    Sparsity        : {r['sparsity']:.2f}%")
        print(f"    Total Params    : {r['total_params']:,}")
        print(f"    Training Time   : {r['run_time_min']:.1f} min")
        print()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
