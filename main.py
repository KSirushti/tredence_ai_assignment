import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# ─────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer where each weight is multiplied by a learnable gate.

    Gate mechanism (key insight):
        gate = clamp(gate_scores, 0, 1)

    Why clamp instead of sigmoid:
    - sigmoid(x) ∈ (0, 1) — NEVER reaches 0, only approaches it asymptotically.
      Even with strong L1 pressure, gates hover near 0 but never hit it.
    - clamp(x, 0, 1) = 0  whenever x ≤ 0.
      Once the L1 penalty pushes a gate_score negative, it snaps to exactly 0
      and stays there. The classification loss gradient is also zeroed out
      (d(clamp)/dx = 0 for x < 0), so only the sparsity gradient remains,
      which keeps pushing it further negative → gate stays reliably pruned.

    Forward pass:
        gates         = clamp(gate_scores, 0, 1)
        pruned_weight = weight ⊙ gates
        output        = input @ pruned_weight.T + bias

    Gradients flow through both weight and gate_scores for active gates (>0),
    and only the sparsity gradient acts on zero/negative gate_scores.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate scores: initialized to 1.0 → clamp(1,0,1)=1 → all gates open
        # The optimizer will learn to push unimportant ones below 0.
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

        # Standard Kaiming init for weights
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp gate_scores to [0,1] — can reach exactly 0
        gates = self.gate_scores.clamp(0.0, 1.0)

        # Prune: multiply each weight element-wise by its gate
        pruned_weight = self.weight * gates

        # Standard linear transformation
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Detached gate values for reporting."""
        return self.gate_scores.clamp(0.0, 1.0).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 (3072-d input, 10 classes).
    All linear layers are PrunableLinear so every weight is gated.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def get_all_gates(self) -> torch.Tensor:
        """Flat tensor of all gate values across every PrunableLinear layer."""
        return torch.cat([
            m.get_gates().view(-1)
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        ])

    def sparsity_loss(self) -> torch.Tensor:
        """
        Part 2: L1 penalty on gate values.

        SparsityLoss = mean over all gates of clamp(gate_score, 0, 1)

        Why L1 encourages sparsity here:
        - Gradient of L1 w.r.t. gate_score = +1 for active gates (>0)
          → constant downward push regardless of gate magnitude
        - Unlike L2 (gradient = 2*gate → vanishes near 0), L1 keeps pushing
          gate_scores negative, where clamp() locks them to 0.
        - Combined with clamp() gate, this creates a "snap to zero" effect:
          once a gate is nudged below 0 it is permanently pruned.

        Normalized by total gate count so λ scale is consistent.
        """
        gates_list = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates_list.append(m.gate_scores.clamp(0.0, 1.0).view(-1))
        return torch.cat(gates_list).mean()   # always in [0, 1]

    def compute_sparsity(self, threshold: float = 1e-3) -> float:
        """Percentage of gates effectively at zero (below threshold)."""
        gates = self.get_all_gates()
        return (gates < threshold).float().mean().item() * 100.0


# ─────────────────────────────────────────────
# Part 3: Training & Evaluation
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """CIFAR-10 loaders with standard augmentation."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        "./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_tf)

    # num_workers=0 and pin_memory=False for broad CPU compatibility
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, lam, device):
    """One training epoch. Returns (total_loss, cls_loss, sparsity_loss) means."""
    model.train()
    tot_sum = cls_sum = sp_sum = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits   = model(imgs)
        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()          # ∈ [0, 1]

        # Total Loss = ClassificationLoss + λ * SparsityLoss
        loss = cls_loss + lam * sp_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        tot_sum += loss.item()
        cls_sum += cls_loss.item()
        sp_sum  += sp_loss.item()

    n = len(loader)
    return tot_sum / n, cls_sum / n, sp_sum / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Test accuracy in percent."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


def train_and_evaluate(lam: float, epochs: int, device,
                       train_loader, test_loader, seed: int = 42) -> dict:
    """Full training run for one λ value."""
    torch.manual_seed(seed)
    model = SelfPruningNet().to(device)

    # Two param groups: gate_scores get 10× higher LR so they can
    # respond quickly to the sparsity signal and cross the zero boundary.
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": 1e-3},
        {"params": gate_params,  "lr": 1e-2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Training  λ={lam:.0f}  |  {epochs} epochs  |  device={device}")
    print(f"{'='*60}")

    history = {"cls_loss": [], "sp_loss": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tot, cls, sp = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        history["cls_loss"].append(cls)
        history["sp_loss"].append(sp)

        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.compute_sparsity()
            # Print min gate_score to confirm they go negative (→ pruning)
            gs_min = min(
                m.gate_scores.min().item()
                for m in model.modules() if isinstance(m, PrunableLinear)
            )
            print(f"  Ep {epoch:>3} | CLS={cls:.4f}  SP={sp:.4f}  "
                  f"Sparse={sparsity:.1f}%  gs_min={gs_min:.3f}")

    acc      = evaluate(model, test_loader, device)
    sparsity = model.compute_sparsity()
    print(f"\n  ✓ Test Accuracy : {acc:.2f}%")
    print(f"  ✓ Sparsity      : {sparsity:.2f}%")
    print(f"  ✓ Elapsed       : {time.time()-t0:.1f}s")

    return {"lambda": lam, "test_acc": acc, "sparsity": sparsity,
            "model": model, "history": history}


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_gate_distribution(model, lam, ax):
    gates = model.get_all_gates().cpu().numpy()
    ax.hist(gates, bins=80, color="#e63946", edgecolor="none", alpha=0.85)
    ax.axvline(x=0.001, color="navy", linestyle="--", lw=1.4,
               label="Prune threshold (0.001)")
    sparsity = (gates < 0.001).sum() / len(gates) * 100
    ax.set_title(f"Gate Distribution  (λ={lam:.0f})", fontweight="bold")
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.text(0.55, 0.88, f"Sparsity: {sparsity:.1f}%",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))


def make_plots(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Self-Pruning Neural Network — CIFAR-10",
                 fontsize=14, fontweight="bold")

    # Top row: gate distributions for lowest and highest λ
    plot_gate_distribution(results[0]["model"],  results[0]["lambda"],  axes[0][0])
    plot_gate_distribution(results[-1]["model"], results[-1]["lambda"], axes[0][1])

    # Bottom-left: accuracy vs sparsity grouped bars
    ax = axes[1][0]
    labels = [f"λ={r['lambda']:.0f}" for r in results]
    accs   = [r["test_acc"]  for r in results]
    spars  = [r["sparsity"]  for r in results]
    x, w   = np.arange(len(labels)), 0.35
    ax.bar(x - w/2, accs,  w, label="Test Accuracy (%)",
           color="#457b9d", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + w/2, spars, w, label="Sparsity (%)",
            color="#e63946", alpha=0.70)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Accuracy (%)");  ax2.set_ylabel("Sparsity (%)")
    ax.set_title("Accuracy vs Sparsity Trade-off", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    # Bottom-right: classification loss over epochs
    ax3 = axes[1][1]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for res, c in zip(results, colors):
        ep = range(1, len(res["history"]["cls_loss"]) + 1)
        ax3.plot(ep, res["history"]["cls_loss"],
                 label=f"λ={res['lambda']:.0f}", color=c, lw=1.8)
    ax3.set_title("Classification Loss During Training", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Cross-Entropy Loss")
    ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → gate_distribution.png")
    plt.show()




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EPOCHS     = 20
    BATCH_SIZE = 128
    LAMBDAS = [1, 5, 20]

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results = []
    for lam in LAMBDAS:
        res = train_and_evaluate(lam, EPOCHS, device, train_loader, test_loader)
        results.append(res)

    # Summary table
    print("\n\n" + "="*50)
    print("  RESULTS SUMMARY")
    print("="*50)
    print(f"{'Lambda':>8}  {'Test Accuracy':>14}  {'Sparsity (%)':>14}")
    print("-"*42)
    for r in results:
        print(f"{r['lambda']:>8.0f}  {r['test_acc']:>13.2f}%  {r['sparsity']:>13.2f}%")

    make_plots(results)


if __name__ == "__main__":
    main()