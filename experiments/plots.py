from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def belief_traces(episodes, path: Path, with_insider: int = 3) -> None:
    """Belief over time for a few markets, with the true insider onset marked."""
    picks = [e for e in episodes if e.onset is not None and e.onset > 5][:with_insider]
    picks += [e for e in episodes if e.onset is None][:1]

    fig, axes = plt.subplots(len(picks), 1, figsize=(6.5, 1.7 * len(picks)),
                             sharex=True, facecolor=SURFACE)
    for ax, e in zip(axes, picks):
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        steps = np.arange(len(e.prices))
        ax.plot(steps, e.beliefs[:, :, 1].sum(axis=1), color=SERIES[1], lw=2,
                label="P(insider active)")
        ax.plot(steps, e.beliefs[:, 1, :].sum(axis=1), color=SERIES[0], lw=2,
                label="P(YES wins)")
        ax.plot(steps, e.prices, color=MUTED, lw=1, ls=":", label="market price")
        if e.onset is None:
            ax.text(0.5, 0.92, "no insider in this market", color=INK, fontsize=7)
        else:
            ax.axvline(e.onset, color=INK, lw=1, ls="--")
            ax.text(e.onset + 0.5, 0.92, "insider starts", color=INK, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"outcome: {'YES' if e.outcome else 'NO'}", fontsize=8, color=MUTED)

    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="lower left",
                   bbox_to_anchor=(0, 1.02))
    axes[-1].set_xlabel("step", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def return_distributions(episodes_by_policy: dict, path: Path) -> None:
    names = list(episodes_by_policy)
    returns = [[e.reward for e in episodes_by_policy[n]] for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    boxes = ax.boxplot(returns, orientation="horizontal", patch_artist=True, widths=0.5,
                       showfliers=False, medianprops={"color": SURFACE, "lw": 2})
    for box, color in zip(boxes["boxes"], SERIES):
        box.set(facecolor=color, edgecolor=color)
    for row, values in enumerate(returns, start=1):
        ax.plot(np.mean(values), row, "D", color=INK, ms=4)

    ax.set_yticks(range(1, len(names) + 1), names, fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"return per market (\$; each contract pays \$1)   ◆ = mean",
                  fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
