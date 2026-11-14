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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Each policy keeps its colour in every figure.
POLICY_COLORS = {
    "buy-and-hold": SERIES[0],
    "QMDP (insider-blind)": SERIES[1],
    "QMDP": SERIES[2],
    "QMDP (particle filter)": SERIES[3],
    "oracle": SERIES[4],
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def belief_traces(episodes, path: Path, with_insider: int = 3, unit: str = "step") -> None:
    """Belief over time for a few markets, with the true insider onset marked.

    The markets are the first ones in seed order with an insider starting
    after step 5, plus the first insider-free market: not hand-picked."""
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
    axes[-1].set_xlabel(unit, fontsize=8, color=MUTED)
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
    for box, name in zip(boxes["boxes"], names):
        color = POLICY_COLORS.get(name, MUTED)
        box.set(facecolor=color, edgecolor=color)
    for row, values in enumerate(returns, start=1):
        ax.plot(np.mean(values), row, "D", color=INK, ms=4)

    ax.set_yticks(range(1, len(names) + 1), names, fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    n = len(returns[0])
    ax.set_xlabel(rf"return per market (\$; each contract pays \$1), {n} markets"
                  "\nbox: middle half   whiskers: 1.5 × IQR   ◆ mean   white bar: median",
                  fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def onset_aligned(episodes_by_filter: dict, path: Path, before: int = 12, after: int = 24,
                  unit: str = "step") -> None:
    """P(insider active) around the true onset, over every market with an insider.

    Line: median across markets; band: middle half. The dotted line is the
    median in insider-free markets, the false-alarm baseline."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    offsets = np.arange(-before, after + 1)

    for k, (name, episodes) in enumerate(episodes_by_filter.items()):
        color = POLICY_COLORS.get(name, MUTED)
        first = k == 0
        rows = []
        for e in episodes:
            if e.onset is None:
                continue
            prob = e.beliefs[:, :, 1].sum(axis=1)
            idx = e.onset + offsets
            row = np.full(len(offsets), np.nan)
            ok = (idx >= 0) & (idx < len(prob))
            row[ok] = prob[idx[ok]]
            rows.append(row)
        rows = np.array(rows)
        counts = np.sum(~np.isnan(rows), axis=0)
        keep = counts >= max(10, len(rows) // 5)   # drop offsets with few markets
        lo, mid, hi = np.nanpercentile(rows[:, keep], [25, 50, 75], axis=0)
        # Only the first filter gets a band and a baseline; the others are
        # drawn as a line on top for comparison.
        if first:
            ax.fill_between(offsets[keep], lo, hi, color=color, alpha=0.2, lw=0,
                            label=f"{name}: middle half of markets")
        ax.plot(offsets[keep], mid, color=color, lw=2 if first else 1.5,
                ls="-" if first else "--", label=f"{name}: median")

        clean = [e.beliefs[:, :, 1].sum(axis=1) for e in episodes if e.onset is None]
        if first and clean:
            baseline = np.median(np.concatenate(clean))
            ax.axhline(baseline, color=MUTED, lw=1, ls=":",
                       label="median in insider-free markets")

    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.text(0.4, 0.95, "insider starts", color=INK, fontsize=7, va="top")
    ax.set_ylim(0, 1)
    ax.set_xlim(-before, after)
    ax.set_ylabel("P(insider active)", fontsize=8, color=MUTED)
    ax.set_xlabel(f"{unit}s since the insider started (all markets with an insider)",
                  fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
