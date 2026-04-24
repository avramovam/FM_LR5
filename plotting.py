import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.0,
})

def setup_figure(figsize=(14, 8), title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    return fig, ax

def add_figure_caption(fig, caption: str, position: float = 0.02):
    fig.text(position, 0.02, caption, ha='left', va='bottom',
             fontsize=12, style='italic', fontweight='normal',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def plot_single_signal(t: np.ndarray, signal: np.ndarray, title: str,
                      xlabel: str = 'Время $t$, с', ylabel: str = 'Амплитуда',
                      filename: Optional[str] = None, caption: Optional[str] = None,
                      xlim: Optional[Tuple[float, float]] = None):
    fig, ax = setup_figure(title=title)
    ax.plot(t, signal, 'b-', linewidth=2.0)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3, linestyle='--')
    if caption:
        add_figure_caption(fig, caption)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)