"""
Функции для построения графиков с красивым оформлением
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

# Настройка стиля для крупных подписей (A4-friendly)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})

def setup_figure(figsize=(14, 8), title: Optional[str] = None):
    """
    Создание и настройка фигуры
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    return fig, ax

def add_figure_caption(fig, caption: str, position: float = 0.02):
    """
    Добавление подрисуночной подписи
    """
    fig.text(position, 0.02, caption, ha='left', va='bottom',
             fontsize=12, style='italic', fontweight='normal',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def plot_signal_comparison(t: np.ndarray, signals: List[Tuple[np.ndarray, str, str]],
                           title: str, xlabel: str = 'Время $t$, с',
                           ylabel: str = 'Амплитуда',
                           filename: Optional[str] = None,
                           show_grid: bool = True,
                           vertical_lines: Optional[List[float]] = None,
                           caption: Optional[str] = None):
    """
    Построение сравнения нескольких сигналов

    Параметры:
    -----------
    t : np.ndarray
        Временная сетка
    signals : List[Tuple[np.ndarray, str, str]]
        Список кортежей (сигнал, название, цвет)
    title : str
        Заголовок графика
    xlabel, ylabel : str
        Подписи осей
    filename : str, optional
        Имя файла для сохранения
    show_grid : bool
        Показывать сетку
    vertical_lines : List[float], optional
        Вертикальные линии для отметок
    caption : str, optional
        Подрисуночная подпись
    """
    fig, ax = setup_figure(title=title)

    for signal, name, color in signals:
        ax.plot(t, signal, linewidth=2.0, label=name, color=color, alpha=0.85)

    if vertical_lines:
        for i, x in enumerate(vertical_lines):
            ax.axvline(x=x, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                      label=f'ν = {x} Гц' if i == 0 else '')

    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=14, framealpha=0.9, edgecolor='black')
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')

    # Добавляем подрисуночную подпись
    if caption:
        add_figure_caption(fig, caption)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")

    return fig, ax

def plot_fourier_comparison(nu: np.ndarray, fourier_list: List[Tuple[np.ndarray, str, str]],
                           title: str, xlabel: str = 'Частота $\\nu$, Гц',
                           ylabel: str = '$|\\hat{\\Pi}(\\nu)|$',
                           use_log: bool = False,
                           filename: Optional[str] = None,
                           xlim: Optional[Tuple[float, float]] = None,
                           vertical_lines: Optional[List[float]] = None,
                           caption: Optional[str] = None):
    """
    Построение сравнения Фурье-образов

    Параметры:
    -----------
    nu : np.ndarray
        Частотная сетка
    fourier_list : List[Tuple[np.ndarray, str, str]]
        Список кортежей (Фурье-образ, название, цвет)
    title : str
        Заголовок графика
    use_log : bool
        Использовать логарифмическую шкалу по y
    vertical_lines : List[float], optional
        Вертикальные линии для отметок
    caption : str, optional
        Подрисуночная подпись
    """
    fig, ax = setup_figure(title=title)

    for fourier, name, color in fourier_list:
        magnitude = np.abs(fourier)
        ax.plot(nu, magnitude, linewidth=2.0, label=name, color=color, alpha=0.85)

    if vertical_lines:
        for i, x in enumerate(vertical_lines):
            ax.axvline(x=x, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                      label=f'ν = {x} Гц' if i == 0 else '')

    if xlim:
        ax.set_xlim(xlim)

    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel(f'{ylabel} (лог. шкала)', fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=14, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')

    if caption:
        add_figure_caption(fig, caption)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")

    return fig, ax

def plot_reconstruction_error(config_names: List[str], errors: List[float],
                             title: str = 'Ошибка восстановления сигнала',
                             filename: Optional[str] = None,
                             caption: Optional[str] = None):
    """
    Построение графика ошибок восстановления
    """
    fig, ax = setup_figure(figsize=(14, 8), title=title)

    colors = ['green' if e < 1e-10 else 'orange' if e < 1e-8 else 'red' for e in errors]
    bars = ax.bar(config_names, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Конфигурация эксперимента', fontsize=16, fontweight='bold')
    ax.set_ylabel('Среднеквадратичная ошибка (MSE)', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Добавление значений на столбцы
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{err:.1e}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=12)

    if caption:
        add_figure_caption(fig, caption)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")

    return fig, ax

def plot_signal_with_samples(t: np.ndarray, signal_cont: np.ndarray,
                            t_samples: np.ndarray, samples: np.ndarray,
                            t_reconstructed: np.ndarray, reconstructed: np.ndarray,
                            title: str = 'Исходный, сэмплированный и восстановленный сигналы',
                            filename: Optional[str] = None,
                            caption1: Optional[str] = None,
                            caption2: Optional[str] = None):
    """
    Построение сигнала с сэмплированием (для задания 2)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Верхний график: исходный + сэмплы
    axes[0].plot(t, signal_cont, 'b-', linewidth=2.0, label='Исходный сигнал', alpha=0.8)
    axes[0].stem(t_samples, samples, linefmt='r-', markerfmt='ro', basefmt='gray',
                label='Сэмплированные значения', markersize=8)
    axes[0].set_xlabel('Время $t$, с', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Амплитуда', fontsize=16, fontweight='bold')
    axes[0].set_title('Исходный сигнал и его дискретные отсчеты', fontsize=16, fontweight='bold')
    axes[0].legend(loc='best', fontsize=14, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    if caption1:
        axes[0].text(0.02, 0.98, caption1, transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Нижний график: исходный + восстановленный
    axes[1].plot(t, signal_cont, 'b-', linewidth=2.0, label='Исходный сигнал', alpha=0.8)
    axes[1].plot(t_reconstructed, reconstructed, 'g--', linewidth=2.0,
                label='Восстановленный сигнал (интерполяция)', alpha=0.9)
    axes[1].set_xlabel('Время $t$, с', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Амплитуда', fontsize=16, fontweight='bold')
    axes[1].set_title('Сравнение исходного и восстановленного сигналов', fontsize=16, fontweight='bold')
    axes[1].legend(loc='best', fontsize=14, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    if caption2:
        axes[1].text(0.02, 0.98, caption2, transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")

    return fig, axes

def plot_single_signal(t: np.ndarray, signal: np.ndarray, title: str,
                      xlabel: str = 'Время $t$, с', ylabel: str = 'Амплитуда',
                      filename: Optional[str] = None, caption: Optional[str] = None,
                      xlim: Optional[Tuple[float, float]] = None):
    """
    Построение одного сигнала с подписью
    """
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
        print(f"График сохранен: {filename}")

    return fig, ax