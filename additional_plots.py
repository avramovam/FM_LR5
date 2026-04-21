"""
Дополнительные графики для лабораторной работы №5
Запустить после основного main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Импорт модулей
from config import ALL_CONFIGS
from signals import rect_pulse, rect_pulse_analytic_fourier
from fourier_methods import FourierMethods

# Создание директории для результатов
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Настройка стиля для крупных жирных подписей
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
})


def run_experiment(config):
    """Запуск эксперимента для заданной конфигурации"""
    print(f"\n{'=' * 60}")
    print(f"Эксперимент: {config.name}")
    print(f"T = {config.T} с, dt = {config.dt} с")
    print(f"{'=' * 60}")

    # Создание временной сетки
    t = np.arange(-config.T / 2, config.T / 2, config.dt)
    signal = rect_pulse(t)

    # Аналитический Фурье-образ
    nu_analytic = np.linspace(-10, 10, 2000)
    fourier_analytic = rect_pulse_analytic_fourier(nu_analytic)

    # Методы FFT
    fourier_methods = FourierMethods(t, signal)
    nu_trapz = np.linspace(-10, 10, 1000)
    fourier_trapz = fourier_methods.method_trapz(nu_trapz)
    nu_fft, f_hat_naive, recovered_naive = fourier_methods.method_naive_fft()
    nu_fft, f_hat_unitary, recovered_unitary = fourier_methods.method_unitary_fft()
    nu_fft, f_hat_smart, recovered_smart = fourier_methods.method_smart_fft()

    results = {
        'config': config,
        't': t,
        'signal': signal,
        'nu_analytic': nu_analytic,
        'fourier_analytic': fourier_analytic,
        'nu_trapz': nu_trapz,
        'fourier_trapz': fourier_trapz,
        'nu_fft': nu_fft,
        'f_hat_naive': f_hat_naive,
        'f_hat_unitary': f_hat_unitary,
        'f_hat_smart': f_hat_smart,
        'recovered_naive': recovered_naive,
        'recovered_unitary': recovered_unitary,
        'recovered_smart': recovered_smart,
    }

    return results


def plot_naive_fft(results, suffix=""):
    """График: Naive FFT vs аналитический"""
    fig, ax = plt.subplots(figsize=(14, 8))

    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim

    ax.plot(results['nu_analytic'][mask_analytic],
            np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT (без масштаба)', alpha=0.7)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_naive_fft.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_unitary_fft(results, suffix=""):
    """График: Unitary FFT vs аналитический"""
    fig, ax = plt.subplots(figsize=(14, 8))

    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim

    ax.plot(results['nu_analytic'][mask_analytic],
            np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT (1/√N)', alpha=0.7)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_unitary_fft.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_trapz_reconstruction(results, suffix=""):
    """График: Восстановление сигнала после trapz"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Для trapz нужно выполнить обратное преобразование
    # Используем тот же принцип, что и для прямого
    nu_trapz = results['nu_trapz']
    fourier_trapz = results['fourier_trapz']

    # Восстановление через обратное преобразование Фурье
    recovered_trapz = np.zeros_like(results['t'], dtype=complex)
    dt = results['t'][1] - results['t'][0]

    for i, t_val in enumerate(results['t']):
        integrand = fourier_trapz * np.exp(2j * np.pi * nu_trapz * t_val)
        recovered_trapz[i] = np.trapz(integrand, nu_trapz)

    recovered_trapz = recovered_trapz.real

    t_center_mask = np.abs(results['t']) <= 2

    ax.plot(results['t'][t_center_mask], results['signal'][t_center_mask],
            'k-', linewidth=2.5, label='Исходный сигнал', alpha=0.9)
    ax.plot(results['t'][t_center_mask], recovered_trapz[t_center_mask],
            'b--', linewidth=2.0, label='Восстановленный (trapz)', alpha=0.7)

    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_trapz_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_methods_accuracy_comparison(results, suffix=""):
    """График: Сравнение точности методов (увеличенный фрагмент)"""
    fig, ax = plt.subplots(figsize=(14, 8))

    nu_lim = 2
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim

    ax.plot(results['nu_analytic'][mask_analytic],
            np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_recovery_error_comparison(results, suffix=""):
    """График: Сравнение ошибок восстановления разных методов"""
    fig, ax = plt.subplots(figsize=(14, 8))

    t_center_mask = np.abs(results['t']) <= 2

    error_naive = results['signal'][t_center_mask] - results['recovered_naive'][t_center_mask]
    error_unitary = results['signal'][t_center_mask] - results['recovered_unitary'][t_center_mask]
    error_smart = results['signal'][t_center_mask] - results['recovered_smart'][t_center_mask]

    ax.plot(results['t'][t_center_mask], error_naive,
            'g-', linewidth=1.5, label='Ошибка Naive FFT', alpha=0.7)
    ax.plot(results['t'][t_center_mask], error_unitary,
            'm-', linewidth=1.5, label='Ошибка Unitary FFT', alpha=0.7)
    ax.plot(results['t'][t_center_mask], error_smart,
            'r-', linewidth=2.0, label='Ошибка Smart FFT', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Ошибка восстановления', fontsize=20, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_recovery_error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_computational_complexity():
    """График: Сравнение вычислительной сложности методов"""
    fig, ax = plt.subplots(figsize=(14, 8))

    N = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])

    # Теоретическая сложность
    trapz_complexity = N * 500  # O(N * M) с M=500
    fft_complexity = N * np.log2(N)

    # Нормализация для наглядности
    trapz_complexity_norm = trapz_complexity / trapz_complexity[0]
    fft_complexity_norm = fft_complexity / fft_complexity[0]

    ax.plot(N, trapz_complexity_norm, 'b-', linewidth=2.5, label='Trapz: O(N·M)', alpha=0.8)
    ax.plot(N, fft_complexity_norm, 'r-', linewidth=2.5, label='FFT/Smart FFT: O(N log N)', alpha=0.8)

    ax.set_xlabel('Количество точек N', fontsize=20, fontweight='bold')
    ax.set_ylabel('Относительное время выполнения', fontsize=20, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "computational_complexity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_parameter_influence():
    """График: Влияние параметров T и dt на ошибку"""
    # Данные из эксперимента
    configs = ['Хороший 1', 'Хороший 2', 'Плохой 1\n(малый T)', 'Плохой 2\n(большой dt)', 'Плохой 3\n(оба)']
    T_values = [32, 64, 4, 32, 4]
    dt_values = [0.01, 0.005, 0.01, 0.5, 0.5]
    errors = [4.14e-8, 2.56e-9, 2.88e-5, 1.38e-2, 1.03e-2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # График зависимости ошибки от T
    colors = ['green' if e < 1e-6 else 'orange' if e < 1e-3 else 'red' for e in errors]
    ax1.scatter(T_values, errors, s=150, c=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, txt in enumerate(configs):
        ax1.annotate(txt, (T_values[i], errors[i]), fontsize=14, ha='center', va='bottom')
    ax1.set_xlabel('Промежуток времени T, с', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Ошибка спектра (MSE)', fontsize=18, fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Влияние T на ошибку', fontsize=16, fontweight='bold')

    # График зависимости ошибки от dt
    ax2.scatter(dt_values, errors, s=150, c=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for i, txt in enumerate(configs):
        ax2.annotate(txt, (dt_values[i], errors[i]), fontsize=14, ha='center', va='bottom')
    ax2.set_xlabel('Шаг дискретизации Δt, с', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Ошибка спектра (MSE)', fontsize=18, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Влияние Δt на ошибку', fontsize=16, fontweight='bold')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "parameter_influence.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_methods_spectrum(results, suffix=""):
    """График: Все методы на одном графике (полный обзор)"""
    fig, ax = plt.subplots(figsize=(14, 8))

    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim

    ax.plot(results['nu_analytic'][mask_analytic],
            np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический (эталон)', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft],
            np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)
    ax.plot(results['nu_trapz'][np.abs(results['nu_trapz']) <= nu_lim],
            np.abs(results['fourier_trapz'][np.abs(results['nu_trapz']) <= nu_lim]),
            'b:', linewidth=2.0, label='Trapz', alpha=0.7)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_all_methods_full.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    """Главная функция"""
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДОПОЛНИТЕЛЬНЫХ ГРАФИКОВ")
    print("=" * 60)

    # Генерация графика вычислительной сложности
    plot_computational_complexity()
    print("✓ computational_complexity.png")

    # Генерация графика влияния параметров
    plot_parameter_influence()
    print("✓ parameter_influence.png")

    # Для каждой конфигурации
    for config in ALL_CONFIGS:
        results = run_experiment(config)
        suffix = config.name.replace(' ', '_').replace('-', '_').replace('ё', 'e')

        # Дополнительные графики
        plot_naive_fft(results, suffix)
        print(f"✓ {suffix}_naive_fft.png")

        plot_unitary_fft(results, suffix)
        print(f"✓ {suffix}_unitary_fft.png")

        plot_trapz_reconstruction(results, suffix)
        print(f"✓ {suffix}_trapz_reconstruction.png")

        plot_methods_accuracy_comparison(results, suffix)
        print(f"✓ {suffix}_accuracy_comparison.png")

        plot_recovery_error_comparison(results, suffix)
        print(f"✓ {suffix}_recovery_error_comparison.png")

        plot_all_methods_spectrum(results, suffix)
        print(f"✓ {suffix}_all_methods_full.png")

    print("\n" + "=" * 60)
    print("ВСЕ ГРАФИКИ УСПЕШНО СОЗДАНЫ!")
    print(f"Результаты сохранены в: {RESULTS_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()