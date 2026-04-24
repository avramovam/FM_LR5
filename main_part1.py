import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import ALL_CONFIGS
from signals import rect_pulse, rect_pulse_analytic_fourier
from fourier_methods import FourierMethods

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 24,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
})


def run_experiment(config):
    print(f"\n{'='*60}")
    print(f"Эксперимент: {config.name}")
    print(f"T = {config.T} с, dt = {config.dt} с")
    print(f"N = {config.N} точек, fs = {1/config.dt:.2f} Гц")
    print(f"{'='*60}")

    t = np.arange(-config.T/2, config.T/2, config.dt)
    signal = rect_pulse(t)

    nu_analytic = np.linspace(-10, 10, 2000)
    fourier_analytic = rect_pulse_analytic_fourier(nu_analytic)

    fourier_methods = FourierMethods(t, signal)
    nu_trapz = np.linspace(-10, 10, 1000)
    fourier_trapz = fourier_methods.method_trapz(nu_trapz)
    nu_fft, f_hat_naive, recovered_naive = fourier_methods.method_naive_fft()
    nu_fft, f_hat_unitary, recovered_unitary = fourier_methods.method_unitary_fft()
    nu_fft, f_hat_smart, recovered_smart = fourier_methods.method_smart_fft()

    recovered_trapz = np.zeros_like(t, dtype=complex)
    for i, t_val in enumerate(t):
        integrand = fourier_trapz * np.exp(2j * np.pi * nu_trapz * t_val)
        recovered_trapz[i] = np.trapz(integrand, nu_trapz)
    recovered_trapz = recovered_trapz.real

    from scipy.interpolate import interp1d
    f_smart_mag = np.abs(f_hat_smart)
    interp_func = interp1d(nu_fft, f_smart_mag, kind='linear', bounds_error=False, fill_value=0)
    f_smart_interp = interp_func(nu_analytic)
    spectrum_error = np.mean((np.abs(fourier_analytic) - f_smart_interp)**2)

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
        'recovered_trapz': recovered_trapz,
        'spectrum_error': spectrum_error
    }

    print(f"Ошибка аппроксимации спектра: {spectrum_error:.6e}")

    alias_info = fourier_methods.analyze_alias()
    if alias_info['aliasing_present']:
        print(f"  ВНИМАНИЕ: Алиасинг! fs/2 = {alias_info['nyquist_frequency']:.2f} Гц")

    return results


def plot_signal(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(results['t'], results['signal'], 'b-', linewidth=2.5)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_signal.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_analytic_spectrum(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask = np.abs(results['nu_analytic']) <= nu_lim
    ax.plot(results['nu_analytic'][mask], np.abs(results['fourier_analytic'][mask]), 'k-', linewidth=2.5)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_analytic_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_analytic_vs_trapz(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_trapz = np.abs(results['nu_trapz']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_trapz'][mask_trapz], np.abs(results['fourier_trapz'][mask_trapz]),
            'b--', linewidth=2.0, label='Trapz', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_analytic_vs_trapz.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_naive_fft(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_naive_fft.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_unitary_fft(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_unitary_fft.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_analytic_vs_smart(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r--', linewidth=2.0, label='Smart FFT', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_analytic_vs_smart.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_methods_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    mask_trapz = np.abs(results['nu_trapz']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)
    ax.plot(results['nu_trapz'][mask_trapz], np.abs(results['fourier_trapz'][mask_trapz]),
            'b:', linewidth=2.0, label='Trapz', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_all_methods_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_accuracy_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 2
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=1.5, label='Naive FFT', alpha=0.5)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=1.5, label='Unitary FFT', alpha=0.5)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reconstruction_smart(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    t_center_mask = np.abs(results['t']) <= 2
    ax.plot(results['t'][t_center_mask], results['signal'][t_center_mask],
            'k-', linewidth=2.5, label='Исходный сигнал', alpha=0.9)
    ax.plot(results['t'][t_center_mask], results['recovered_smart'][t_center_mask],
            'r--', linewidth=2.0, label='Восстановленный (Smart FFT)', alpha=0.7)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_smart_vs_trapz_vs_analytic(results, suffix="", save=True):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    mask_trapz = np.abs(results['nu_trapz']) <= nu_lim

    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_trapz'][mask_trapz], np.abs(results['fourier_trapz'][mask_trapz]),
            'b--', linewidth=2.0, label='Trapz', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / f"{suffix}_smart_vs_trapz_zoom.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_reconstruction_error(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    t_center_mask = np.abs(results['t']) <= 2
    error = results['signal'][t_center_mask] - results['recovered_smart'][t_center_mask]
    ax.plot(results['t'][t_center_mask], error, 'r-', linewidth=2.0, label='Ошибка восстановления')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Ошибка', fontsize=20, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_reconstruction_error.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_recovery_error_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    t_center_mask = np.abs(results['t']) <= 2
    error_naive = results['signal'][t_center_mask] - results['recovered_naive'][t_center_mask]
    error_unitary = results['signal'][t_center_mask] - results['recovered_unitary'][t_center_mask]
    error_smart = results['signal'][t_center_mask] - results['recovered_smart'][t_center_mask]
    error_trapz = results['signal'][t_center_mask] - results['recovered_trapz'][t_center_mask]
    ax.plot(results['t'][t_center_mask], error_naive, 'g-', linewidth=1.5, label='Naive FFT', alpha=0.7)
    ax.plot(results['t'][t_center_mask], error_unitary, 'm-', linewidth=1.5, label='Unitary FFT', alpha=0.7)
    ax.plot(results['t'][t_center_mask], error_smart, 'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)
    ax.plot(results['t'][t_center_mask], error_trapz, 'b--', linewidth=1.5, label='Trapz', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Ошибка восстановления', fontsize=20, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_recovery_error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_log_spectrum(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    magnitude = np.abs(results['f_hat_smart'][mask_fft])
    magnitude[magnitude < 1e-10] = 1e-10
    ax.semilogy(results['nu_fft'][mask_fft], magnitude, 'r-', linewidth=2.0)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$ (лог. шкала)', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_log_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_methods_full(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    mask_trapz = np.abs(results['nu_trapz']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_naive'][mask_fft]),
            'g--', linewidth=2.0, label='Naive FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_unitary'][mask_fft]),
            'm-.', linewidth=2.0, label='Unitary FFT', alpha=0.7)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r-', linewidth=2.0, label='Smart FFT', alpha=0.8)
    ax.plot(results['nu_trapz'][mask_trapz], np.abs(results['fourier_trapz'][mask_trapz]),
            'b:', linewidth=2.0, label='Trapz', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=14, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_all_methods_full.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_low_resolution_spectrum(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]), 'r-', linewidth=2.0)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_low_res_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_low_resolution_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r--', linewidth=2.0, label='Smart FFT', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_low_res_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_aliasing_spectrum(results, suffix=""):
    config = results['config']
    fs = 1 / config.dt
    nyquist = fs / 2
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]), 'r-', linewidth=2.0)
    ax.axvline(x=nyquist, color='red', linestyle='--', alpha=0.7, linewidth=2.5, label=f'Частота Найквиста = {nyquist:.2f} Гц')
    ax.axvline(x=-nyquist, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_aliasing_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_aliasing_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 10
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r--', linewidth=2.0, label='Smart FFT (искажён)', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_aliasing_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_zoom_spectrum(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 2
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]), 'r-', linewidth=2.0)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_zoom_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_zoom_comparison(results, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 8))
    nu_lim = 2
    mask_analytic = np.abs(results['nu_analytic']) <= nu_lim
    mask_fft = np.abs(results['nu_fft']) <= nu_lim
    ax.plot(results['nu_analytic'][mask_analytic], np.abs(results['fourier_analytic'][mask_analytic]),
            'k-', linewidth=2.5, label='Аналитический', alpha=0.9)
    ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
            'r--', linewidth=2.0, label='Smart FFT', alpha=0.7)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{suffix}_zoom_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_errors_summary(all_results):
    fig, ax = plt.subplots(figsize=(16, 10))
    config_names = [r['config'].name for r in all_results]
    errors = [r['spectrum_error'] for r in all_results]
    colors = ['green' if e < 1e-6 else 'orange' if e < 1e-3 else 'red' for e in errors]
    bars = ax.bar(config_names, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Среднеквадратичная ошибка (MSE)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Конфигурация эксперимента', fontsize=20, fontweight='bold')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{err:.2e}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "errors_summary.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_all_spectra_comparison(all_results):
    fig, ax = plt.subplots(figsize=(16, 10))
    nu_lim = 10
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    line_styles = ['-', '--', '-.', ':', '-']
    for i, results in enumerate(all_results):
        cfg = results['config']
        mask_fft = np.abs(results['nu_fft']) <= nu_lim
        ax.plot(results['nu_fft'][mask_fft], np.abs(results['f_hat_smart'][mask_fft]),
                color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                linewidth=2.0, label=cfg.name, alpha=0.8)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{\\Pi}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=16, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "all_spectra_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_parameter_influence(all_results):
    configs = [r['config'] for r in all_results]
    T_values = [cfg.T for cfg in configs]
    dt_values = [cfg.dt for cfg in configs]
    errors = [r['spectrum_error'] for r in all_results]
    colors = ['green' if e < 1e-6 else 'orange' if e < 1e-3 else 'red' for e in errors]
    names = [cfg.name for cfg in configs]

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for i, name in enumerate(names):
        ax1.scatter(T_values[i], errors[i], s=150, color=colors[i], alpha=0.7,
                    edgecolors='black', linewidth=1.5, label=name, zorder=5)
    ax1.set_xlabel('Промежуток времени T, с', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Ошибка спектра (MSE)', fontsize=20, fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Влияние T на ошибку', fontsize=20, fontweight='bold')
    ax1.legend(fontsize=16, loc='best', framealpha=0.9, prop={'weight': 'bold'})
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / "parameter_influence_T.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    for i, name in enumerate(names):
        ax2.scatter(dt_values[i], errors[i], s=150, color=colors[i], alpha=0.7,
                    edgecolors='black', linewidth=1.5, label=name, zorder=5)
    ax2.set_xlabel('Шаг дискретизации Δt, с', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Ошибка спектра (MSE)', fontsize=20, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Влияние Δt на ошибку', fontsize=20, fontweight='bold')
    ax2.legend(fontsize=16, loc='best', framealpha=0.9, prop={'weight': 'bold'})
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "parameter_influence_dt.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)


def plot_computational_complexity():
    fig, ax = plt.subplots(figsize=(14, 8))
    N = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
    trapz_complexity = N * 500
    fft_complexity = N * np.log2(N)
    trapz_complexity_norm = trapz_complexity / trapz_complexity[0]
    fft_complexity_norm = fft_complexity / fft_complexity[0]
    ax.plot(N, trapz_complexity_norm, 'b-', linewidth=2.5, label='Trapz: O(N·M)', alpha=0.8)
    ax.plot(N, fft_complexity_norm, 'r-', linewidth=2.5, label='FFT/Smart FFT: O(N log N)', alpha=0.8)
    ax.set_xlabel('Количество точек N', fontsize=20, fontweight='bold')
    ax.set_ylabel('Относительное время выполнения', fontsize=20, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=18, framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "computational_complexity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_results_for_config(results):
    config = results['config']
    suffix = config.name.replace(' ', '_').replace('-', '_').replace('ё', 'e')

    plot_signal(results, suffix)
    plot_analytic_spectrum(results, suffix)
    plot_analytic_vs_trapz(results, suffix)
    plot_naive_fft(results, suffix)
    plot_unitary_fft(results, suffix)
    plot_analytic_vs_smart(results, suffix)
    plot_all_methods_comparison(results, suffix)
    plot_accuracy_comparison(results, suffix)
    plot_reconstruction_smart(results, suffix)
    plot_reconstruction_error(results, suffix)
    plot_recovery_error_comparison(results, suffix)
    plot_log_spectrum(results, suffix)
    plot_all_methods_full(results, suffix)
    plot_smart_vs_trapz_vs_analytic(results, suffix)

    fs = 1 / config.dt
    nyquist = fs / 2

    if nyquist < 10:
        plot_aliasing_spectrum(results, suffix)
        plot_aliasing_comparison(results, suffix)
        plot_zoom_spectrum(results, suffix)
        plot_zoom_comparison(results, suffix)
    elif config.T < 20:
        plot_low_resolution_spectrum(results, suffix)
        plot_low_resolution_comparison(results, suffix)
        plot_zoom_spectrum(results, suffix)
        plot_zoom_comparison(results, suffix)


def main():
    all_results = []
    for config in ALL_CONFIGS:
        results = run_experiment(config)
        plot_results_for_config(results)
        all_results.append(results)

    plot_errors_summary(all_results)
    plot_all_spectra_comparison(all_results)
    plot_parameter_influence(all_results)
    plot_computational_complexity()

    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    print(f"{'Конфигурация':<35} {'T (с)':<10} {'dt (с)':<10} {'fs/2 (Гц)':<12} {'Ошибка спектра':<15}")
    print("-"*80)
    for r in all_results:
        cfg = r['config']
        fs_half = 1/(2*cfg.dt)
        print(f"{cfg.name:<35} {cfg.T:<10.2f} {cfg.dt:<10.4f} {fs_half:<12.2f} {r['spectrum_error']:<15.2e}")
    print("="*80)

    print(f"\nРезультаты сохранены в: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()