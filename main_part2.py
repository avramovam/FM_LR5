import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
})


def dft_reconstruction(samples, t_samples, t_rec, B):
    """
    Восстановление сигнала через ДПФ (эквивалентно интерполяции Котельникова)

    Алгоритм:
    1. Вычисляем спектр сэмплированного сигнала через БПФ
    2. Обнуляем частоты выше B (идеальный фильтр)
    3. Восстанавливаем сигнал через обратное БПФ
    4. Интерполируем на нужную сетку
    """
    dt_sample = t_samples[1] - t_samples[0]
    N = len(samples)

    # Вычисляем спектр
    fft_vals = np.fft.fft(samples)
    fft_vals = np.fft.fftshift(fft_vals)

    # Частотная сетка
    freqs = np.fft.fftshift(np.fft.fftfreq(N, dt_sample))

    # Обнуляем частоты выше B (идеальный фильтр)
    fft_filtered = fft_vals.copy()
    fft_filtered[np.abs(freqs) > B] = 0

    # Восстанавливаем сигнал
    recovered_n = np.fft.ifft(np.fft.ifftshift(fft_filtered)).real

    # Интерполируем на нужную сетку
    interp = interp1d(t_samples, recovered_n, kind='cubic',
                      bounds_error=False, fill_value=0)
    recovered = interp(t_rec)

    return recovered


def sinc_interpolation(samples, t_samples, t_rec):
    """
    Интерполяционная формула Котельникова
    
    f(t) = Σ f[n] * sinc((t - n*Ts)/Ts) = Σ f[n] * sinc(fs * (t - n*Ts))
    
    где Ts - шаг сэмплирования, fs = 1/Ts - частота сэмплирования
    """
    Ts = t_samples[1] - t_samples[0]  # Шаг сэмплирования
    fs = 1.0 / Ts  # Частота сэмплирования
    
    recovered = np.zeros_like(t_rec, dtype=float)

    for i, t_val in enumerate(t_rec):
        arg = fs * (t_val - t_samples)  # Правильный аргумент: fs * (t - n*Ts)
        sinc_vals = np.sinc(arg)
        recovered[i] = np.sum(samples * sinc_vals)

    return recovered


def analyze_signal(y, t, t_samples, samples, recovered, B, name):
    """Анализ сигнала: вычисление ошибки и спектров через Smart FFT (п.1.8)"""
    mse = np.mean((y - recovered)**2)
    print(f"  MSE восстановления для {name}: {mse:.6e}")

    from fourier_methods import FourierMethods

    # Спектр исходного непрерывного сигнала
    fourier_methods_cont = FourierMethods(t, y)
    nu_cont, f_cont, _ = fourier_methods_cont.method_smart_fft()

    # Спектр сэмплированного сигнала
    fourier_methods_samp = FourierMethods(t_samples, samples)
    nu_samp, f_samp, _ = fourier_methods_samp.method_smart_fft()

    # Спектр восстановленного сигнала
    fourier_methods_rec = FourierMethods(t, recovered)
    nu_rec, f_rec, _ = fourier_methods_rec.method_smart_fft()

    return {
        'mse': mse,
        'nu_cont': nu_cont, 'f_cont': f_cont,
        'nu_samp': nu_samp, 'f_samp': f_samp,
        'nu_rec': nu_rec, 'f_rec': f_rec,
        'B': B
    }


def plot_three_spectra_comparison(results, signal_name, dt_sample, save=True):
    """
    Сравнение трёх спектров: исходный / сэмплированный / восстановленный
    с вертикальными линиями частоты полосы сигнала B (теоретической)
    """
    nu_lim = 25 if signal_name == 'y1' else 15

    # Теоретическая полоса сигнала (для вертикальных линий)
    if signal_name == 'y1':
        B_theoretical = 12.0  # максимальная частота сигнала
    else:
        B_theoretical = 8.0  # полоса спектра sinc

    # Параметр фильтрации (чуть больше для устойчивости)
    B_filter = results['B']

    fig, ax = plt.subplots(figsize=(14, 8))

    mask_cont = np.abs(results['nu_cont']) <= nu_lim
    mask_samp = np.abs(results['nu_samp']) <= nu_lim
    mask_rec = np.abs(results['nu_rec']) <= nu_lim

    ax.plot(results['nu_cont'][mask_cont], np.abs(results['f_cont'][mask_cont]),
            'b-', linewidth=2.0, label='Исходный сигнал', alpha=0.8)
    ax.plot(results['nu_samp'][mask_samp], np.abs(results['f_samp'][mask_samp]),
            'r--', linewidth=1.5, label='Сэмплированный', alpha=0.7)
    ax.plot(results['nu_rec'][mask_rec], np.abs(results['f_rec'][mask_rec]),
            'g-.', linewidth=1.5, label='Восстановленный', alpha=0.7)

    # Вертикальные линии — ТЕОРЕТИЧЕСКАЯ полоса сигнала
    ax.axvline(x=B_theoretical, color='black', linestyle=':', linewidth=2.5,
               label=f'Полоса сигнала $B = {B_theoretical}$ Гц', alpha=0.9)
    ax.axvline(x=-B_theoretical, color='black', linestyle=':', linewidth=2.5, alpha=0.9)

    # Частота Найквиста
    nyquist = 1 / (2 * dt_sample)
    ax.axvline(x=nyquist, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
               label=f'Частота Найквиста = {nyquist:.2f} Гц')
    ax.axvline(x=-nyquist, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{y}(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.legend(loc='best', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(f'Сравнение спектров {signal_name.upper()} ($\\Delta t = {dt_sample}$ с)',
                 fontsize=20, fontweight='bold')

    plt.tight_layout()
    if save:
        safe_name = signal_name.replace('$', '').replace('_', '')
        fig.savefig(RESULTS_DIR / f"{safe_name}_spectra_three_dt{dt_sample}.png",
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def y1_signal(t, a1=1.0, a2=0.5, f1=5.0, f2=12.0):
    """y1(t) = a1·sin(2π f1 t) + a2·sin(2π f2 t)"""
    return a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)


def y2_signal(t, B=8.0):
    """y2(t) = sinc(2·B·t)"""
    return np.sinc(2 * B * t)


# ========== ПОСТРОЕНИЕ БАЗОВЫХ ГРАФИКОВ ==========

def plot_y1_continuous(save=True):
    """График y1(t) без видимой дискретизации — пункт 2.2(а)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    T_total = 5.0
    dt_fine = 0.001
    t = np.arange(0, T_total, dt_fine)
    y = y1_signal(t)
    ax.plot(t, y, 'b-', linewidth=2.0)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.set_xlim(0, T_total)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_title('Сигнал $y_1(t) = \\sin(2\\pi\\cdot5t) + 0.5\\sin(2\\pi\\cdot12t)$',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "y1_continuous.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_y1_spectrum(save=True):
    """Спектр сигнала y1(t)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    T_total = 10.0
    dt_fine = 0.001
    t = np.arange(0, T_total, dt_fine)
    y = y1_signal(t)
    from fourier_methods import FourierMethods
    fourier_methods = FourierMethods(t, y)
    nu_fft, f_hat, _ = fourier_methods.method_smart_fft()
    nu_lim = 20
    mask = np.abs(nu_fft) <= nu_lim
    ax.plot(nu_fft[mask], np.abs(f_hat[mask]), 'b-', linewidth=2.0)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{y}_1(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='5 Гц')
    ax.axvline(x=-5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=12, color='green', linestyle='--', alpha=0.7, linewidth=2, label='12 Гц')
    ax.axvline(x=-12, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "y1_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_y2_continuous(save=True):
    """График y2(t) без видимой дискретизации — пункт 2.2(а)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    T_total = 5.0
    dt_fine = 0.001
    t = np.arange(0, T_total, dt_fine)
    y = y2_signal(t)
    ax.plot(t, y, 'b-', linewidth=2.0)
    ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.set_xlim(0, T_total)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_title('Сигнал $y_2(t) = \\operatorname{sinc}(16\\pi t)$',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "y2_continuous.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_y2_spectrum(save=True):
    """Спектр сигнала y2(t)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    T_total = 10.0
    dt_fine = 0.001
    t = np.arange(0, T_total, dt_fine)
    y = y2_signal(t)
    from fourier_methods import FourierMethods
    fourier_methods = FourierMethods(t, y)
    nu_fft, f_hat, _ = fourier_methods.method_smart_fft()
    nu_lim = 15
    mask = np.abs(nu_fft) <= nu_lim
    ax.plot(nu_fft[mask], np.abs(f_hat[mask]), 'b-', linewidth=2.0)
    ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('$|\\hat{y}_2(\\nu)|$', fontsize=20, fontweight='bold')
    ax.set_xlim(-nu_lim, nu_lim)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, linewidth=2, label='8 Гц')
    ax.axvline(x=-8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "y2_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ========== ЭКСПЕРИМЕНТЫ ==========

def experiment_y1(dt_sample, T_sampling=10.0, T_display=5.0, save=True):
    """
    Эксперимент с сигналом y1(t)

    Параметры:
    dt_sample - шаг дискретизации
    T_sampling - интервал сэмплирования (10 с, большой для точности)
    T_display - интервал отображения (5 с, короткий для наглядности)
    """
    print(f"\n  Исследование y1(t) с dt_sample = {dt_sample} с (fs = {1/dt_sample:.2f} Гц)")

    dt_fine = 0.0005
    B = 12.5  # Чуть больше максимальной частоты (12 Гц)
    nyquist = 1 / (2 * dt_sample)

    # Непрерывный сигнал на полном интервале
    t_full = np.arange(0, T_sampling, dt_fine)
    y_full = y1_signal(t_full)

    # Сэмплирование на полном интервале
    t_samples = np.arange(0, T_sampling, dt_sample)
    samples = y1_signal(t_samples)

    # Интервал отображения и восстановления
    t_rec = np.arange(0, T_display, dt_fine)
    y_original_display = y1_signal(t_rec)

    # Восстановление через ДПФ
    recovered = dft_reconstruction(samples, t_samples, t_rec, B)

    # Анализ
    results = analyze_signal(y_original_display, t_rec, t_samples, samples, recovered, B,
                             f"y1 (dt={dt_sample})")

    if save:
        # 1. График сэмплирования (дискретный график поверх непрерывного) — пункт 2.2(б,в)
        fig, ax = plt.subplots(figsize=(14, 8))
        mask_display = t_full <= T_display
        ax.plot(t_full[mask_display], y_full[mask_display], 'b-', linewidth=2.0,
                label='Исходный сигнал', alpha=0.7)
        mask_samples_display = t_samples <= T_display
        ax.plot(t_samples[mask_samples_display], samples[mask_samples_display],
                'ro', markersize=8, label=f'Отсчёты (Δt={dt_sample} с)')
        ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
        ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
        ax.set_xlim(0, T_display)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Дискретизация сигнала $y_1(t)$ (Δt = {dt_sample} с, fs/2 = {nyquist:.2f} Гц)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y1_sampling_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


        fig, ax = plt.subplots(figsize=(14, 8))


        ax.plot(t_rec, y_original_display, 'b-', linewidth=2.0,
                label='Исходный сигнал', alpha=0.7)


        ax.plot(t_rec, recovered, 'r--', linewidth=2.0,
                label='Восстановленный сигнал (Котельников)', alpha=0.7)


        mask_samples_display = t_samples <= T_display
        ax.plot(t_samples[mask_samples_display], samples[mask_samples_display],
                'ro', markersize=6, label=f'Отсчёты (Δt={dt_sample} с)', alpha=0.8)

        ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
        ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
        ax.set_xlim(0, T_display)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Восстановление сигнала $y_1(t)$\n'
                     f'$\\Delta t = {dt_sample}$ с, $f_s/2 = {nyquist:.2f}$ Гц, '
                     f'MSE = {results["mse"]:.2e}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y1_recovered_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 8))
        nu_lim = 25
        mask_cont = np.abs(results['nu_cont']) <= nu_lim
        mask_rec = np.abs(results['nu_rec']) <= nu_lim
        ax.plot(results['nu_cont'][mask_cont], np.abs(results['f_cont'][mask_cont]),
                'b-', linewidth=2.0, label='Спектр исходного сигнала', alpha=0.7)
        ax.plot(results['nu_rec'][mask_rec], np.abs(results['f_rec'][mask_rec]),
                'r--', linewidth=2.0, label='Спектр восстановленного сигнала', alpha=0.7)
        ax.axvline(x=nyquist, color='red', linestyle=':', alpha=0.7, linewidth=2,
                   label=f'Частота Найквиста = {nyquist:.2f} Гц')
        ax.axvline(x=-nyquist, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax.axvline(x=12, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='12 Гц')
        ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='5 Гц')
        ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
        ax.set_ylabel('$|\\hat{y}(\\nu)|$', fontsize=20, fontweight='bold')
        ax.set_xlim(-nu_lim, nu_lim)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Спектры сигнала $y_1(t)$ (Δt = {dt_sample} с)',
                     fontsize=20, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y1_spectra_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_three_spectra_comparison(results, signal_name='y1', dt_sample=dt_sample, save=True)

    return results


def experiment_y2(dt_sample, T_sampling=10.0, T_display=5.0, save=True):
    """Эксперимент с сигналом y2(t)"""
    print(f"\n  Исследование y2(t) с dt_sample = {dt_sample} с (fs = {1/dt_sample:.2f} Гц)")

    dt_fine = 0.0005
    B = 8.5  # Чуть больше полосы сигнала (8 Гц)
    nyquist = 1 / (2 * dt_sample)

    t_full = np.arange(0, T_sampling, dt_fine)
    y_full = y2_signal(t_full)

    t_samples = np.arange(0, T_sampling, dt_sample)
    samples = y2_signal(t_samples)

    t_rec = np.arange(0, T_display, dt_fine)
    y_original_display = y2_signal(t_rec)

    recovered = dft_reconstruction(samples, t_samples, t_rec, B)

    results = analyze_signal(y_original_display, t_rec, t_samples, samples, recovered, B,
                             f"y2 (dt={dt_sample})")

    if save:
        # График сэмплирования
        fig, ax = plt.subplots(figsize=(14, 8))
        mask_display = t_full <= T_display
        ax.plot(t_full[mask_display], y_full[mask_display], 'b-', linewidth=2.0,
                label='Исходный сигнал', alpha=0.7)
        mask_samples_display = t_samples <= T_display
        ax.plot(t_samples[mask_samples_display], samples[mask_samples_display],
                'ro', markersize=8, label=f'Отсчёты (Δt={dt_sample} с)')
        ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
        ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
        ax.set_xlim(0, T_display)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Дискретизация сигнала $y_2(t)$ (Δt = {dt_sample} с, fs/2 = {nyquist:.2f} Гц)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y2_sampling_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # График восстановления (исходный + сэмплы + восстановленный)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(t_rec, y_original_display, 'b-', linewidth=2.0,
                label='Исходный сигнал', alpha=0.7)
        ax.plot(t_rec, recovered, 'r--', linewidth=2.0,
                label='Восстановленный сигнал (Котельников)', alpha=0.7)
        mask_samples_display = t_samples <= T_display
        ax.plot(t_samples[mask_samples_display], samples[mask_samples_display],
                'ro', markersize=6, label=f'Отсчёты (Δt={dt_sample} с)', alpha=0.8)
        ax.set_xlabel('Время $t$, с', fontsize=20, fontweight='bold')
        ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
        ax.set_xlim(0, T_display)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Восстановление сигнала $y_2(t)$\n'
                     f'$\\Delta t = {dt_sample}$ с, $f_s/2 = {nyquist:.2f}$ Гц, '
                     f'MSE = {results["mse"]:.2e}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y2_recovered_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # График спектров
        fig, ax = plt.subplots(figsize=(14, 8))
        nu_lim = 15
        mask_cont = np.abs(results['nu_cont']) <= nu_lim
        mask_rec = np.abs(results['nu_rec']) <= nu_lim
        ax.plot(results['nu_cont'][mask_cont], np.abs(results['f_cont'][mask_cont]),
                'b-', linewidth=2.0, label='Спектр исходного сигнала', alpha=0.7)
        ax.plot(results['nu_rec'][mask_rec], np.abs(results['f_rec'][mask_rec]),
                'r--', linewidth=2.0, label='Спектр восстановленного сигнала', alpha=0.7)
        ax.axvline(x=nyquist, color='red', linestyle=':', alpha=0.7, linewidth=2,
                   label=f'Частота Найквиста = {nyquist:.2f} Гц')
        ax.axvline(x=-nyquist, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax.axvline(x=8, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='8 Гц')
        ax.axvline(x=-8, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Частота $\\nu$, Гц', fontsize=20, fontweight='bold')
        ax.set_ylabel('$|\\hat{y}(\\nu)|$', fontsize=20, fontweight='bold')
        ax.set_xlim(-nu_lim, nu_lim)
        ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Спектры сигнала $y_2(t)$ (Δt = {dt_sample} с)',
                     fontsize=20, fontweight='bold')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"y2_spectra_dt{dt_sample}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_three_spectra_comparison(results, signal_name='y2', dt_sample=dt_sample, save=True)
    return results


# ========== СВОДНЫЕ ГРАФИКИ ==========

def plot_mse_summary_y1(y1_results, save=True):
    """Сводный график MSE для y1(t) — пункт 2.2(б,д)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    dt_values = [r['dt'] for r in y1_results]
    mse_values = [r['mse'] for r in y1_results]
    ax.semilogy(dt_values, mse_values, 'bo-', linewidth=2.5, markersize=10, alpha=0.8)
    ax.set_xlabel('Шаг дискретизации Δt, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('MSE восстановления', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title('Сигнал $y_1(t)$ (5 Гц + 12 Гц)', fontsize=20, fontweight='bold')
    ax.axvline(x=1/(2*12), color='green', linestyle='--', alpha=0.7, linewidth=2.5,
                label=f'Критический Δt = {1/(24):.4f} с (fs/2 = 12 Гц)')
    ax.legend(fontsize=16, loc='best')
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "mse_summary_y1.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mse_summary_y2(y2_results, save=True):
    """Сводный график MSE для y2(t) — пункт 2.2(б,д)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    dt_values = [r['dt'] for r in y2_results]
    mse_values = [r['mse'] for r in y2_results]
    ax.semilogy(dt_values, mse_values, 'bo-', linewidth=2.5, markersize=10, alpha=0.8)
    ax.set_xlabel('Шаг дискретизации Δt, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('MSE восстановления', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title('Сигнал $y_2(t) = \\operatorname{sinc}(16\\pi t)$', fontsize=20, fontweight='bold')
    ax.axvline(x=1/(2*8), color='green', linestyle='--', alpha=0.7, linewidth=2.5,
                label=f'Критический Δt = {1/16:.4f} с (fs/2 = 8 Гц)')
    ax.legend(fontsize=16, loc='best')
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "mse_summary_y2.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ========== ОСНОВНАЯ ФУНКЦИЯ ==========

def main():
    print("\nПостроение базовых графиков...")
    plot_y1_continuous(save=True)
    plot_y1_spectrum(save=True)
    plot_y2_continuous(save=True)
    plot_y2_spectrum(save=True)
    print("  Базовые графики сохранены")

    print("\n" + "="*60)
    print("ИССЛЕДОВАНИЕ СИГНАЛА y1(t)")
    print("="*60)

    dt_values_y1 = [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1]
    y1_results = []
    for dt in dt_values_y1:
        results = experiment_y1(dt, T_sampling=10.0, T_display=5.0, save=True)
        results['dt'] = dt
        y1_results.append(results)

    print("\n" + "="*60)
    print("ИССЛЕДОВАНИЕ СИГНАЛА y2(t)")
    print("="*60)

    dt_values_y2 = [0.02, 0.04, 0.06, 0.0625, 0.07, 0.08, 0.1]
    y2_results = []
    for dt in dt_values_y2:
        results = experiment_y2(dt, T_sampling=10.0, T_display=5.0, save=True)
        results['dt'] = dt
        y2_results.append(results)

    plot_mse_summary_y1(y1_results, save=True)
    plot_mse_summary_y2(y2_results, save=True)

    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*60)

    print("\nСигнал y1(t) (5 Гц + 12 Гц):")
    print(f"{'Δt, с':<10} {'fs/2, Гц':<12} {'MSE':<15} {'Статус':<25}")
    print("-"*62)
    for r in y1_results:
        nyquist = 1/(2*r['dt'])
        status = "Условие Найквиста выполнено" if nyquist > 12 else "Нарушено (алиасинг)"
        print(f"{r['dt']:<10.4f} {nyquist:<12.2f} {r['mse']:<15.2e} {status:<25}")

    print("\nСигнал y2(t) (8 Гц):")
    print(f"{'Δt, с':<10} {'fs/2, Гц':<12} {'MSE':<15} {'Статус':<25}")
    print("-"*62)
    for r in y2_results:
        nyquist = 1/(2*r['dt'])
        status = "Условие Найквиста выполнено" if nyquist > 8 else "Нарушено (алиасинг)"
        print(f"{r['dt']:<10.4f} {nyquist:<12.2f} {r['mse']:<15.2e} {status:<25}")



    print(f"\nРезультаты сохранены в: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()