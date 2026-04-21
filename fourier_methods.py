"""
Различные методы вычисления преобразования Фурье
"""

import numpy as np
from scipy.integrate import trapezoid  # Используем trapezoid вместо trapz

class FourierMethods:
    """Класс, реализующий различные методы преобразования Фурье"""

    def __init__(self, t: np.ndarray, signal: np.ndarray):
        """
        Инициализация

        Параметры:
        -----------
        t : np.ndarray
            Временная сетка
        signal : np.ndarray
            Значения сигнала на временной сетке
        """
        self.t = t
        self.signal = signal
        self.dt = t[1] - t[0] if len(t) > 1 else 0.01
        self.T = t[-1] - t[0] if len(t) > 1 else 1.0
        self.N = len(t)

        # Частотная сетка для fft (от -fs/2 до fs/2)
        self.fs = 1 / self.dt if self.dt > 0 else 100.0
        self.nu_fft = np.fft.fftshift(np.fft.fftfreq(self.N, self.dt))

    def method_trapz(self, nu: np.ndarray) -> np.ndarray:
        """
        Метод 1: Численное интегрирование через trapezoid

        Параметры:
        -----------
        nu : np.ndarray
            Частотная сетка для вычисления образа

        Возвращает:
        ------------
        np.ndarray : Фурье-образ на частотах nu
        """
        fourier = np.zeros(len(nu), dtype=complex)

        for i, freq in enumerate(nu):
            integrand = self.signal * np.exp(-2j * np.pi * freq * self.t)
            fourier[i] = trapezoid(integrand, self.t)

        return fourier

    def method_naive_fft(self) -> tuple:
        """
        Метод 2: Наивное использование fft (неунитарное)

        Возвращает:
        ------------
        tuple: (частоты, Фурье-образ, восстановленный сигнал)
        """
        # Прямое преобразование
        f_hat_naive = np.fft.fftshift(np.fft.fft(self.signal))

        # Обратное преобразование
        recovered_naive = np.fft.ifft(np.fft.ifftshift(f_hat_naive)).real

        return self.nu_fft, f_hat_naive, recovered_naive

    def method_unitary_fft(self) -> tuple:
        """
        Метод 2b: Унитарное использование fft (правильное масштабирование)

        Унитарное преобразование: коэффициент 1/sqrt(N)

        Возвращает:
        ------------
        tuple: (частоты, Фурье-образ, восстановленный сигнал)
        """
        N = self.N

        # Прямое унитарное преобразование
        f_hat_unitary = np.fft.fftshift(np.fft.fft(self.signal) / np.sqrt(N))

        # Обратное унитарное преобразование
        recovered_unitary = np.fft.ifft(np.fft.ifftshift(f_hat_unitary) * np.sqrt(N)).real

        return self.nu_fft, f_hat_unitary, recovered_unitary

    def method_smart_fft(self) -> tuple:
        """
        Метод 3: Умное использование fft (аппроксимация непрерывного ПФ)

        Теоретический вывод:
        Непрерывное ПФ: F(ν) = ∫_{-T/2}^{T/2} f(t) e^{-2πiνt} dt

        Аппроксимация суммой Римана:
        F(ν_m) ≈ Δt * ∑_{n=0}^{N-1} f(t_n) e^{-2πiν_m t_n}

        Связь с DFT: F(ν_m) = Δt * DFT{f}[k] для определенных ν_m

        Получаем поправочный коэффициент: c = dt

        Возвращает:
        ------------
        tuple: (частоты, Фурье-образ, восстановленный сигнал)
        """
        dt = self.dt

        # Прямое преобразование с поправочным коэффициентом dt
        f_hat_smart = dt * np.fft.fftshift(np.fft.fft(self.signal))

        # Обратное преобразование: делим на dt
        recovered_smart = np.fft.ifft(np.fft.ifftshift(f_hat_smart / dt)).real

        return self.nu_fft, f_hat_smart, recovered_smart

    def method_smart_fft_with_window(self, window_func=None) -> tuple:
        """
        Метод 3b: Умное использование fft с оконной функцией

        Параметры:
        -----------
        window_func : callable, optional
            Оконная функция (например, np.hanning)
        """
        dt = self.dt

        if window_func is not None:
            window = window_func(self.N)
            signal_windowed = self.signal * window
        else:
            signal_windowed = self.signal

        # Прямое преобразование
        f_hat_smart = dt * np.fft.fftshift(np.fft.fft(signal_windowed))

        # Обратное преобразование
        recovered_smart = np.fft.ifft(np.fft.ifftshift(f_hat_smart / dt)).real

        return self.nu_fft, f_hat_smart, recovered_smart

    def analyze_alias(self, signal_bandwidth: float = 10.0) -> dict:
        """
        Анализ алиасинга для текущей конфигурации

        Параметры:
        -----------
        signal_bandwidth : float
            Оценка полосы сигнала (для rect pulse ~10 Гц)

        Возвращает:
        ------------
        dict : Информация об алиасинге
        """
        fs = self.fs
        nyquist = fs / 2

        return {
            'fs': fs,
            'nyquist_frequency': nyquist,
            'signal_bandwidth': signal_bandwidth,
            'aliasing_present': signal_bandwidth > nyquist,
            'aliasing_ratio': signal_bandwidth / nyquist if nyquist > 0 else np.inf
        }


def compute_cm_coefficient(dt: float, T: float) -> float:
    """
    Вычисление поправочного коэффициента cm для аппроксимации непрерывного ПФ

    Теоретический вывод:
    F(ν) ≈ ∑_{n} f(t_n) e^{-2πiν t_n} Δt

    При использовании fft, частоты имеют вид: ν_k = k / T (k = -N/2,..., N/2-1)

    Коэффициент cm = Δt

    Возвращает:
    ------------
    float : Поправочный коэффициент cm
    """
    return dt