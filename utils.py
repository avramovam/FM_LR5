"""
Вспомогательные функции
"""

import numpy as np
from typing import Tuple


def create_time_grid(T: float, dt: float, symmetric: bool = True) -> np.ndarray:
    """
    Создание временной сетки

    Параметры:
    -----------
    T : float
        Общая длительность
    dt : float
        Шаг дискретизации
    symmetric : bool
        Если True, сетка от -T/2 до T/2, иначе от 0 до T

    Возвращает:
    ------------
    np.ndarray : Временная сетка
    """
    if symmetric:
        return np.arange(-T / 2, T / 2, dt)
    else:
        return np.arange(0, T, dt)


def calculate_frequency_grid(N: int, dt: float) -> Tuple[np.ndarray, float, float]:
    """
    Расчет частотной сетки для FFT

    Параметры:
    -----------
    N : int
        Количество точек
    dt : float
        Шаг по времени

    Возвращает:
    ------------
    tuple: (частоты, максимальная частота, шаг по частоте)
    """
    fs = 1 / dt
    nu = np.fft.fftshift(np.fft.fftfreq(N, dt))
    return nu, fs / 2, 1 / (N * dt)


def mse(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Среднеквадратичная ошибка между двумя сигналами
    """
    return np.mean((signal1 - signal2) ** 2)


def psnr(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Пиковое отношение сигнал-шум (PSNR)
    """
    mse_val = mse(signal1, signal2)
    if mse_val == 0:
        return float('inf')
    max_val = max(np.max(np.abs(signal1)), np.max(np.abs(signal2)))
    return 20 * np.log10(max_val / np.sqrt(mse_val))


def find_bandwidth(fourier: np.ndarray, nu: np.ndarray, threshold: float = 0.01) -> float:
    """
    Нахождение ширины спектра по порогу

    Параметры:
    -----------
    fourier : np.ndarray
        Фурье-образ
    nu : np.ndarray
        Частотная сетка
    threshold : float
        Порог относительно максимума

    Возвращает:
    ------------
    float : Ширина спектра
    """
    magnitude = np.abs(fourier)
    max_mag = np.max(magnitude)

    # Индексы, где спектр выше порога
    above_threshold = magnitude > threshold * max_mag

    if not np.any(above_threshold):
        return 0.0

    # Ширина спектра
    freq_above = nu[above_threshold]
    return freq_above[-1] - freq_above[0]