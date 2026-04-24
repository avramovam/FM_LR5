"""
Определение сигналов и их аналитических Фурье-образов
"""

import numpy as np


def rect_pulse(t: np.ndarray) -> np.ndarray:
    """
    Прямоугольная функция Π(t)
    Π(t) = 1 для |t| <= 1/2, иначе 0
    """
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def rect_pulse_analytic_fourier(nu: np.ndarray) -> np.ndarray:
    """
    Аналитический Фурье-образ прямоугольной функции Π(t)

    Теоретический вывод:
    Π̂(ν) = ∫_{-1/2}^{1/2} e^{-2πiνt} dt = [e^{-2πiνt} / (-2πiν)]_{-1/2}^{1/2}
          = (e^{-πiν} - e^{πiν}) / (-2πiν) = (e^{πiν} - e^{-πiν}) / (2πiν)
          = sin(πν) / (πν) = sinc(ν)

    где sinc(ν) = sin(πν)/(πν)
    """
    # Защита от деления на ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sin(np.pi * nu) / (np.pi * nu)
        # В точке ν = 0 предел равен 1
        result[np.abs(nu) < 1e-12] = 1.0
    return result


def harmonic_signal(t: np.ndarray, a: float, omega: float, phi: float) -> np.ndarray:
    """
    Гармонический сигнал: a * sin(ωt + φ)
    """
    return a * np.sin(omega * t + phi)


def two_harmonics_signal(t: np.ndarray, a1: float, omega1: float, phi1: float,
                         a2: float, omega2: float, phi2: float) -> np.ndarray:
    """
    Сумма двух гармоник: a1*sin(ω1*t+φ1) + a2*sin(ω2*t+φ2)
    """
    return harmonic_signal(t, a1, omega1, phi1) + harmonic_signal(t, a2, omega2, phi2)


# Функции для второго задания (будет позже)
def y1_signal(t: np.ndarray, a1=1.0, a2=0.5, omega1=2 * np.pi * 5, omega2=2 * np.pi * 12,
              phi1=0, phi2=0) -> np.ndarray:
    """
    y1(t) = a1*sin(ω1*t+φ1) + a2*sin(ω2*t+φ2)
    Частоты: 5 Гц и 12 Гц
    """
    return harmonic_signal(t, a1, omega1, phi1) + harmonic_signal(t, a2, omega2, phi2)


def y2_signal(t: np.ndarray, B=5.0) -> np.ndarray:
    """
    y2(t) = sinc(2*B*t) согласно заданию.
    Полоса спектра строго ограничена частотой B Гц.
    """
    return np.sinc(2 * B * t)