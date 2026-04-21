"""
Конфигурационный файл с параметрами экспериментов
"""

from dataclasses import dataclass
from typing import List

@dataclass
class ExperimentConfig:
    """Конфигурация для одного эксперимента"""
    name: str                    # Название эксперимента
    T: float                     # Промежуток времени [-T/2, T/2]
    dt: float                    # Шаг дискретизации по времени
    description: str             # Описание

    @property
    def N(self) -> int:
        """Количество точек дискретизации"""
        if self.dt <= 0:
            return 1000
        return int(self.T / self.dt)

    @property
    def V(self) -> float:
        """Промежуток по частоте (теоретический)"""
        if self.dt <= 0:
            return 100.0
        return 1 / self.dt

    @property
    def dv(self) -> float:
        """Шаг по частоте"""
        if self.T <= 0:
            return 0.1
        return 1 / self.T

# Предопределенные конфигурации для исследования
# Хорошие случаи
GOOD_CONFIGS = [
    ExperimentConfig(
        name="Хороший случай 1",
        T=32.0,
        dt=0.01,
        description="Достаточно большой T и малый dt"
    ),
    ExperimentConfig(
        name="Хороший случай 2",
        T=64.0,
        dt=0.005,
        description="Большой T и очень малый dt"
    ),
]

# Плохие случаи (для демонстрации проблем)
BAD_CONFIGS = [
    ExperimentConfig(
        name="Плохой случай 1 - малый T",
        T=4.0,
        dt=0.01,
        description="Слишком малый T - низкое частотное разрешение"
    ),
    ExperimentConfig(
        name="Плохой случай 2 - большой dt",
        T=32.0,
        dt=0.5,
        description="Слишком большой dt - алиасинг"
    ),
    ExperimentConfig(
        name="Плохой случай 3 - оба плохих",
        T=4.0,
        dt=0.5,
        description="Малый T и большой dt - низкое разрешение + алиасинг"
    ),
]

# Все конфигурации для полного исследования
ALL_CONFIGS = GOOD_CONFIGS + BAD_CONFIGS

# Параметры для сравнения методов
COMPARISON_CONFIG = ExperimentConfig(
    name="Сравнительный эксперимент",
    T=32.0,
    dt=0.01,
    description="Базовый конфиг для сравнения методов"
)