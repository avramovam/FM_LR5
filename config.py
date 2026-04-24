from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    name: str
    T: float
    dt: float
    description: str

    @property
    def N(self) -> int:
        if self.dt <= 0:
            return 1000
        return int(self.T / self.dt)

    @property
    def V(self) -> float:
        if self.dt <= 0:
            return 100.0
        return 1 / self.dt

    @property
    def dv(self) -> float:
        if self.T <= 0:
            return 0.1
        return 1 / self.T

GOOD_CONFIGS = [
    ExperimentConfig(name="Хороший случай 1", T=32.0, dt=0.01, description="Достаточно большой T и малый dt"),
    ExperimentConfig(name="Хороший случай 2", T=64.0, dt=0.005, description="Большой T и очень малый dt"),
]

BAD_CONFIGS = [
    ExperimentConfig(name="Плохой случай 1 - малый T", T=4.0, dt=0.01, description="Слишком малый T - низкое частотное разрешение"),
    ExperimentConfig(name="Плохой случай 2 - большой dt", T=32.0, dt=0.5, description="Слишком большой dt - алиасинг"),
    ExperimentConfig(name="Плохой случай 3 - оба плохих", T=4.0, dt=0.5, description="Малый T и большой dt"),
]

ALL_CONFIGS = GOOD_CONFIGS + BAD_CONFIGS

COMPARISON_CONFIG = ExperimentConfig(name="Сравнительный эксперимент", T=32.0, dt=0.01, description="Базовый конфиг для сравнения методов")