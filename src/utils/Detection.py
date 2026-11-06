from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    pos: np.ndarray
    size: float
    scores: dict[str, float] # map<>