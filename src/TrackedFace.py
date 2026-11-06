import numpy as np


class TrackedFace:
    # also has velocity
    def __init__(self, pos: np.ndarray, size: np.ndarray, scores):
        self.vel = 0
        self.pos = pos # vec2
        self.size = size # vec2
        self.scores = scores # dict med person -> score

    def process_new_data(self, pos, size, scores):
        """called per frame, new data is inputted and processed"""
        self.vel = self.pos - pos
        self.pos = pos
        self.size = size
        self.scores = scores # will interpolate based on data, instead of just replacing. this is to prevent flickering and losing track from just 1/a few failed frame

    @property
    def diagonal(self) -> float:
        """distance between opposite corners"""
        return np.linalg.norm(self.size) # pytagoras

    @property
    def box(self) -> np.ndarray:
        """(x, y, w, h) with top left corner"""
        top_left = self.pos - self.size / 2
        return np.array([top_left[0], top_left[1], self.size[0], self.size[1]])

    @property
    def best_match(self) -> tuple[str, float]:
        """person with the highest score. (name, score) ?: ("Unknown", 0.0)"""
        return max(self.scores.items(), key=lambda item: item[1], default=("Unknown", 0.0))

    @property
    def certainty(self) -> float:
        """ diff between first and second score, if exists. if there isnt any scores then 0. if just 1 score, then pretend second is 0.2"""
        top = sorted(self.scores.values(), reverse=True)
        if len(top) == 0:
            return 0.0
        if len(top) == 1:
            return top[0] - 0.2
        return top[0] - top[1]
