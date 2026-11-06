class TrackedFace:
    # also has velocity
    def __init__(self, pos, size, scores):
        self.vel = 0
        self.pos = pos # vec2
        self.size = size # avstand mellom opposite hjørner
        self.scores = scores # dict med person -> score

    def process_new_data(self, pos, size, scores):
        self.vel = self.pos - pos
        self.pos = pos
        self.size = size
        self.scores = scores # will interpolate based on data, instead of just replacing. this is to prevent flickering and losing track from just 1/a few failed frame