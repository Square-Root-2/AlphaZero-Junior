import AlphaZeroConfig

import numpy


class ReplayBuffer(object):

    def __init__(self, config: AlphaZeroConfig.AlphaZeroConfig):
        self.window_size = config.window_size
        self.pseudobatch_size = config.pseudobatch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_pseudobatch(self):
        # Sample uniformly across positions.
        import tensorflow as tf
        move_sum = float(sum(len(g.history) - 1 for g in self.buffer))
        games = numpy.random.choice(
            self.buffer,
            size=self.pseudobatch_size,
            p=[(len(g.history) - 1) / move_sum for g in self.buffer])
        game_pos = [(g, numpy.random.randint(len(g.history) - 1)) for g in games]
        images = [g.make_image(i) for (g, i) in game_pos]
        target_values = []
        target_policies = []
        for g, i in game_pos:
            target = g.make_target(i)
            target_values.append(target[0])
            target_policies.append(target[1])
        return tf.stack(images), tf.stack(target_values), tf.stack(target_policies)

    def is_empty(self):
        return len(self.buffer) == 0
