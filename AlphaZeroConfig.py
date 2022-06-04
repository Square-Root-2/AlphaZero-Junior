import tensorflow as tf


class AlphaZeroConfig(object):

    class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            if step < 100e3:
                return 2e-1
            if step < 300e3:
                return 2e-2
            if step < 500e3:
                return 2e-3
            return 2e-4

    def __init__(self):
        ### Self-Play
        self.num_actors = 500

        self.num_sampling_moves = 30
        self.max_moves = 512
        self.num_simulations = 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096
        self.pseudobatch_size = 1024

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.learning_rate_schedule = AlphaZeroConfig.CustomLearningRateSchedule()
