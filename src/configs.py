
class Configs():

    def __init__(self, dict):
        self.input_size = dict['input_size']
        self.output_size = dict['output_size']
        self.feedforward = dict['feedforward']
        self.weight_mean = dict['weight_mean']
        self.weight_std = dict['weight_std']
        self.perturb_std = dict['perturb_std']
