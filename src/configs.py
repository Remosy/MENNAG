
class Configs():

    def __init__(self, dict):
        self.max_depth = dict['max_depth']
        self.input_size = dict['input_size']
        self.output_size = dict['output_size']
        self.feedforward = dict['feedforward']
        self.forward_prob = dict['forward_prob']
        self.weight_mean = dict['weight_mean']
        self.weight_std = dict['weight_std']
        self.perturb_std = dict['perturb_std']
