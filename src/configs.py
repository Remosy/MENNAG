
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
        self.cross_rate = dict['cross_rate']
        self.conn_relocate_rate = dict['conn_relocate_rate']
        self.weight_perturb_rate = dict['weight_perturb_rate']
        self.weight_reset_rate = dict['weight_reset_rate']
        self.insertion_rate = dict['insertion_rate']
        self.deletion_rate = dict['deletion_rate']
        self.random_tree_rate = dict['random_tree_rate']
