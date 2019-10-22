import numpy as np
import pickle
import os

from targets import target_distribution_gen_all

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        ## Note that here I generate target distributions with a custom target_distribution_gen_all fn,
        ## however, in general, you can input any np.array as self.target_distributions with shape (number of distributions, size of a single distribution)

        # Define target distributions to loop through.
        ## Set up custom target distribution generator function
        self.target_distr_name = "Fritz-visibility" # check targets.py for possible names
        self.param_range = np.linspace(0.5, 1, 10)
        self.which_param = 2 # Specifies whether we want to loop through param1, a distribution parameter (not relevant sometimes, e.g. for elegant distr.), or param2, the noise parameter.
        self.other_param = 1 # Fixes other parameter, which we don't loop through.

        ## Set target distributions and their ids
        self.target_distributions = target_distribution_gen_all(self.target_distr_name,  self.param_range, self.which_param, self.other_param)
        self.target_ids = self.param_range
        self.a_outputsize = 4 # Number of output bits for Alice
        self.b_outputsize = 4 # Number of output bits for Bob
        self.c_outputsize = 4 # Number of output bits for Charlie

        # Neural network parameters
        self.latin_depth = 3
        self.latin_width = 16

        # Training procedure parameters
        self.batch_size = 6000
        self.no_of_batches = 8000 # How many batches to go through during training.
        self.weight_init_scaling = 2.#10. # default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta'
        self.lr = 2
        self.decay = 0.001
        self.momentum = 0.25

        # Initialize result variables
        self.rms_distances = np.ones_like(self.target_ids)*100
        self.distances = np.ones_like(self.target_ids)*100
        self.distributions = np.ones_like(self.target_distributions)/(self.target_distributions.shape[0]) # learned distributions

        # Neural network parameters that I don't change much
        self.no_of_validation_batches = 100 # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.change_batch_size(self.batch_size) #updates test batch size
        self.greek_depth = 0 # set to 0 if trivial neural networks at sources
        self.greek_width = 1
        self.activ = 'tanh' # activation for most of NN
        self.activ2 = 'softmax' # activation for last dense layer
        self.kernel_reg = None

        # If you want to start from somewhere:
        self.start_from = None

    def change_p_target(self,id):
        self.p_target = self.target_distributions[id,:] # current target distribution
        self.p_id = self.target_ids[id] # current target id
        self.y_true = np.array([self.p_target for _ in range(self.batch_size)]) # keras technicality that in supervised training y_true should have batch_size dimension as well
        self.savebestpath = './saved_models/best_'+str(id).zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'.hdf5'

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size
        self.batch_size_test = int(self.no_of_validation_batches*self.batch_size) # in case we update batch_size we should also update the test batch size

    def save(self,name):
        with open('./saved_configs/'+name, 'wb') as f:
            pickle.dump(self, f)

def load_config(name):
    with open('./saved_configs/'+name, 'rb') as f:
        temp = pickle.load(f)
    return temp

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()
    pnn.save('initial_pnn')
