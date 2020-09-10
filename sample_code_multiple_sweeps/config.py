import numpy as np
import pickle
import os

from targets import target_distribution_gen_all

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        ## Note that here I generate target distributions with a custom target_distribution_gen_all fn,
        ## however, in general, you can input any np.array as self.target_distributions with shape (number of distributions, size of a single distribution)

        # Define target distributions to sweep through.
        ## Set up custom target distribution generator function
        self.target_distr_name = "Fritz-visibility" # check targets.py for possible names
        self.param_range = np.linspace(0.5, 1, 5)
        self.which_param = 2 # Specifies whether we want to sweep through param1, a distribution parameter (not relevant sometimes, e.g. for elegant distr.), or param2, the noise parameter.
        self.other_param = 1 # Fixes other parameter, which we don't sweep through.

        ## Set target distributions and their ids
        self.target_distributions = target_distribution_gen_all(self.target_distr_name,  self.param_range, self.which_param, self.other_param)
        self.target_ids = self.param_range
        self.a_outputsize = 4 # Number of outputs for Alice
        self.b_outputsize = 4 # Number of outputs for Bob
        self.c_outputsize = 4 # Number of outputs for Charlie

        # Neural network parameters
        self.latin_depth = 5
        self.latin_width = 30

        # Training procedure parameters
        self.batch_size = 8000
        self.no_of_batches = 10000 # How many batches to go through during training.
        self.weight_init_scaling = 2.#10. # default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta'
        self.lr = 0.5
        self.decay = 0.001
        self.momentum = 0.25
        self.loss = 'kl'

        # Initialize some variables
        self.euclidean_distances = np.ones_like(self.target_ids)*100
        self.distances = np.ones_like(self.target_ids)*100
        self.distributions = np.ones_like(self.target_distributions)/(self.target_distributions.shape[0]) # learned distributions
        self.set_starting_points(fresh_start = True) # Don't change fresh start here. If you want to change it, change it in your main python script (required due to dependencies).
        self.sweep_id = 0 # id of sweep. Only used if multiple sweeps are done.

        # Neural network parameters that I don't change much
        self.no_of_validation_batches = 100 # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.change_batch_size(self.batch_size) #updates test batch size
        self.greek_depth = 0 # set to 0 if trivial neural networks at sources
        self.greek_width = 1
        self.activ = 'relu' # activation for most of NN
        self.activ2 = 'softmax' # activation for last dense layer
        self.kernel_reg = None

    def change_p_target(self,id):
        self.p_target = self.target_distributions[id,:] # current target distribution
        self.p_id = self.target_ids[id] # current target id
        self.y_true = np.array([self.p_target for _ in range(self.batch_size)]) # keras technicality that in supervised training y_true should have batch_size dimension as well
        self.savebestpath = './saved_models/best_'+str(id).zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'.hdf5'
        self.start_from = self.start_from_array[id]

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size
        self.batch_size_test = int(self.no_of_validation_batches*self.batch_size) # in case we update batch_size we should also update the test batch size

    def save(self,name):
        with open('./saved_configs/'+name, 'wb') as f:
            pickle.dump(self, f)

    def set_starting_points(self,broadness_left=0, broadness_right=0, fresh_start=False):
        """ Sets where to continue learning from by looking at neighbors' models in a range of [broadness_left, broadness_right].
        Useful if you want to smooth out distance function. """
        # If fresh_start, then put None for start_from, which results in not loading a model but doing random initialization fo weights.
        if fresh_start:
            self.start_from_array = np.array([None for _ in range(self.target_distributions.shape[0])])
        else:
            # Set starting ids to be just their own ids by default.
            starting_ids = np.arange(self.target_distributions.shape[0])

            # Change this default if either broadness_left or broadness_right is nonzero
            if broadness_left!=0 or broadness_right!=0:
                from utils_nn import np_distance # have to import here due to dependency issues
                for i in range(self.target_distributions.shape[0]):
                    # cross_distances_i measures the distance between p_target(i) and p_machine (j) for each j. We then choose the best model for starting from i, in a range of [broadness_left, broadness_right]
                    cross_distances_i = np.array([np_distance(self.target_distributions[i,:], self.distributions[j,:]) for j in range(self.target_distributions.shape[0]) ])
                    # in temp we just removes those distances which we shouldn't consider (by setting them to a large value, i.e. to 1)
                    temp = np.ones_like(cross_distances_i)
                    temp[max(0,i-broadness_left):min(self.target_distributions.shape[0],i+1+broadness_right)] = cross_distances_i[max(0,i-broadness_left):min(self.target_distributions.shape[0],i+1+broadness_right)]
                    # starting index of i is where temp is smallest.
                    start_from_index = np.argmin(temp)
                    # (This if is probably redundant but I left it here as a sanity check.)
                    if start_from_index<i-broadness_left or start_from_index>i+broadness_right:
                        start_from_index = i
                    starting_ids[i] = start_from_index
                    print("\nStarting from index {} instead of {}.".format(start_from_index,i))
            else:
                print("Starting points have not been changed.")
            # Generate model names from starting_ids.
            self.start_from_array = np.array(['./saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(self.target_distributions.shape[0]))))+'.hdf5' for i in starting_ids])

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
