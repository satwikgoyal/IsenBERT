
# Interface for Strategy pattern
class algorithm:

    def train(self, env: object, N: int, parameters: set):

        """ Trains the environment to find the best parameters"""
        """ Should return two lists and parameters"""
        """ 1. list where it contains the current number of actions taken """
        """ 2. list where it contains the number of steps to the goal state per episode"""
        """ 3. parameters in order to play the game (parameters to get the best policy)"""
        """ parameters can be more than one variable"""

        pass

    def plot_graphs(self, env: object, eps: int, runs: int, parameters: set):

        """ Runs the train algorithm multiple times"""
        """ Then take the mean of the two lists"""
        """ Plot two graphs"""
        """ 1. Number of actions vs Number of Episodes"""
        """ 2. Number of Episodes vs Number of steps to goal"""

        pass

    def run_game(self, final_parameters):

        """ given the best parameters run one episode of the environment"""

        pass