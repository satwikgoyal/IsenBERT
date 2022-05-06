from abc import ABC, abstractmethod
# abstract class for template method
class algorithm:

    @abstractmethod
    def train(self, env: object, N: int, parameters: tuple):

        """ Trains the environment to find the best parameters"""
        """ Should return list and parameters"""
        """ 1. list where it contains the number of steps to the goal state per episode"""
        """ 2. parameters in order to play the game (parameters to get the best policy)"""
        """ parameters can be more than one variable"""

        pass

    @abstractmethod
    def plot_graphs(self, env: object, eps: int, runs: int, parameters: tuple):

        """ Runs the train algorithm multiple times"""
        """ Then take the mean of lists"""
        """ Plot graph"""
        """ Number of Episodes vs Number of steps to goal"""

        pass

    @abstractmethod
    def run_game(self, final_parameters):

        """ given the best parameters run one episode of the environment"""

        pass