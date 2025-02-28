from deap import base, creator, tools, algorithms
import numpy as np
from rankSelection import rank_selection
def configure_ga(lambda_values):
    number_of_variables = len(lambda_values)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -10000, 10000) 
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                  toolbox.attr_float, number_of_variables)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selBest)
    toolbox.register("select", tools.selTournament,tournsize=3)

    return toolbox, number_of_variables

# Example usage:
# lambda_values = [1, 2, 3, 4]
# toolbox, number_of_variables = configure_ga(lambda_values)

