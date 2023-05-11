import numpy as np
import random
from encoder import *
from ga import *

'''
Mutation operator
'''

def cp_mutation(parameters):
    if parameters["COMP_MUT"] == 1:
        if 0 < parameters["COMP_MUT_PERC"] < 1:
            if parameters["MIN_POS"] < parameters["COMP_MUT_STD"] < parameters["MAX_POS"]:
                return 1
            else:
                errorWarning(3.3, "algoConfig.ini", "COMP_MUT_STD", "The percentage parameter of the Elitism component Elitism should be in the interval ]0, 1[")
                return 0
        else:
            errorWarning(3.2, "algoConfig.ini", "COMP_MUT_PERC", "The percentage parameter of the Elitism component Elitism should be in the interval ]0, 1[")
            return 0
    elif parameters["COMP_MUT"] != 0:
        errorWarning(3.1, "algoConfig.ini", "COMP_MUT", "Component Elitism should be 0 or 1")
        return 0


def mutation(pop, parameters):
    for i in range(int(parameters["COMP_ELI_PERC"]*parameters["POPSIZE"]), parameters["POPSIZE"]):
        if parameters["ENCODER"]:
            for j in range(parameters["INDSIZE"]):
                if random.random() < parameters["COMP_MUT_PERC"]:
                    pop.ind[i] = encoder(pop.ind[i], parameters)
                    pop.ind[i]["pos"][j] = 1 - pop.ind[i]["pos"][j]
                    pop.ind[i] = decoder(pop.ind[i], parameters)
        else:
            for d in range(parameters["NDIM"]):
                if random.random() < parameters["COMP_MUT_PERC"]:
                    pop.ind[i]["pos"][d] += np.random.normal(loc = 0.0, scale = parameters["COMP_MUT_STD"])
    return pop
