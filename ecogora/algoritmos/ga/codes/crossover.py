import random
from encoder import *
from ga import *

def cp_crossover(parameters):
    if parameters["COMP_CROSS"] == 1:
        if 0 < parameters["COMP_CROSS_PERC"] < 1:
            return 1
        else:
            errorWarning(2.2, "algoConfig.ini", "COMP_CROSS_PERC", "The percentage parameter of the Elitism component Elitism should be in the interval ]0, 1[")
            return 0
    elif parameters["COMP_CROSS"] != 0:
        errorWarning(2.1, "algoConfig.ini", "COMP_CROSS", "Component Elitism should be 0 or 1")
        return 0


def condition(individual):
    return individual["fit"]
def tournament(pop, parameters):
    l = int(len(pop.ind)/2)
    c = []
    for _ in range(3):
        c.append(random.choice(pop.ind[:int(parameters["COMP_CROSS_PERC"]*parameters["POPSIZE"])]))

    choosed = min(c, key=condition)
    return choosed

'''
Crossover operator
'''
def crossover(pop, newPop, parameters):
    for i in range(1, int((parameters["POPSIZE"] - parameters["COMP_ELI_PERC"]*parameters["POPSIZE"])), 2):
        parent1 = tournament(pop, parameters)
        parent2 = tournament(pop, parameters)
        child1 = parent1.copy()
        child2 = parent2.copy()

        if parameters["ENCODER"]:
            parent1 = encoder(parent1, parameters)
            parent2 = encoder(parent2, parameters)
            cutPoint = random.choice(range(len(parent1["pos"])))
            child1["pos"], child2["pos"]  = parent1["pos"][:cutPoint] + parent2["pos"][cutPoint:], \
                                            parent2["pos"][:cutPoint] + parent1["pos"][cutPoint:]
            child1 = decoder(child1, parameters)
            child2 = decoder(child2, parameters)
        else:
            cutPoint = random.choice(range(1, parameters["NDIM"]))
            child1["pos"], child2["pos"]  = parent1["pos"][:cutPoint] + parent2["pos"][cutPoint:], \
                                            parent2["pos"][:cutPoint] + parent1["pos"][cutPoint:]

        newPop.addInd(parameters, i)
        newPop.ind[-1]["pos"] = child1["pos"].copy()
        newPop.addInd(parameters, i+1)
        newPop.ind[-1]["pos"] = child2["pos"].copy()

        '''
        print(f"p1:{parent1['pos']}")
        print(f"p2:{parent2['pos']}")
        print(f"c1:{child1}")
        print(f"c2:{child2}")
        '''

    return newPop
