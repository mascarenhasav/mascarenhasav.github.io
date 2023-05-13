from elitism import *
from crossover import *
from mutation import *
from abcd import *

'''
    GA optimizer
'''
def ga(pop, parameters):
    newPop = population(parameters, id = 0, fill = 0)
    tempPop = pop
    tempPop.ind = sorted(tempPop.ind, key = lambda x:x["id"])

    newPop = elitism(tempPop, newPop, parameters)
    newPop = crossover(tempPop, newPop, parameters)
    newPop = mutation(newPop, parameters)

    for i, ind in enumerate(newPop.ind):
        if ind["id"] <= int(parameters["GA_POP_PERC"]*parameters["POPSIZE"]):
            pop.ind[i] = ind

    return pop
