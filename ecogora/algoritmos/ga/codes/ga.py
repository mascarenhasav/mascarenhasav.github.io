import elitism
import crossover
import mutation
import abcd
import copy

'''
    GA optimizer
'''
def ga(pop, parameters):
    newPop = abcd.population(parameters, id = 0, fill = 0)
    tempPop = copy.deepcopy(pop)
    tempPop.ind = sorted(tempPop.ind, key = lambda x:x["id"])

    if mutation.cp_mutation(parameters):
        newPop = elitism.elitism(tempPop, newPop, parameters)
        newPop = crossover.crossover(tempPop, newPop, parameters)
        newPop = mutation.mutation(newPop, parameters)

    for i, ind in enumerate(newPop.ind):
        if ind["id"] <= int(parameters["GA_POP_PERC"]*parameters["POPSIZE"]):
            pop.ind[i] = ind

    return pop
