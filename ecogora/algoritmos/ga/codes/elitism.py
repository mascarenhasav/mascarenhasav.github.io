'''
Elitism operator
'''
def cp_elitism(parameters):
    if parameters["COMP_ELI"] == 1:
        if 0 < parameters["COMP_ELI_PERC"] < 1:
            return 1
        else:
            errorWarning(1.2, "algoConfig.ini", "COMP_ELI_PERC", "The percentage parameter of the Elitism component Elitism should be in the interval ]0, 1[")
            return 0
    elif parameters["COMP_ELI"] != 0:
        errorWarning(1.1, "algoConfig.ini", "COMP_ELI", "Component Elitism should be 0 or 1")
        return 0



def elitism(pop, newPop, parameters):
    for i in range(int(parameters["COMP_ELI_PERC"]*parameters["GA_POP_PERC"]*parameters["POPSIZE"])):
        newPop.addInd(parameters)
        newPop.ind[i] = pop.ind[i]
    return newPop
