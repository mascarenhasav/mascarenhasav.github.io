import globalVar
import fitFunction

def evaluate(x, parameters):
    '''
    Fitness function. Returns the error between the fitness of the particle
    and the global optimum
    '''
    fitness = fitFunction.fitnessFunction(x['pos'], parameters)
    globalVar.nevals += 1
    return fitness


def detection(pop, parameters):
    '''
    Check if a change occurred in the environment
    '''
    if parameters["COMP_CHANGE_DETECT_MODE"] == 0:
        #print(f"flag: {globalVar.flagChangeEnv}\t {len(parameters['CHANGES_NEVALS'])}")
        if parameters["CHANGES_NEVALS"][globalVar.flagChangeEnv] <=  globalVar.nevals < parameters["CHANGES_NEVALS"][globalVar.flagChangeEnv+1]:
        #if globalVar.nevals in parameters["CHANGES_NEVALS"]:
            if globalVar.flagChangeEnv < len(parameters["CHANGES_NEVALS"])-2:
                globalVar.flagChangeEnv += 1
            globalVar.mpb.changePeaks()
            return 1
        else:
            return 0

    elif parameters["COMP_CHANGE_DETECT_MODE"] == 1:
        sensor = evaluate(pop.best, parameters)
        if(abs(sensor-pop.best["fit"]) > 0.001):
        #if(pop.best["fit"] != )
            #print(f"[CHANGE OCCURED][NEVAL: {globalVar.nevals}][GEN: {gen}][POP ID: {pop.id}]]")
            print(f"POP BEST: {pop.best}][SENSOR: {sensor}]")
            #print(f"[CHANGE] nevals: {nevals}  sensor: {sensor}  sbest:{swarm.best.fitness.values[0]}")
            pop.best["fit"] = sensor
            return 1
        else:
            return 0
