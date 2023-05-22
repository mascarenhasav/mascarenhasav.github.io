import globalVar
import fitFunction

def evaluate(x, parameters):
    '''
    Fitness function. Returns the error between the fitness of the particle
    and the global optimum
    '''
    fitness = round(fitFunction.fitnessFunction(x['pos'], parameters), 3)
    globalVar.nevals += 1
    return fitness


def detection(pop, parameters):
    '''
    Check if a change occurred in the environment
    '''
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
