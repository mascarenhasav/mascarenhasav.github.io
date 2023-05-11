'''
Base code for GA algirithm with DEAP library.

Ecogora

Alexandre Mascarenhas

2023/1
'''
import itertools
import random
import datetime
import csv
import sys
import getopt
import time
import numbers
import numpy as np
from encoder import *
from mutation import *
from crossover import *
from elitism import *
from fitFunction import *

# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

nevals = 0



def updateBest(ind, best):
    '''
    Update the global best individual
    '''
    if not isinstance(best["fit"], numbers.Number): # If first gen, just copy the ind
        best = ind.copy()
    elif (best["fit"] > ind["fit"]):
        best = ind.copy()

    return best


def evaluate(x, parameters):
    '''
    Fitness function. Returns the error between the fitness of the particle
    and the global optimum
    '''
    global nevals
    fitness = round(fitnessFunction(x['pos'], parameters), 3)
    nevals += 1
    return fitness



class population():
    '''
        Class of the population
    '''
    newid = itertools.count().__next__    # Get a new id for the population

    def __init__(self, parameters, id = 1, fill = 1):
        if(id == 0):    # Temporary population doesnt have an id
            self.id = 0
        else:
            self.id = population.newid()

        self.popsize = parameters["POPSIZE"]

        self.ind = []
        if fill == 1:
            for i in range(1, parameters["POPSIZE"]+1):
                self.addInd(parameters, i)

        self.best = self.createInd(parameters, 0)

    def createInd(self, parameters, ind_id=-1):
        attr = {"pop_id": self.id, \
                "id": ind_id, \
                "type": "GA", \
                "pos": [0 for _ in range(parameters["NDIM"])], \
                "vel": [0 for _ in range(parameters["NDIM"])], \
                "best_pos": [0 for _ in range(parameters["NDIM"])], \
                "best_fit": "NaN", \
                "fit": "NaN" \
                }
        return attr

    def addInd(self, parameters, ind_id=-1):
        flag = 0
        ids = [d["id"] for d in self.ind]
        while flag == 0:
            if ind_id in ids:  # If id is already in the population go to next
                ind_id += 1
            else:
                flag = 1
        self.ind.append(self.createInd(parameters, ind_id))

def createPopulation(parameters):
    '''
        This function is to create the populations and individuals
    '''
    pop = []
    if(parameters["COMP_MULTIPOP"] == 1):
        for _ in range (parameters["COMP_MULTIPOP_N"]):
            pop.append(population(parameters))
    elif(parameters["COMP_MULTIPOP"] == 0):
        pop.append(population(parameters))
    else:
        errorWarning(0.1, "algoConfig.ini", "COMP_MULTIPOP", "Component Multipopulation should be 0 or 1")

    best = pop[0].ind[0].copy()
    best["id"] = "NaN"

    return pop, best


def randInit(pop, parameters):
    '''
        Random initialization of the individuals
    '''
    for ind in pop.ind:
        ind["pos"] = [float(random.choice(range(parameters['MIN_POS'], parameters['MAX_POS']))) for _ in range(parameters["NDIM"])]
        if ind["type"] == "PSO":
            ind["vel"] = [float(random.choice(range(parameters["MIN_VEL"], parameters["MAX_VEL"]))) for _ in range(parameters["NDIM"])]
    return pop


def errorWarning(nError=0.0, file="NONE", parameter="NONE", text="NONE"):
    '''
        Print error function
    '''
    print(f"[ERROR][{nError}]")
    print(f"--[File: '{file}']")
    print(f"--[parameter: '{parameter}']")
    print(f"----[{text}]")
    sys.exit()


'''
Algorithm
'''
def ga(parameters, seed):
    startTime = time.time()

    global nevals
    bestRuns = []

    #####################################
    # Main loop of the runs
    #####################################

    for run in range(1, parameters["RUNS"]+1):
        if parameters["DEBUG_RUN_2"]:
            print(f"\n==============================================")
            print(f"[START][RUN:{run:02}]\n[NEVALS:{nevals:06}]")
            print(f"==============================================")

        random.seed(seed*run**5)
        nevals = 0
        gen = 1

        # Create the population with POPSIZE individuals
        pops, best = createPopulation(parameters)


        #####################################
        # For each pop in pops do the job
        #####################################
        for pop in pops:
            pop = randInit(pop, parameters)
            for ind in pop.ind:
                ind["fit"] = evaluate(ind, parameters)
                best = updateBest(ind, best)

            pop.ind = sorted(pop.ind, key = lambda x:x["fit"])
            pop.best = pop.ind[0].copy()

            # Debug in individual level
            if parameters["DEBUG_IND"]:
                for ind in pop.ind:
                    print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']} ERROR:{ind['fit']}]\t[BEST {best['id']:04}: {best['pos']} ERROR:{best['fit']}]")


        #####################################
        # Debug in pop and generation level
        #####################################
        if parameters["DEBUG_POP"]:
            for pop in pops:
                print(f"[POP {pop.id:04}][BEST {pop.best['id']:04}: {pop.best['pos']} ERROR:{best['fit']}]")

        if parameters["DEBUG_GEN"]:
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][POP {best['pop_id']:04}][BEST {best['id']:04}:{best['pos']}] ERROR:{best['fit']}")



        # Repeat until reach the number of evals
        while nevals < (parameters["NEVALS"]-parameters["POPSIZE"])+1 and best["fit"] != 42:

            #####################################
            # For each pop in pops do the job
            #####################################
            for pop in pops:
                newPop = population(parameters, id = 0, fill = 0)

                #####################################
                # Apply the components in the pops
                #####################################

                if cp_elitism(parameters):
                    newPop = elitism(pop, newPop, parameters)

                if cp_crossover(parameters):
                    newPop = crossover(pop, newPop, parameters)

                if cp_mutation(parameters):
                    newPop = mutation(newPop, parameters)

                pop = newPop


                # Evaluate all the individuals in the pop and update the global best
                for ind in pop.ind:
                    ind["fit"] = evaluate(ind, parameters)
                    for i, d in enumerate(ind["pos"]):
                        ind["pos"][i] = round(d, 4)
                    best = updateBest(ind, best)

                pop.ind = sorted(pop.ind, key = lambda x:x["fit"])
                pop.best = pop.ind[0].copy()

                # Debug in individual level
                if parameters["DEBUG_IND"]:
                    for ind in pop.ind:
                        print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']} ERROR:{ind['fit']}]\t[BEST {best['id']:04}: {best['pos']} ERROR:{best['fit']}]")

            gen += 1

            #####################################
            # Save the log only with the bests of each generation
            #####################################

            #log = [{"run": run, "gen": gen, "nevals":nevals, "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
            #writeLog(mode=1, filename=filename, header=header, data=log)


            #####################################
            # Debug in pop and generation level
            #####################################

            if parameters["DEBUG_POP"]:
                for pop in pops:
                    print(f"[POP {pop.id:04}][BEST {pop.best['id']:04}: {pop.best['pos']} ERROR:{pop.best['fit']}]")

            if parameters["DEBUG_GEN"]:
                print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][POP {best['pop_id']:04}][BEST {best['id']:04}:{best['pos']}] ERROR:{best['fit']}")


        #####################################
        # End of the run
        #####################################

        bestRuns.append(best)

        if parameters["DEBUG_RUN"]:
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][POP {best['pop_id']:04}][BEST {best['id']:04}:{best['pos']}] ERROR:{best['fit']}")
        if parameters["DEBUG_RUN_2"]:
            print(f"\n==============================================")
            print(f"[END][RUN:{run:02}]\n[GEN:{gen:04}][NEVALS:{nevals:06}]")
            print(f"[BEST: IND {best['id']:04} from POP {best['pop_id']:04}")
            print(f"    -[POS: {best['pos']}]")
            print(f"    -[Error: {best['fit']}]")
            print(f"==============================================")


    if parameters["RUNS"] > 1:
        bests = [d["fit"] for d in bestRuns]
        meanBest = np.mean(bests)
        stdBest = np.std(bests)
        if parameters["DEBUG_RUN"]:
            print(f"\n==============================================")
            print(f"[END][RUNS:{parameters['RUNS']}]")
            print(f"[BEST MEAN: {meanBest:.2f}({stdBest:.2f})]")
            print(f"==============================================")

    executionTime = (time.time() - startTime)

    #print(f"File generated: {path}/data.csv")
    print(f'\nTime Exec: {str(executionTime)} s\n')




def main():
    global path
    seed = minute
    arg_help = "{0} -s <seed> -p <path>".format(sys.argv[0])
    path = "."

    parameters = {
    "RUNS": 10,
    "NEVALS": 100000,
    "POPSIZE": 100,
    "NDIM": 2,
    "ENCODER": 0,
    "INDSIZE": 16,
    "MIN_POS": -100,
    "MAX_POS": 100,
    "MIN_VEL": -100,
    "MAX_VEL": 100,
    "COMP_MULTIPOP": 0,
    "COMP_MULTIPOP_N": 2,
    "COMP_ELI": 1,
    "COMP_ELI_PERC": 0.1,
    "COMP_CROSS": 1,
    "COMP_CROSS_PERC": 0.5,
    "COMP_MUT": 1,
    "COMP_MUT_PERC": 0.1,
    "COMP_MUT_STD": 5,
    "BENCHMARK": "H1",
    "DEBUG_RUN": 1,
    "DEBUG_RUN_2": 0,
    "DEBUG_GEN": 0,
    "DEBUG_POP": 0,
    "DEBUG_IND": 0
    }

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:p:", ["help", "seed=", "path="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--seed"):
            seed = arg
        elif opt in ("-p", "--path"):
            path = arg

    print(f"======================================================")
    print(f"                   GA algorithm                       ")
    print(f"======================================================\n")
    print(f"[ALGORITHM SETUP]")
    print(f"{parameters}")
    print("\n[START]\n")
    ga(parameters, seed)
    print("\n[END]\nThx :)\n")


if __name__ == "__main__":
    main()


