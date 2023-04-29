'''
Base code for GA algirithm with DEAP library.


Ecogora

Alexandre Mascarenhas

2023/1
'''
import json
import shutil
import itertools
import operator
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
import ast
import sys
import getopt
import fitFunction
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.benchmarks import movingpeaks
import time

# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

searchSpace = '''ABCDEFGHIJKLMNOPQRSTUVWXYZ'''
nevals = 0



'''
Decode function for continous problems
'''
def decode(binValue, parameters):
	sum = 0
	j = 0
	l = int(len(binValue)/parameters["NDIM"])
	result = []
	precision = (parameters["MAX"]-parameters["MIN"])/(2**(l)-1)
	for d in range(1, parameters["NDIM"]+1):
		for i, bin in enumerate(binValue[j:d*l], 0):
			sum += bin*(2**(i))
		decode = sum*precision + parameters["MIN"]
		result.append(decode)
		j = d*l
		sum = 0
	return result

'''
Generate a new Individual
'''
def generate(parameters):
	if(parameters["TYPE"] == "CHAR"):
		ind = creator.Individual([random.choice(searchSpace) for _ in range(parameters["INDSIZE"]*parameters["NDIM"])])
	else:
		ind = creator.Individual([random.randrange(0, 2, 1) for _ in range(parameters["INDSIZE"]*parameters["NDIM"])])
	return ind

'''
Define the functions responsibles to create the objects of the algorithm
'''
def createToolbox(parameters):
	toolbox = base.Toolbox()
	toolbox.register("individual", generate, parameters=parameters)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	return toolbox


'''
Update the best individuals
'''
def updateBest(ind, best):
    # Check if the particles are the best of itself and best at all
    if not best or best.fitness < ind.fitness:
        best = creator.Individual(ind)
        best.fitness.values = ind.fitness.values

    return ind, best

'''
Fitness function. Returns the error between the fitness of the particle
and the global optimum
'''
def evaluate(x, function, parameters):
	global nevals

	if(parameters["BENCHMARK"] == "ECOGORA"):
		fitness = fitFunction.ecogora(x, parameters)
	else:
		x = decode(x, parameters)
		fitness = fitFunction.function(x, parameters)

	nevals += 1

	return fitness,


'''
Elitism operator
'''
def elitism(pop, newPop, parameters):
	for i in range(parameters["ELI"]):
		newPop.append(pop[i])
	return newPop




def condition(element):
    return element.fitness.values[0]
def tournament(pop, parameters):
    l = int(len(pop)/2)
    c = []
    #for _ in range(int(0.1*parameters["POPSIZE"])):
    for _ in range(3):
        c.append(random.choice(pop[:int(parameters["CROSS"]*parameters["POPSIZE"])]))
    #print(c)

    choosed = min(c, key=condition)
    return choosed

'''
Crossover operator
'''
def crossover(pop, newPop, function, parameters):
#    for _ in range(int((1-parameters["ELI"])*parameters["POPSIZE"])):
    for _ in range(parameters["POPSIZE"] - parameters["ELI"]):
        parent1 = tournament(pop, parameters)
        parent2 = tournament(pop, parameters)
        cutPoint = random.choice(range(len(parent1)))
        child = creator.Individual(parent1)
        for i in range(cutPoint):
            child[i] = parent2[i]
        child.fitness.values = evaluate(child, function, parameters)
        newPop.append(child)

    return newPop



'''
Mutation operator
'''
def mutation(pop, parameters):
    #for i in range(int(parameters["ELI"]*parameters["POPSIZE"]), parameters["POPSIZE"]):
    for i in range(parameters["ELI"], parameters["POPSIZE"]):
        for j in range(parameters["INDSIZE"]):
            if random.random() < parameters["MUT"]:
                #print("[MUTATION]")
                #print(f"[BEF: {ind}]")
                #bit = random.randint(0, parameters["INDSIZE"]*parameters["NDIM"]-1)
                if(parameters["TYPE"] == "CHAR"):
                    pop[i][j] = random.choice(searchSpace)
                    break
                    #print(f"[AFT: {ind}]")
                else:
                    if pop[i][j] == 1:
                        pop[i][j] = 0
                    else:
                        pop[i][j] = 1
    return pop



'''
Algorithm
'''
def ga(parameters, seed):
    startTime = time.time()
    # Create the DEAP creators
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the toolbox functions
    toolbox = createToolbox(parameters)

    global nevals


    # Main loop of ITER runs
    for run in range(1, parameters["RUNS"]+1):
        random.seed(run**5)
        best = None
        nevals = 0
        gen = 0

        # Initialize the benchmark for each run with seed being the minute
        random.seed(a=None)
        #fitFunction = benchmarks.h1
        #fitFunction = benchmarks.bohachevsky
        function = "ecogora"

        # Create the population with POPSIZE individuals
        pop = toolbox.population(n=parameters["POPSIZE"])
        for ind in pop:
            ind.fitness.values = evaluate(ind, function, parameters)
            ind, best = updateBest(ind, best)
        pop = sorted(pop, key = lambda x:x.fitness.values[0])
        print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][BEST:{best}] Best:{best.fitness.values[0]:.4f}")
        for ind in pop:
            print(f"[IND: {ind}][FIT:{ind.fitness.values[0]}] [BEST: {best}]")

        # Repeat until reach the number of evals
        while nevals < parameters["NEVALS"]+1 and best.fitness.values[0] != 0:
            newPop = []
            #print(pop)
            if (parameters["ELI"] > 0 and parameters["ELI"] <= 1):
                newPop = elitism(pop, newPop, parameters)
            #if (parameters["SEL"] > 0 and parameters["SEL"] <= 1):
                #newPop = selection(pop, newPop, parameters)
            if (parameters["CROSS"] > 0 and parameters["CROSS"] <= 1):
                newPop = crossover(pop, newPop, function, parameters)
            if (parameters["MUT"] > 0 and parameters["MUT"] <= 1):
                newPop = mutation(pop, parameters)

            pop = newPop.copy()

            for ind in pop:
                ind.fitness.values = evaluate(ind, function, parameters)
                ind, best = updateBest(ind, best)

            pop = sorted(pop, key = lambda x:x.fitness.values[0])


            # Save the log only with the bests of each generation
            #log = [{"run": run, "gen": gen, "nevals":nevals, "best": best, "bestError": best.fitness.values[0], "Eo": Eo, "env": env}]
            #writeLog(mode=1, filename=filename, header=header, data=log)
            gen += 1
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][BEST:{best}] Error:{best.fitness.values[0]:.4f}")

            #for ind in pop:
                #print(f"[IND: {ind}][FIT:{ind.fitness.values[0]}] [BEST: {best}]")


        if(parameters["TYPE"] == "CHAR"):
            print(f"[END][RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}]\n[BEST:{best}]\n[Error:{best.fitness.values[0]}]")
        else:
            print(f"[END][RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}]\n[BEST:{best}][DECODE: {decode(best, parameters)}]\n[Error:{best.fitness.values[0]:.4f}]")

    executionTime = (time.time() - startTime)

    #print(f"File generated: {path}/data.csv")
    print(f'\nTime Exec: {str(executionTime)} s\n')




def main():
    global path
    seed = minute
    arg_help = "{0} -s <seed> -p <path>".format(sys.argv[0])
    path = "."

    parameters = {
    "POPSIZE": 100,
    "INDSIZE": 4,
    "NDIM": 1,
    "MIN": -100,
    "MAX": 100,
    "NEVALS": 1000000,
    "RUNS": 1,
    "ELI": 20,
    "CROSS": 0.2,
    "MUT": 0.2,
    "TYPE": "CHAR",
    "BENCHMARK": "ECOGORA"
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


