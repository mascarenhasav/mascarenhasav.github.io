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

searchSpace = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
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
	pop = sorted(pop, key = lambda x:x.fitness.values[0])
	for i in range(int(parameters["ELI"]*parameters["POPSIZE"])):
		newPop.append(pop[i])
	#for ind in newPop:
		#print(f"[IND: {ind}][FIT: ind.fitness.values]")
	return newPop

'''
Selection operator
'''
def selection(pop, newPop, parameters):
	sumFitness = 0.0
	relFitness = []
	parents = []
	for _ in range(int(parameters["SEL"]*parameters["POPSIZE"])):
		subPop = random.sample(pop, 2)
		subPop = sorted(subPop, key = lambda x:x.fitness.values[0])
		newPop.append(subPop[0])

	'''
	for ind in pop:
		sumFitness += ind.fitness.values[0]
	for ind in pop:
		relFitness.append(ind.fitness.values[0]/sumFitness)
	#Generate probability intervals for each individual
	prob = [sum(relFitness[:i+1]) for i in range(len(relFitness))]

	for _ in range(2):
		r = random.random()
		for i, ind in enumerate(pop):
			if(r <= prob[i]):
				parents.append(ind)
				parents.append(i)
				break
	'''

	return newPop


'''
Crossover operator
'''
def crossover(pop, newPop, parameters):
	l = int(len(pop[0])/2)
	child1 = creator.Individual([random.choice(searchSpace) for _ in range(parameters["INDSIZE"]*parameters["NDIM"])])
	child2 = creator.Individual([random.choice(searchSpace) for _ in range(parameters["INDSIZE"]*parameters["NDIM"])])
	'''
	if random.random() < parameters["CROSS"]:
		print("[CROSSOVER]")
		#parent1, p1Id, parent2, p2Id = selection(pop, parameters)
		print(f"[PARENT1: {parent1}]")
		print(f"[PARENT2: {parent2}]")
		child1[:] = parent1[0:l] + parent2[l:]
		child2[:] = parent2[0:l] + parent1[l:]
		print(f"[CHILD1: {child1}]")
		print(f"[CHILD2: {child2}]")
		pop[p1Id] = child1
		pop[p2Id] = child2
	return pop
	'''

	for _ in range(int((1-parameters["ELI"]+parameters["SEL"])*parameters["POPSIZE"])):
		parent1 = random.choice(pop)
		parent2 = random.choice(pop)
		if random.random() < parameters["CROSS"]:
			#print("[CROSSOVER]")
			#parent1, p1Id, parent2, p2Id = selection(pop, parameters)
			#print(f"[PARENT1: {parent1}]")
			#print(f"[PARENT2: {parent2}]")
			child1[:] = parent1[0:l] + parent2[l:]
			child2[:] = parent2[0:l] + parent1[l:]
			#print(f"[CHILD1: {child1}]")
			#print(f"[CHILD2: {child2}]")
			newPop.append(child1)
			#newPop.append(child2)
		else:
			newPop.append(parent1)
			#newPop.append(parent2)

	return newPop



'''
Mutation operator
'''
def mutation(pop, parameters):
	for ind in pop:
		if random.random() < parameters["MUT"]:
			#print("[MUTATION]")
			#print(f"[BEF: {ind}]")
			bit = random.randint(0, parameters["INDSIZE"]*parameters["NDIM"]-1)
			if(parameters["TYPE"] == "CHAR"):
				ind[bit] = random.choice(searchSpace)
				break
				#print(f"[AFT: {ind}]")
			else:
				if ind[bit] == 1:
					ind[bit] = 0
				else:
					ind[bit] = 1
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
		while nevals < parameters["NEVALS"]+1:
			newPop = []
			#print(pop)
			if (parameters["ELI"] > 0 and parameters["ELI"] <= 1):
				newPop = elitism(pop, newPop, parameters)
			if (parameters["SEL"] > 0 and parameters["SEL"] <= 1):
				newPop = selection(pop, newPop, parameters)
			if (parameters["CROSS"] > 0 and parameters["CROSS"] <= 1):
				newPop = crossover(pop, newPop, parameters)
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

			for ind in pop:
				print(f"[IND: {ind}][FIT:{ind.fitness.values[0]}] [BEST: {best}]")


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
	"INDSIZE": 8,
	"NDIM": 1,
	"MIN": -100,
	"MAX": 100,
	"NEVALS": 200000,
	"RUNS": 1,
	"ELI": 0.2,
	"SEL": 0.2,
	"CROSS": 1,
	"MUT": 0.001,
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


