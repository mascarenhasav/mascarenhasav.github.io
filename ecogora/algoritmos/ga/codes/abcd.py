'''
Base code for GA algorithm

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
import json
import shutil
import numpy as np
from mutation import *
from crossover import *
from elitism import *
from fitFunction import *
from aux import *
from ga import *
from pso import *
from es import *
from deap import benchmarks

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
    if not isinstance(best["fit"], numbers.Number) or (best["fit"] > ind["fit"]): # If first gen, just copy the ind
        best = ind.copy()

    '''
    Update the ind best
    '''
    if not isinstance(ind["best_fit"], numbers.Number) or (ind["best_fit"] > ind["fit"]): # If first gen, just copy the ind
        ind["best_fit"] = ind["fit"]
        ind["best_pos"] = ind["pos"]

    return ind, best


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
    newid = itertools.count(1).__next__    # Get a new id for the population

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
                "type": "NaN", \
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

    def resetId():
        population.newid = itertools.count(1).__next__    # Get a new id for the population


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

    flag = 0

    if 0 < parameters["GA_POP_PERC"] <= 1:
        for ind in pop.ind:
            if ind["id"] <= int(parameters["GA_POP_PERC"]*parameters["POPSIZE"]):
                ind["type"] = "GA"
            else:
                flag = ind["id"]-1
                break

    if 0 < parameters["PSO_POP_PERC"] <= 1:
        for i in range(flag, len(pop.ind)):
            if pop.ind[i]["id"] <= int(parameters["PSO_POP_PERC"]*parameters["POPSIZE"]+flag):
                pop.ind[i]["type"] = "PSO"
            else:
                flag = pop.ind[i]["id"]-1
                break

    if 0 < parameters["ES_POP_PERC"] <= 1:
        for i in range(flag, len(pop.ind)):
            if pop.ind[i]["id"] <= int(parameters["ES_POP_PERC"]*parameters["POPSIZE"]+flag):
                pop.ind[i]["type"] = "ES"
            else:
                break


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
def abcd(parameters, seed):
    startTime = time.time()

    global nevals
    bestRuns = []
    Eo = 0
    env = 0
    filename = f"{path}/{parameters['FILENAME']}"


    # Headers of the log files
    if(parameters["LOG_ALL"]):
        header = ["run", "gen", "nevals", "popId", "indId", "indPos", "indError", "popBestId", "popBestPos", "popBestError", "bestId", "bestPos", "bestError", "Eo", "env"]
    else:
        header = ["run", "gen", "nevals", "bestId", "bestPos", "bestError", "Eo", "env"]
    writeLog(mode=0, filename=filename, header=header)
    #headerOPT = [f"opt{i}" for i in range(parameters["NPEAKS_MPB"])]
    #writeLog(mode=0, filename=f"{path}/optima.csv", header=headerOPT)


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
                ind, best = updateBest(ind, best)

            pop.ind = sorted(pop.ind, key = lambda x:x["fit"])
            pop.best = pop.ind[0].copy()

            # Debug in individual level
            if parameters["DEBUG_IND"] or parameters["LOG_ALL"]:
                for ind in pop.ind:
                    if parameters["LOG_ALL"]:
                        log = [{"run": run, "gen": gen, "nevals":nevals, "popId": pop.id, "indId": ind["id"], "indPos": ind["pos"], "indError": ind["fit"], "popBestId": pop.best["id"], "popBestPos": pop.best["pos"], "popBestError": pop.best["fit"], "bestId": best["id"], "bestPos": best["pos"], "bestError": best["fit"], "Eo": Eo, "env": env}]
                        writeLog(mode=1, filename=filename, header=header, data=log)
                    if parameters["DEBUG_IND"]:
                        print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']} ERROR:{ind['fit']}]\t[BEST {best['id']:04}: {best['pos']} ERROR:{best['fit']}]")


        if not parameters["LOG_ALL"]:
            log = [{"run": run, "gen": gen, "nevals":nevals, "bestId": best["id"], "bestPos": best["pos"], "bestError": best["fit"], "Eo": Eo, "env": env}]
            writeLog(mode=1, filename=filename, header=header, data=log)


        #####################################
        # Debug in pop and generation level
        #####################################
        if parameters["DEBUG_POP"]:
            for pop in pops:
                print(f"[POP {pop.id:04}][BEST {pop.best['id']:04}: {pop.best['pos']} ERROR:{best['fit']}]")

        if parameters["DEBUG_GEN"]:
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][POP {best['pop_id']:04}][BEST {best['id']:04}:{best['pos']} ERROR:{best['fit']}]")


        ###########################################################################
        # LOOP UNTIL FINISH THE RUN
        ###########################################################################

        while nevals < (parameters["NEVALS"]-parameters["POPSIZE"])+1 and best["fit"] != 42:
            for pop in pops:
                #####################################
                # Apply the components in the pops
                #####################################

                pop = ga(pop, parameters)

                for ind in pop.ind:
                    if ind["type"] == "PSO":
                        ind = pso(ind, best, parameters)
                    elif ind["type"] == "ES":
                        ind = es(ind, pop.best, parameters)

                #newPop = pso(pop, newPop, parameters)

                '''
                    The next componentes should be here
                '''

                # Evaluate all the individuals in the pop and update the global best
                for ind in pop.ind:
                    ind["fit"] = evaluate(ind, parameters)
                    for i, d in enumerate(ind["pos"]):
                        ind["pos"][i] = round(d, 4)
                    ind, best = updateBest(ind, best)

                pop.ind = sorted(pop.ind, key = lambda x:x["fit"])
                pop.best = pop.ind[0].copy()

                # Debug in individual level
                if parameters["DEBUG_IND"] or parameters["LOG_ALL"]:
                    for ind in pop.ind:
                        if parameters["LOG_ALL"]:
                            log = [{"run": run, "gen": gen, "nevals":nevals, "popId": pop.id, "indId": ind["id"], "indPos": ind["pos"], "indError": ind["fit"], "popBestId": pop.best["id"], "popBestPos": pop.best["pos"], "popBestError": pop.best["fit"], "bestId": best["id"], "bestPos": best["pos"], "bestError": best["fit"], "Eo": Eo, "env": env}]
                            writeLog(mode=1, filename=filename, header=header, data=log)
                        if parameters["DEBUG_IND"]:
                            print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']} ERROR:{ind['fit']}]\t[BEST {best['id']:04}: {best['pos']} ERROR:{best['fit']}]")

            gen += 1

            #####################################
            # Save the log only with the bests of each generation
            #####################################

            if not parameters["LOG_ALL"]:
                log = [{"run": run, "gen": gen, "nevals":nevals, "bestId": best["id"], "bestPos": best["pos"], "bestError": best["fit"], "Eo": Eo, "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)


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
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{nevals:06}][POP {best['pop_id']:04}][BEST {best['id']:04}:{best['pos']} ERROR:{best['fit']}]")
        if parameters["DEBUG_RUN_2"]:
            print(f"\n==============================================")
            print(f"[RUN:{run:02}]\n[GEN:{gen:04}][NEVALS:{nevals:06}]")
            print(f"[BEST: IND {best['id']:04} from POP {best['pop_id']:04}")
            print(f"    -[POS: {best['pos']}]")
            print(f"    -[Error: {best['fit']}]")
            print(f"==============================================")


        population.resetId()


    if parameters["RUNS"] > 1:
        bests = [d["fit"] for d in bestRuns]
        meanBest = np.mean(bests)
        stdBest = np.std(bests)
        if parameters["DEBUG_RUN"]:
            print(f"\n==============================================")
            print(f"[RUNS:{parameters['RUNS']}]")
            print(f"[BEST MEAN: {meanBest:.2f}({stdBest:.2f})]")
            print(f"==============================================\n")

    executionTime = (time.time() - startTime)

    #print(f"File generated: {path}/data.csv")
    if(parameters["DEBUG_RUN"]):
        print(f"File generated: {path}/data.csv")
        print(f'Time Exec: {str(executionTime)} s\n')


    # Copy the config.ini file to the experiment dir
    if(parameters["CONFIG_COPY"]):
        shutil.copyfile("algoConfig.ini", f"{path}/algoConfig.ini")
        shutil.copyfile("benchConfig.ini", f"{path}/benchConfig.ini")
        shutil.copyfile("frameConfig.ini", f"{path}/frameConfig.ini")

    # Evaluate the offline error
    if(parameters["OFFLINE_ERROR"]):
        if (parameters["DEBUG_RUN"]):
            print("\n[METRIC]")
            os.system(f"python3 {sys.path[0]}/metrics/offlineError.py -p {path} -d 1")
        else:
            os.system(f"python3 {sys.path[0]}/metrics/offlineError.py -p {path}")

    if(parameters["BEBC_ERROR"]):
        if (parameters["DEBUG_RUN"]):
            print("\n[METRICS]")
            os.system(f"python3 {sys.path[0]}/metrics/bestBeforeChange.py -p {path} -d 1")
        else:
            os.system(f"python3 {sys.path[0]}/metrics/bestBeforeChange.py -p {path}")




def main():
    global path
    seed = minute
    arg_help = "{0} -s <seed> -p <path>".format(sys.argv[0])
    path = "."

   # Read the parameters from the config file
    with open(f"{path}/algoConfig.ini") as f:
        parameters0 = json.loads(f.read())
    with open(f"{path}/benchConfig.ini") as f:
        parameters1 = json.loads(f.read())
    with open(f"{path}/frameConfig.ini") as f:
        parameters2 = json.loads(f.read())

    parameters = parameters0 | parameters1 | parameters2

    if path == ".":
        path = f"{parameters['PATH']}/{parameters['ALGORITHM']}"
        path = checkDirs(path)

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

    if parameters["DEBUG_RUN"]:
        print(f"======================================================")
        print(f"   AbCD Framework for Dynamic Optimization Problems")
        print(f"======================================================\n")
        print(f"[ALGORITHM SETUP]")
        print(f"- Name: {parameters['ALGORITHM']}")
        print(f"- Population (percentage of total individuals):")
        print(f"-- GA:\t{parameters['GA_POP_PERC']*100}%")
        print(f"-- PSO:\t{parameters['PSO_POP_PERC']*100}%")
        print(f"-- ES:\t{parameters['ES_POP_PERC']*100}%")
        if(parameters["ES_POP_PERC"] > 0):
            print(f"--- [ES]: Rcloud={parameters['ES_RCLOUD']}")
        print(f"-- DE:\t{parameters['DE_POP_PERC']*100}%")
        print(f"- Components used:")
        if(parameters["EXCLUSION_COMP"]):
            print(f"-- [Exlcusion]: Rexcl={parameters['REXCL']}")
        if(parameters["ANTI_CONVERGENCE_COMP"]):
            print(f"-- [ANTI-CONVERGENCE]: Rconv={parameters['RCONV']}")
        if(parameters["LOCAL_SEARCH_COMP"]):
            print(f"-- [LOCAL_SEARCH]: Rls={parameters['RLS']}")

        #print(benchmarks.schwefel([420, 420]))
        #print(benchmarks.schwefel([-2.0, 2.0]))
        #e()

        print()
        print(f"[BENCHMARK SETUP]")
        print(f"- Name: {parameters['BENCHMARK']}")
        print(f"- NDIM: {parameters['NDIM']}")

    print("\n[START]\n")
    abcd(parameters, seed)
    print("\n[END]\nThx :)\n")


if __name__ == "__main__":
    main()


