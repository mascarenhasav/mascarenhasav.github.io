#! /usr/bin/env python3

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
from deap import benchmarks
# ABCD files
import globalVar
import fitFunction
import pso
import de
import ga
import es
import mutation
import changeDetection
from aux import *


# datetime variables
cDate = datetime.datetime.now()
year = cDate.year
month = cDate.month
day = cDate.day
hour = cDate.hour
minute = cDate.minute

#nevals = 0



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
    #fitness = round(fitFunction.fitnessFunction(x['pos'], parameters), 3)
    x["fit"] = fitFunction.fitnessFunction(x['pos'], parameters)
    globalVar.nevals += 1
    if parameters["OFFLINE_ERROR"] and isinstance(globalVar.best["fit"], numbers.Number):
        globalVar.eo_sum += globalVar.best["fit"]
    x["ae"] = 1
    return x



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
                "fit": "NaN", \
                "ae": 0 \
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
    perc_pop = parameters["GA_POP_PERC"]   \
               +parameters["PSO_POP_PERC"] \
               +parameters["DE_POP_PERC"]  \
               +parameters["ES_POP_PERC"]
    print(perc_pop)

    if (abs(perc_pop-1) > 0.001):
        errorWarning(0.0, "algoConfig.ini", "XXX_POP_PERC", "The sum of the percentage of the population to perform the optimizers should be in 1")
        sys.exit()

    if 0 < parameters["GA_POP_PERC"] <= 1:
        for ind in pop.ind:
            if ind["id"] <= int(parameters["GA_POP_PERC"]*parameters["POPSIZE"]):
                ind["type"] = "GA"
            else:
                flag = ind["id"]-1
                break
    elif parameters["GA_POP_PERC"] != 0:
        errorWarning(0.2, "algoConfig.ini", "GA_POP_PERC", "Percentage of the population to perform GA should be in [0, 1]")
        sys.exit()

    if 0 < parameters["PSO_POP_PERC"] <= 1:
        for i in range(flag, len(pop.ind)):
            if pop.ind[i]["id"] <= int(parameters["PSO_POP_PERC"]*parameters["POPSIZE"]+flag):
                pop.ind[i]["type"] = "PSO"
            else:
                flag = pop.ind[i]["id"]-1
                break
    elif parameters["PSO_POP_PERC"] != 0:
        errorWarning(0.3, "algoConfig.ini", "PSO_POP_PERC", "Percentage of the population to perform PSO should be in [0, 1]")
        sys.exit()

    if 0 < parameters["DE_POP_PERC"] <= 1:
        for i in range(flag, len(pop.ind)):
            if pop.ind[i]["id"] <= int(parameters["DE_POP_PERC"]*parameters["POPSIZE"]+flag):
                pop.ind[i]["type"] = "DE"
            else:
                flag = pop.ind[i]["id"]-1
                break
    elif parameters["DE_POP_PERC"] != 0:
        errorWarning(0.4, "algoConfig.ini", "DE_POP_PERC", "Percentage of the population to perform DE should be in [0, 1]")
        sys.exit()

    if 0 < parameters["ES_POP_PERC"] <= 1:
        for i in range(flag, len(pop.ind)):
            if pop.ind[i]["id"] <= int(parameters["ES_POP_PERC"]*parameters["POPSIZE"]+flag):
                pop.ind[i]["type"] = "ES"
            else:
                break
    elif parameters["ES_POP_PERC"] != 0:
        errorWarning(0.5, "algoConfig.ini", "ES_POP_PERC", "Percentage of the population to perform ES should be in [0, 1]")
        sys.exit()


    #random.seed(parameters["SEED"])
    for ind in pop.ind:
        ind["pos"] = [float(globalVar.rng.choice(range(parameters['MIN_POS'], parameters['MAX_POS']))) for _ in range(parameters["NDIM"])]
        if ind["type"] == "PSO":
            ind["vel"] = [float(globalVar.rng.choice(range(parameters["PSO_MIN_VEL"], parameters["PSO_MAX_VEL"]))) for _ in range(parameters["NDIM"])]
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



def evaluatePop(pop, best, parameters):
    '''
    Reevaluate each particle attractor and update swarm best
    If ES_CHANGE_COMP is activated, the position of particles is
    changed by ES strategy
    '''
    for ind in pop.ind:
        if ind["ae"] == 0:
            ind = evaluate(ind, parameters)
            '''
            for i, d in enumerate(ind["pos"]):
                ind["pos"][i] = round(d, 4)
            '''
            ind, best = updateBest(ind, best)

    pop.ind = sorted(pop.ind, key = lambda x:x["fit"])
    pop.best = pop.ind[0].copy()

    return pop, best




'''
Algorithm
'''
def abcd(parameters, seed):
    startTime = time.time()
    filename = f"{path}/{parameters['FILENAME']}"

    bestRuns = []

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
            print(f"[START][RUN:{run:02}]\n[NEVALS:{globalVar.nevals:06}]")
            print(f"==============================================")

        seed = seed*run**5
        parameters["SEED"] = seed

        globalVar.rng = np.random.default_rng(seed)
        globalVar.nevals = 0
        globalVar.mpb = None
        globalVar.best = None
        globalVar.eo_sum = 0

        gen = 1
        genChangeEnv = 0
        env = 0
        flagEnv = 0
        Eo = 0
        change = 0

        # Create the population with POPSIZE individuals
        pops, globalVar.best = createPopulation(parameters)

        #####################################
        # For each pop in pops do the job
        #####################################
        for pop in pops:
            pop = randInit(pop, parameters)

            # Evaluate all the individuals in the pop and update the bests
            pop, globalVar.best = evaluatePop(pop, globalVar.best, parameters)
            for ind in pop.ind:
                ind["ae"] = 0
                # Debug in individual level
                if parameters["LOG_ALL"]:
                    log = [{"run": run, "gen": gen, "nevals":globalVar.nevals, "popId": pop.id, "indId": ind["id"], "indPos": ind["pos"], "indError": ind["fit"], "popBestId": pop.best["id"], "popBestPos": pop.best["pos"], "popBestError": pop.best["fit"], "bestId": globaVar.best["id"], "bestPos": globalVar.best["pos"], "bestError": globalVar.best["fit"], "Eo": Eo, "env": env}]
                    writeLog(mode=1, filename=filename, header=header, data=log)
                if parameters["DEBUG_IND"]:
                    print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']}\t\tERROR:{ind['fit']:.04f}]\t[BEST {globalVar.best['id']:04}: {globalVar.best['pos']}\t\tERROR:{globalVar.best['fit']:.04f}]")


        if not parameters["LOG_ALL"]:
            log = [{"run": run, "gen": gen, "nevals":globalVar.nevals, "bestId": globalVar.best["id"], "bestPos": globalVar.best["pos"], "bestError": globalVar.best["fit"], "Eo": Eo, "env": env}]
            writeLog(mode=1, filename=filename, header=header, data=log)


        #####################################
        # Debug in pop and generation level
        #####################################
        if parameters["DEBUG_POP"]:
            for pop in pops:
                print(f"[POP {pop.id:04}][BEST {pop.best['id']:04}: {pop.best['pos']} ERROR:{globalVar.best['fit']}]")

        if parameters["DEBUG_GEN"]:
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{globalVar.nevals:06}][POP {globalVar.best['pop_id']:04}][BEST {globalVar.best['id']:04}:{globalVar.best['pos']}][ERROR:{globalVar.best['fit']:.04f}][Eo: {Eo:.04f}]")


        ###########################################################################
        # LOOP UNTIL FINISH THE RUN
        ###########################################################################

        while globalVar.nevals < (parameters["NEVALS"]-parameters["POPSIZE"])+1 and globalVar.best["fit"] != 42:
            for pop in pops:

                # Change detection component in the environment

                if(parameters["COMP_CHANGE_DETECT"] == 1):
                    if not change:
                        change = changeDetection.detection(pop, parameters)
                    if change:
                        print(f"[NEVALS: {globalVar.nevals}]")
                        globalVar.best["fit"] = "NaN"
                        pop, globalVar.best = evaluatePop(pop, globalVar.best, parameters)
                        if flagEnv == 0:
                            env += 1
                            genChangeEnv = gen
                            flagEnv = 1
                elif(parameters["COMP_CHANGE_DETECT"] != 0):
                    errorWarning(0.1, "algoConfig.ini", "COMP_CHANGE_DETECT", "Component Change Detection should be 0 or 1")


                #####################################
                # Apply the optimizers in the pops
                #####################################



                if parameters["GA_POP_PERC"]:
                    pop = ga.ga(pop, parameters)


                if parameters["DE_POP_PERC"]:
                    pop = de.de(pop, parameters)

                for i in range(len(pop.ind)):
                    if pop.ind[i]["type"] == "PSO":
                        pop.ind[i] = pso.pso(pop.ind[i], globalVar.best, parameters)
                    elif pop.ind[i]["type"] == "ES":
                        pop.ind[i] = es.es(pop.ind[i], pop.best, parameters)
                        #print(ind)


                #####################################
                # Apply the optimizers in the pops
                #####################################

                if mutation.cp_mutation(parameters, comp=1):
                    pop = mutation.mutation(pop, parameters, comp=1)


                '''
                    The next componentes should be here
                '''


                # Evaluate all the individuals in the pop and update the bests
                pop, globalVar.best = evaluatePop(pop, globalVar.best, parameters)


                for ind in pop.ind:
                    ind["ae"] = 0 # Allow new evaluation
                    # Debug in individual level
                    if parameters["LOG_ALL"]:
                        log = [{"run": run, "gen": gen, "nevals":globalVar.nevals, "popId": pop.id, "indId": ind["id"], "indPos": ind["pos"], "indError": ind["fit"], "popBestId": pop.best["id"], "popBestPos": pop.best["pos"], "popBestError": pop.best["fit"], "bestId": globalVar.best["id"], "bestPos": globalVar.best["pos"], "bestError": globalVar.best["fit"], "Eo": Eo, "env": env}]
                        writeLog(mode=1, filename=filename, header=header, data=log)
                    if parameters["DEBUG_IND"]:
                        print(f"[POP {pop.id:04}][IND {ind['id']:04}: {ind['pos']}\t\tERROR:{ind['fit']:.04f}]\t[BEST {globalVar.best['id']:04}: {globalVar.best['pos']}\t\tERROR:{globalVar.best['fit']:.04f}]")


            change = 0
            if abs(gen - genChangeEnv) > 2:
                flagEnv = 0
                change = 0

            gen += 1

            #####################################
            # Save the log only with the bests of each generation
            #####################################

            if not parameters["LOG_ALL"]:
                Eo = globalVar.eo_sum/globalVar.nevals
                log = [{"run": run, "gen": gen, "nevals":globalVar.nevals, "bestId": globalVar.best["id"], "bestPos": globalVar.best["pos"], "bestError": globalVar.best["fit"], "Eo": Eo, "env": env}]
                writeLog(mode=1, filename=filename, header=header, data=log)


            #####################################
            # Debug in pop and generation level
            #####################################

            if parameters["DEBUG_POP"]:
                for pop in pops:
                    print(f"[POP {pop.id:04}][BEST {pop.best['id']:04}: {pop.best['pos']} ERROR:{pop.best['fit']}]")

            if parameters["DEBUG_GEN"]:
                print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{globalVar.nevals:06}][POP {globalVar.best['pop_id']:04}][BEST {globalVar.best['id']:04}:{globalVar.best['pos']}][ERROR:{globalVar.best['fit']:.04f}][Eo: {Eo:.04f}]")


        #####################################
        # End of the run
        #####################################

        bestRuns.append(globalVar.best)

        if parameters["DEBUG_RUN"]:
            print(f"[RUN:{run:02}][GEN:{gen:04}][NEVALS:{globalVar.nevals:06}][POP {globalVar.best['pop_id']:04}][BEST {globalVar.best['id']:04}:{globalVar.best['pos']} ERROR:{globalVar.best['fit']}]")
        if parameters["DEBUG_RUN_2"]:
            print(f"\n==============================================")
            print(f"[RUN:{run:02}]\n[GEN:{gen:04}][NEVALS:{globalVar.nevals:06}]")
            print(f"[BEST: IND {globalVar.best['id']:04} from POP {globalVar.best['pop_id']:04}")
            print(f"    -[POS: {globalVar.best['pos']}]")
            print(f"    -[Error: {globalVar.best['fit']}]")
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
        print(f"\nFile generated: {path}/data.csv")
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

    if parameters["SEED"] >= 0:
        seed = parameters["SEED"]

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
            seed = int(arg)
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
        if(parameters["GA_POP_PERC"] > 0):
            print(f"--- [GA] Elitism:  \t{parameters['GA_ELI_PERC']*100:.0f}%")
            print(f"--- [GA] Crossover:\t{parameters['GA_CROSS_PERC']*100}%")
            print(f"--- [GA] Mutation: \t{parameters['GA_MUT_PERC']}")
        print(f"-- PSO:\t{parameters['PSO_POP_PERC']*100}%")
        if(parameters["PSO_POP_PERC"] > 0):
            print(f"--- [PSO] Phi1:\t\t{parameters['PSO_PHI1']}")
            print(f"--- [PSO] Phi2:\t\t{parameters['PSO_PHI2']}")
            print(f"--- [PSO] W:\t\t{parameters['PSO_W']}")
        print(f"-- DE:\t{parameters['DE_POP_PERC']*100}%")
        if(parameters["DE_POP_PERC"] > 0):
            print(f"--- [DE] F:\t\t{parameters['DE_F']}")
            print(f"--- [DE] CR:\t\t{parameters['DE_CR']}")
        print(f"-- ES:\t{parameters['ES_POP_PERC']*100}%")
        if(parameters["ES_POP_PERC"] > 0):
            print(f"--- [ES] Rcloud:\t{parameters['ES_RCLOUD']}")
        print(f"- Components used:")
        if(parameters["EXCLUSION_COMP"]):
            print(f"-- [Exlcusion]: Rexcl={parameters['REXCL']}")
        if(parameters["ANTI_CONVERGENCE_COMP"]):
            print(f"-- [ANTI-CONVERGENCE]: Rconv={parameters['RCONV']}")
        if(parameters["LOCAL_SEARCH_COMP"]):
            print(f"-- [LOCAL_SEARCH]: Rls={parameters['RLS']}")


        print()
        print(f"[BENCHMARK SETUP]")
        print(f"- Name: {parameters['BENCHMARK']}")
        print(f"- NDIM: {parameters['NDIM']}")

    time.sleep(1)
    try:
        input("\n\n[Press enter to start...]")
    except SyntaxError:
        pass

    print("\n[START]\n")
    abcd(parameters, seed)
    print("\n[END]\nThx :)\n")


if __name__ == "__main__":
    main()


