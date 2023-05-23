import globalVar

'''
Apply ES on the particle
'''
def es(ind, popBest, parameters, P=1):
    rcloud = parameters["ES_RCLOUD"]
    for i in range(parameters["NDIM"]):
        ind["pos"][i] = popBest["pos"][i] + P*(globalVar.rng.uniform(-1, 1)*rcloud)
    return ind

