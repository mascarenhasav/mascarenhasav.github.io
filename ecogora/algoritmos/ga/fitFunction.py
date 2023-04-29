import random

def function(x, parameters):
	globalOP = 0
	if(parameters["BENCHMARK"] == "H1"):
	    x = decode(x, parameters)
	    globalOP = function([8.6998, 6.7665])[0]
	elif(parameters["BENCHMARK"] == "BOHA"):
	    x = decode(x, parameters)
	    globalOP = function([0 for _ in range(parameters["NDIM"])])[0]

	fitInd = 0
	fitness = [abs( fitInd - globalOP )]
	return fitness


def ecogora(x, parameters):
	NAISU = "MANO"
	erro = 0
	for i, j in zip(x, NAISU):
		if i != j: erro += 2
	#print(f"[IND:{x}]\t[T: {NAISU}]\n[ERROR: {erro}]")
	return erro
