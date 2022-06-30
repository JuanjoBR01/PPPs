# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:10:49 2020
@author: n.torres11
"""

import numpy as np
from math import log, exp, sqrt
from scipy.stats import norm
import statistics as stats
import degradation
import models


import winsound
frequency = 2000  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second

# SAA iteration
def iteration(I, typeDeterioration, args, riskMeasure, batchSize, numBatches, sampleEvaluation, sampleSelection):
    
    objFunctions = []
    results = {}
    saaFunctions = []
    
    for batch in range(numBatches):
        sample = degradation.generate(I,batchSize,typeDeterioration,args)
        x_batch,v,c,ea,ex,OF = models.PPPs_Maintenance(I,sample,riskMeasure)
        objFunctions.append(OF)
        results[batch] = x_batch
        AUX_ES,OF_ES = models.PPPs_Function(I,sampleEvaluation,riskMeasure,x_batch)
        saaFunctions.append(OF_ES)
    
    bestSolution = results[np.array(saaFunctions).argmin()]
    bestOF = np.array(saaFunctions).min()
    gap = np.mean(objFunctions) - bestOF
    
    AUX_SS,OF_SS = models.PPPs_Function(I,sampleSelection,riskMeasure,bestSolution)
    
    gap = np.mean(objFunctions) - OF_SS
    percentageGap = gap/np.mean(objFunctions)
    
    var = stats.variance(AUX_SS) + stats.variance(objFunctions)
    acc = gap + norm.ppf(0.95)*sqrt(var)
       
    return(gap,percentageGap,var,acc,bestSolution,OF_SS)
    
#Sample average approximation
def SAA(I, typeDeterioration, args, riskMeasure, batchSize, numBatch, evaluationSize, selectionSize, tolerance):
    optimalityGap = {}
    percentajeGap = {}
    varianceGap = {}
    accuracyGap = {}
    objectiveFunction ={}
    solution = {}
    
    sampleEvaluation = degradation.generate(I,evaluationSize,typeDeterioration,args)
    sampleSelection = degradation.generate(I,selectionSize,typeDeterioration,args)
    
    for M in numBatch:
        auxOG = []
        auxPG = []
        auxVG = []
        auxAG = []
        auxOF = []
        auxS = {}
        
        for N in batchSize:
            OG,PG,VG,AG,BS,BOF = iteration(I,typeDeterioration,args,riskMeasure,N,M,sampleEvaluation,sampleSelection)
            auxOG.append(OG)
            auxPG.append(PG)
            auxVG.append(VG)
            auxAG.append(AG)
            auxOF.append(BOF)
            auxS[N] = BS
            winsound.Beep(frequency, duration)
            
            if PG <= tolerance:
                print(str(N)+ " sample size")
                break
        
        if PG <= tolerance:
            print(str(M)+ " replications")
            break
        
        optimalityGap[M] = auxOG
        percentajeGap[M] = auxPG
        varianceGap[M] = auxVG
        accuracyGap[M] = auxAG
        objectiveFunction[M] = auxOF
        solution[M] = auxS
        
    return(optimalityGap,percentajeGap,varianceGap,accuracyGap,objectiveFunction,solution) 

# SAA iteration
def iteration_SAA(I, typeDeterioration, args, riskMeasure, batchSize, numBatches, sampleEvaluation, sampleSelection, listOF, dictResults, listSAA):
       
    for batch in range(numBatches-10,numBatches,1):
        sample = degradation.generate(I,batchSize,typeDeterioration,args)
        x_batch,v,c,ea,ex,OF = models.PPPs_Maintenance(I,sample,riskMeasure)
        listOF.append(OF)
        dictResults[batch] = x_batch
        AUX_ES,OF_ES = models.PPPs_Function(I,sampleEvaluation,riskMeasure,x_batch)
        listSAA.append(OF_ES)
    
    bestSolution = dictResults[np.array(listSAA).argmax()]
    
    AUX_SS,OF_SS = models.PPPs_Function(I,sampleSelection,riskMeasure,bestSolution)
    
    gap = np.mean(listOF) - OF_SS
    pGap = gap/np.mean(listOF)
    
    var = stats.variance(AUX_SS) + stats.variance(listOF)
    acc = gap + norm.ppf(0.95)*sqrt(var)
    pAcc = acc/np.mean(listOF)
       
    return(gap,pGap,var,acc,pAcc,bestSolution,OF_SS,listOF,dictResults,listSAA)


# SAA 2
def SAA_Method(I, typeDeterioration, args, riskMeasure, maxSample, maxBatches, evaluationSize, selectionSize, tolerance):    
    optimalityGap = {}
    percentageGap = {}
    varianceGap = {}
    accuracyGap = {}
    percentageAccuracyGap = {}
    objectiveFunction ={}
    solution = {}
    
    sampleEvaluation = degradation.generate(I,evaluationSize,typeDeterioration,args)
    sampleSelection = degradation.generate(I,selectionSize,typeDeterioration,args)
    
    N = 10
    
    while N <= maxSample:
        auxOG = []
        auxPG = []
        auxVG = []
        auxAG = []
        auxPAG = []
        auxOF = []
        auxS = {}
        M =  10
        
        objFunctions = []
        results = {}
        saaFunctions = []
        
        while M <= maxBatches:
            OG,PG,VG,AG,PAG,BS,BOF,objFunctions,results,saaFunctions = iteration_SAA(I,typeDeterioration,args,riskMeasure,N,M,sampleEvaluation,sampleSelection,objFunctions,results,saaFunctions)
            auxOG.append(OG)
            auxPG.append(PG)
            auxVG.append(VG)
            auxAG.append(AG)
            auxPAG.append(PAG)
            auxOF.append(BOF)
            auxS[M] = BS
            winsound.Beep(frequency, duration)
            
            if PG <= tolerance:
                print(str(M)+ " replications")
                break
            else:
                M = M + 10
                
        optimalityGap[N] = auxOG
        percentageGap[N] = auxPG
        varianceGap[N] = auxVG
        accuracyGap[N] = auxAG
        percentageAccuracyGap[N] = auxPAG
        objectiveFunction[N] = auxOF
        solution[N] = auxS
        
        if PG <= tolerance:
            print(str(N)+ " sample size")
            print(BOF)
            print(OG)
            print(PG)
            print(sqrt(VG))
            break
        else:
            N = N + 10
        
    return(optimalityGap,percentageGap,varianceGap,accuracyGap,percentageAccuracyGap,objectiveFunction,solution) 