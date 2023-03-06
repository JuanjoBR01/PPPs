# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:17:08 2020

@author: n.torres11
"""
from math import log
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from math import sin

# Progresive degradation
def shapeFunction(function,t,delta):
    return(function(t)-function(t-delta))

def GammaProcess(T, delta, shape, scale):
    steps = int(T/delta)
    det = [0]
    t = delta
    for i in range(steps) :
        GP = rd.gammavariate(alpha = shapeFunction(shape,t,delta), beta = scale)
        det.append(det[len(det)-1]+GP)
        t = t + delta
    return(np.array(det))

def Progresive(shape,scale,Tmax,delta):
    GP = GammaProcess(T = Tmax, delta = delta, shape = shape, scale = scale)
    det = GP/100
    per = np.maximum(0,1-det) 
    return(per)

# Shock-based degradation
def sizeFunction(function,args):
    if function == "exp":
        response = rd.expovariate(args)
    elif function == "log":
        response = log(rd.lognormvariate(*args))
    elif function == "unif":
        response = rd.uniform(*args)
    return(response)

from stochastic.processes.continuous import PoissonProcess
def CompoundPoissonProcess(T, time, size, arg, delta):
    PP = PoissonProcess(time)
    N = PP.sample(length = T)
    x = [0]
    t = 0
    
    for i in range(1,len(N)):
        while t < N[i] and t < T:
            x.append(x[len(x)-1])
            t = t + delta
        s = sizeFunction(size,arg)
        if t < T:
            x.append(x[len(x)-1]+s)
            t = t + delta
        
    return(np.array(x))
    
def Shock_Based(rateTime,funSize,argSize,Tmax,delta):
    CPP = CompoundPoissonProcess(T = Tmax,time = rateTime,size = funSize,arg =argSize,delta = delta)
    det = CPP/100
    per = np.maximum(0,1-det)    
    return(per)

# Combined degradation
def Combined(shape,scale,rateTime,funSize,argSize,Tmax,delta):
    GP = GammaProcess(T = Tmax, delta = delta, shape = shape, scale = scale)
    detP = GP/100
    CPP = CompoundPoissonProcess(T = Tmax,time = rateTime,size = funSize,arg =argSize,delta = delta)
    detSB = CPP/100
    det = detP + detSB
    per = np.maximum(0,1-det)    
    return(per)
    
# Generate scenarios
def generate(T,num,typeDeterioration,args):
    gamma = []
    for s in range(num):
        if typeDeterioration == "P":
            aux = Progresive(*args)
        elif typeDeterioration == "SB":
            aux = Shock_Based(*args)
        elif typeDeterioration == "C":
            aux = Combined(*args)
        
        #gamma[s] = [round(aux[t*10],2) for t in range(T)]
        gamma.append([round(aux[t*10],2) for t in range(T)])
    return(gamma)

def q_shape(t):
    return(8/90*(t**2))

def s_shape(t):
    return(40*sin(t/9.6+4.725))


# Generate scenarios
# Podemos cambiar el primer parámetro que es a 
scenarios = generate(30,100,'P',(s_shape,1,30,0.1))
scenarios = generate(30,50,'C',(q_shape,1,0.2,"log",(10,2),30,0.1))

'''
for i in range(30):
    plt.plot(range(30), scenarios[i], label = "line " + str(i))

plt.legend()
plt.show()
'''

