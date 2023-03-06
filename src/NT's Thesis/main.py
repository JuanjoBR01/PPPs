# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:23:49 2020

@author: n.torres11
"""
import numpy as np
import random as rd
from math import log, exp, sqrt, sin
import matplotlib.pyplot as plt
import degradation
import sampling
import models
import instance

import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second

# Create instance
I = instance.importPPP()

# Create deterioration
'''
Progresive(shape,scale,Tmax,delta)
Shock_Based(rateTime,funSize,argSize,Tmax,delta)
Combined(shape,scale,rateTime,funSize,argSize,Tmax,delta)
'''

# Function to shape
def q_shape(t):
    return(8/90*(t**2))

def s_shape(t):
    return(40*sin(t/9.6+4.725))

# Generate scenarios
scenarios = degradation.generate(I,50,'C',(q_shape,1,0.2,"log",(10,2),30,0.1))
scenarios = degradation.generate(I,100,'P',(s_shape,1,30,0.1))

# Run model for scenarios generated
x,v,cash,inc,out,OF = models.PPPs_Maintenance(I,scenarios,1)

# SAA 
# Progressive - Quadratic - RN
OG1,PG1,VG1,AG1,PAG1,BOF1,BS1 = sampling.SAA_Method(I,'P',(q_shape,1,30,0.1),1,100,100,500,1000,0.005)

# Progressive - Sigmoidal - RA
OG4,PG4,VG4,AG4,PAG4,BOF4,BS4 = sampling.SAA_Method(I,'P',(s_shape,1,30,0.1),2,100,100,500,1000,0.005)

# Combined - LN - Sigmoidal - RN
OG5,PG5,VG5,AG5,PAG5,BOF5,BS5 = sampling.SAA_Method(I,'C',(s_shape,1,0.2,"log",(10,1),30,0.1),1,100,100,500,1000,0.005)

# Combined - EXP - Quadratic - RA
OG12,PG12,VG12,AG12,PAG12,BOF12,BS12 = sampling.SAA_Method(I,'C',(q_shape,1,0.2,"exp",0.1,30,0.1),2,100,100,500,1000,0.005)
