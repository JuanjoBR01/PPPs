'''
Maintenance Model PPP's (with maintenance as a proactive decision)
Author: Natalia Torres
Created on Mar 14 2020
 '''

import numpy as np
import random as rd
from math import log, exp, sqrt
from scipy.stats import norm
import statistics as stats
from gurobipy import *
import degradation

# Model Solving for the Agent
def PPPs_Maintenance(I, gamma, riskMeasure):
    '''
    Parameters
    '''
    I.S = gamma.keys()
    I.prob = {s:1/len(I.S) for s in I.S}
    
    '''
    MODEL
    '''
    m = Model()
    
    '''
    VARIABLES
    '''
    x = {t:m.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}
    y = {t:m.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}
    b = {(t,tau):m.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}
    z = {(t,l,s):m.addVar(vtype=GRB.BINARY, name="z_"+str((t,l,s))) for t in I.T for l in I.L for s in I.S}
    v = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="v_"+str((t,s))) for t in I.T for s in I.S}
    pplus = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str((t,s))) for t in I.T for s in I.S}
    pminus = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str((t,s))) for t in I.T for s in I.S}
    pdot = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str((t,s))) for t in I.T for s in I.S}
    w = {t:m.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}
    u = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="u_"+str((t,s))) for t in I.T for s in I.S}
    
    '''
    OBJECTIVE
    '''
    # Earnings-Expenditures
    m.setObjective(quicksum(I.prob[s]*quicksum(pplus[t,s]-pminus[t,s] for t in I.T) for s in I.S), GRB.MAXIMIZE)
    	
    '''
    CONSTRAINTS
    '''
    # Initialization
    m.addConstr(y[0] == 0, "iniY") 
    m.addConstr(w[0] == 0, "iniW") 
    
    for s in I.S:    	
        m.addConstr(u[0,s] == 0, "iniU_"+str(s))
        m.addConstr(pdot[0,s] == pplus[0,s] - pminus[0,s], "iniCash_"+str(s)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            m.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            m.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            m.addConstr(w[t] >= y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            m.addConstr(w[t] <= len(I.T)*x[t], "linW3_"+str(t))
            
            for s in I.S:
                # Linearization for u (for performance)
                m.addConstr(u[t,s] <= v[t,s], "linU1_"+str((t,s)))
                m.addConstr(u[t,s] >= v[t,s] - (1-x[t]), "linU2_"+str((t,s)))
                m.addConstr(u[t,s] <= x[t], "linU3_"+str((t,s)))
                
                # Update available cash
                m.addConstr(pdot[t,s] == pdot[t-1,s] + pplus[t,s] - pminus[t,s], "cash_"+str((t,s)))
            
        # Binarization of y (to retrieve performance)
        m.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        m.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        for s in I.S:
            # Quantification of v (get performance)
            m.addConstr(v[t,s] == quicksum(gamma[s][tau]*b[t,tau] for tau in I.T), "quantV_"+str((t,s)))
            
            # Specification of service-level (ranges)
            m.addConstr(v[t,s] <= quicksum(I.xi_U[l]*z[t,l,s] for l in I.L), "rangeU_"+str((t,s)))
            m.addConstr(v[t,s] >= quicksum(I.xi_L[l]*z[t,l,s] for l in I.L), "rangeL_"+str((t,s)))
            
            # Binarization of z
            m.addConstr(quicksum(z[t,l,s] for l in I.L) == 1, "1_serv_"+str((t,s)))
            
            # Profit (budget balance)
            m.addConstr(pplus[t,s] == I.alpha + I.f[t] + quicksum((I.d[l,t]+I.k[l,t])*z[t,l,s] for l in I.L), "earn_"+str((t,s)))
            m.addConstr(pminus[t,s] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t,s], "spend_"+str((t,s)))
            m.addConstr(pminus[t,s] <= pdot[t,s] , "bud_"+str((t,s)))
    
    # Return
    for s in I.S:
        m.addConstr(quicksum(pplus[t,s] for t in I.T) <= (1+I.epsilon)*quicksum(pminus[t,s] for t in I.T), "return")
    
    # Risk measure
    if riskMeasure == 1:
        # Average
        for t in I.T:
            m.addConstr(quicksum(I.prob[s]*v[t,s] for s in I.S) >= I.minP, "minPerf_"+str(t))
           
    elif riskMeasure == 2:
        # For each scenario
        for s in I.S:
            for t in I.T:
                m.addConstr(v[t,s] >= I.minP, "minPerf_"+str((t,s)))
    
    m.update()
    m.setParam("OutputFlag",False)
    m.setParam("NumericFocus",True)
    m.setParam("DualReductions",0)
    m.optimize()

    if m.status == 2:
        X = [int(x[t].x) for t in I.T]
        V = {s:[v[t,s].x for t in I.T] for s in I.S}
        CASH = {s:[pdot[t,s].x for t in I.T] for s in I.S}
        EARNINGS = {s:[pplus[t,s].x for t in I.T] for s in I.S} 
        EXPENDITURES = {s:[pminus[t,s].x for t in I.T] for s in I.S} 
        OF = m.objVal
    else:
        X = []
        V = {}
        CASH = {}
        EARNINGS = {}
        EXPENDITURES = {}
        OF = '0'
        
    return(X,V,CASH,EARNINGS,EXPENDITURES,OF)

# For SAA proof
def PPPs_Evaluation(I, gamma, riskMeasure, x_opt):
    '''
    Parameters
    '''
    I.S = gamma.keys()
    I.prob = {s:1/len(I.S) for s in I.S}
    
    '''
    MODEL
    '''
    m = Model()
    
    '''
    VARIABLES
    '''
    y = {t:m.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}
    b = {(t,tau):m.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}
    z = {(t,l,s):m.addVar(vtype=GRB.BINARY, name="z_"+str((t,l,s))) for t in I.T for l in I.L for s in I.S}
    v = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="v_"+str((t,s))) for t in I.T for s in I.S}
    pplus = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str((t,s))) for t in I.T for s in I.S}
    pminus = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str((t,s))) for t in I.T for s in I.S}
    pdot = {(t,s):m.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str((t,s))) for t in I.T for s in I.S}
    slack1 = {t:m.addVar(vtype=GRB.BINARY, name="h1_"+str(t)) for t in I.T}
    slack2 = {(t,s):m.addVar(vtype=GRB.BINARY, name="h2_"+str((t,s))) for t in I.T for s in I.S}
    aux = {s:m.addVar(vtype=GRB.CONTINUOUS, name="aux_"+str(s)) for s in I.S}
    
    '''
    OBJECTIVE
    '''
    # Earnings-Expenditures
    m.setObjective(quicksum(I.prob[s]*quicksum(pplus[t,s]-pminus[t,s] for t in I.T) for s in I.S) - quicksum(I.cp[t]*(slack1[t]+quicksum(slack2[t,s] for s in I.S)) for t in I.T), GRB.MAXIMIZE)
    	
    '''
    CONSTRAINTS
    '''
    # Initialization
    m.addConstr(y[0] == 0, "iniY") 
    
    for s in I.S:    	
        m.addConstr(pdot[0,s] == pplus[0,s] - pminus[0,s], "iniCash_"+str(s)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            m.addConstr(y[t] == y[t-1] + 1 - y[t-1]*x_opt[t] - x_opt[t], "inv_"+str(t))
            
            for s in I.S:
                # Update available cash
                m.addConstr(pdot[t,s] == pdot[t-1,s] + pplus[t,s] - pminus[t,s], "cash_"+str((t,s)))
            
        # Binarization of y (to retrieve performance)
        m.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        m.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        for s in I.S:
            # Quantification of v (get performance)
            m.addConstr(v[t,s] == quicksum(gamma[s][tau]*b[t,tau] for tau in I.T), "quantV_"+str((t,s)))
            
            # Specification of service-level (ranges)
            m.addConstr(v[t,s] <= quicksum(I.xi_U[l]*z[t,l,s] for l in I.L), "rangeU_"+str((t,s)))
            m.addConstr(v[t,s] >= quicksum(I.xi_L[l]*z[t,l,s] for l in I.L), "rangeL_"+str((t,s)))
            
            # Binarization of z
            m.addConstr(quicksum(z[t,l,s] for l in I.L) == 1, "1_serv_"+str((t,s)))
            
            # Profit (budget balance)
            m.addConstr(pplus[t,s] == I.alpha + I.f[t] + quicksum((I.d[l,t]+I.k[l,t])*z[t,l,s] for l in I.L), "earn_"+str((t,s)))
            m.addConstr(pminus[t,s] == (I.cf[t]+I.cv[t])*x_opt[t]-I.cv[t]*v[t,s]*x_opt[t], "spend_"+str((t,s)))
    
    # Return
    for s in I.S:
        m.addConstr(aux[s] >= quicksum(pplus[t,s]-pminus[t,s] for t in I.T), "cash_"+str(s))
    
    # Risk measure
    if riskMeasure == 1:
        # Average
        for t in I.T:
            m.addConstr(quicksum(I.prob[s]*v[t,s] for s in I.S) + slack1[t] >= I.minP, "minPerf_"+str(t))
            
    elif riskMeasure == 2:
        # For each scenario
        for s in I.S:
            for t in I.T:
                m.addConstr(v[t,s] + slack2[t,s] >= I.minP, "minPerf_"+str((t,s)))
    
    m.update()
    m.setParam("OutputFlag",False)
    m.setParam("NumericFocus",True)
    m.optimize()

    if m.status == 2:
        AUX = [aux[s].x for s in I.S]
        OF = m.objVal
    else:
        AUX = []
        OF = '0'
        
    return(AUX,OF)

# Function
def PPPs_Function(I, gamma, riskMeasure, x_opt):
    '''
    Parameters
    '''
    I.S = gamma.keys()
    I.prob = {s:1/len(I.S) for s in I.S}   
   
    '''
    VARIABLES
    '''
    y = {t:0 for t in I.T}
    b = {(t,tau):0 for t in I.T for tau in I.T}
    z = {(t,l,s):0 for t in I.T for l in I.L for s in I.S}
    v = {(t,s):0 for t in I.T for s in I.S}
    pplus = {(t,s):0 for t in I.T for s in I.S}
    pminus = {(t,s):0 for t in I.T for s in I.S}
    pdot = {(t,s):0 for t in I.T for s in I.S}
    slack1 = {t:0 for t in I.T}
    slack2 = {(t,s):0 for t in I.T for s in I.S}
    aux = [0 for s in I.S]
    
    '''
    CONSTRAINTS
    '''
    y[0] = 0
    for t in I.T:
        if t>0:
            y[t] = y[t-1] + 1 - y[t-1]*x_opt[t] - x_opt[t]
        
        b[t,y[t]] = 1
        
        for s in I.S:
            v[t,s] = sum(gamma[s][tau]*b[t,tau] for tau in I.T)
            
            for l in I.L: 
                if v[t,s] <= I.xi_U[l] and v[t,s] >= I.xi_L[l]:
                    z[t,l,s] = 1
             
    for t in I.T:        
        for s in I.S:
            pplus[t,s] = I.alpha + I.f[t] + sum((I.d[l,t]+I.k[l,t])*z[t,l,s] for l in I.L)
            pminus[t,s] = (I.cf[t]+I.cv[t])*x_opt[t]-I.cv[t]*v[t,s]*x_opt[t]
    
    for s in I.S:
        pdot[0,s] = pplus[0,s] - pminus[0,s]
        
        for t in I.T:    
            if t>0:
                pdot[t,s] = pdot[t-1,s] + pplus[t,s] - pminus[t,s]
   
    for s in I.S:
        aux[s] = sum(pplus[t,s]-pminus[t,s] for t in I.T)
    
    # Risk measure
    if riskMeasure == 1:
        for t in I.T:
            slack1[t] = np.ceil(I.minP - sum(I.prob[s]*v[t,s] for s in I.S))
            
    elif riskMeasure == 2:
        for s in I.S:
            for t in I.T:
                slack2[t,s] = np.ceil(I.minP - v[t,s])
    
    '''
    OBJECTIVE
    '''
    # Earnings-Expenditures
    OF = sum(I.prob[s]*sum(pplus[t,s]-pminus[t,s] for t in I.T) for s in I.S) - sum(I.cp[t]*slack1[t] for t in I.T) - sum(I.prob[s]*sum(I.cp[t]*slack2[t,s] for t in I.T) for s in I.S)
    
    return(aux,OF)