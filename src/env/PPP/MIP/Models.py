from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import *
import PPP_Env_V1
from matplotlib import style
style.use("ggplot")

def get_level(perf):

    if perf < .2:
        return 1
    elif perf < .4:
        return 2
    elif perf < .6:
        return 3
    elif perf < .8:
        return 4
    else:
        return 5

I = PPP_Env_V1.EnvPPP()

'''perf_tt = exp(-I.Lambda*I.ttf)

gamma = [round(exp(-I.Lambda*tau),2) for tau in range(I.T)]

print(gamma)

incentive_val = [I.incentive(S=[0,gamma_val], W = I.W) for gamma_val in gamma]

bond = {}

for level in I.L:
    average_l = 0
    count_l = 0
    for gamma_val in gamma:
        if get_level(gamma_val) == level:
            average_l += I.incentive(S=[0,gamma_val], W = I.W) 
            count_l += 1
    bond[level] = average_l/count_l

print(bond)

aaaaa
'''
x_param = {'q_'+str(i-1):1 if (i % 5 == 0 and i>0) else 0 for i in range(1,I.T+1)}

#print(x_param)

def follower_PPP(I,x_param):
    
    gamma = [exp(-I.Lambda*tau) for tau in range(I.T)]
    print(gamma)
    
    '''delta = [exp(-I.Lambda*tau) - exp(-I.Lambda*(tau-1)) for tau in range(1,I.T)]
                
                gamma_2 = [1]
                for tau in range(1,I.T):
                    gamma_2.append(gamma_2[tau-1]+delta[tau-1])
            
                print(gamma_1==gamma_2)
                aaaaa
                '''

    fc = [I.FC for _ in range(I.T)]
    vc = [I.VC for _ in range(I.T)]

    xi_L = {1:0, 2:.21, 3:.41, 4:.61, 5:.81}
    xi_U = {1:.2, 2:.4, 3:.6, 4:.8, 5:1}   
    bond = {}
    for level in I.L:
        average_l = 0
        count_l = 0
        for gamma_val in gamma:
            if get_level(gamma_val) == level:
                average_l += 7*I.incentive(gamma_val) 
                count_l += 1
        bond[level] = average_l/count_l

    print(bond)
    Follower = Model('Follower_PPP')
    
    '''
    FOLLOWER VARIABLES
    '''
   
    x = {t:Follower.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in range(I.T)}                             # Whether a maintenance action is applied at t
    y = {t:Follower.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in range(I.T)}					             # Number of periods after last restoration
    b = {(t,tau):Follower.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in range(I.T) for tau in range(I.T)}    # Whether yt=tau
    z = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in range(I.T) for l in I.L}		      # Whether system is at service level l at t
    v = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in range(I.T)}							# Performance at t
    pplus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in range(I.T)}				# Earnings at t
    pminus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in range(I.T)}				# Expenditures at t
    pdot = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in range(I.T)}				# Money at t
    w = {t:Follower.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in range(I.T)}							# Linearization of y*x
    u = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in range(I.T)}						# Lineartization for v*x
    aux = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in range(I.T) for l in I.L}              # variable for linearization ztl*qt
    Follower.update()
    '''
    OBJECTIVE
    '''
    #Follower Objective
    Follower.setObjective(-quicksum(pplus[t]-pminus[t] for t in range(I.T)), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    Follower.addConstr(y[0] == 0, "iniY") 
    Follower.addConstr(w[0] == 0, "iniW") 	
    Follower.addConstr(u[0] == 0, "iniU") 
    #Follower.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    Follower.addConstr(pdot[0] == I.S[2], "cash_"+str(0))
    
    for t in range(I.T):
        if t>0:   
            # Restoration inventory
            Follower.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            Follower.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            Follower.addConstr(w[t] >= y[t-1] - I.T*(1-x[t]), "linW2_"+str(t))
            Follower.addConstr(w[t] <= I.T*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            Follower.addConstr(u[t] <= v[t], "linU1_"+str(t))
            Follower.addConstr(u[t] >= v[t] - (1-x[t]), "linU2_"+str(t))
            Follower.addConstr(u[t] <= x[t], "linU3_"+str(t))
            
            # Update available cash
            Follower.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        Follower.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in range(I.T)), "binY1_"+str(t))
        Follower.addConstr(quicksum(b[t,tau] for tau in range(I.T)) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        Follower.addConstr(v[t] == quicksum(gamma[tau]*b[t,tau] for tau in range(I.T)), "quantV_"+str(t))
        
        # Linearization for service level
        Follower.addConstr(v[t] <= quicksum(xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        Follower.addConstr(v[t] >= quicksum(xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        Follower.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        Follower.addConstr(pminus[t] == (fc[t]+vc[t])*x[t]-vc[t]*u[t], "spend_"+str(t))
        Follower.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return IMPORTANTEEEEEEEE
    #Follower.addConstr(quicksum(pplus[t] for t in range(I.T)) >= (1+1.4*100)*quicksum(pminus[t] for t in range(I.T)), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in range(I.T):
        for l in I.L:
            Follower.addConstr(aux[t,l] <= x_param["q_"+str(t)], name = "binaux1_"+str((t,l)))
            Follower.addConstr(aux[t,l] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(aux[t,l] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in range(I.T):
        Follower.addConstr(pplus[t] == quicksum(aux[t,l]*bond[l] for l in I.L), name = "Agents_earnings_"+str(t)) #I.a
    '''
    for t in I.T:
        for l in I.L:
            Follower.addConstr(x_param["aux_"+str((t,l))] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(x_param["aux_"+str((t,l))] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        Follower.addConstr(pplus[t] == I.a + quicksum(x_param["q_"+str(t)]*z[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    '''

    thyMaintenance = {'x_0': 1, 'x_1': 1, 'x_2': 1, 'x_3': 1, 'x_4': 1, 'x_5': 1, 'x_6': 1, 'x_7': 1, 'x_8': 1, 'x_9': 1, 'x_10': 1, 'x_11': 1, 'x_12': 1, 'x_13': 1, 'x_14': 1, 'x_15': 1, 'x_16': 1, 'x_17': 1, 'x_18': 1, 'x_19': 1, 'x_20': 1, 'x_21': 1, 'x_22': 1, 'x_23': 1, 'x_24': 1, 'x_25': 1, 'x_26': 1, 'x_27': 1, 'x_28': 1, 'x_29': 1}
    for name in thyMaintenance.keys():
         Follower.addConstr(Follower.getVarByName(name) == thyMaintenance[name])

    Follower.update()
    
    return Follower

private = follower_PPP(I,x_param)

private.optimize()

if private.status == 2:

    print(f'Objective: {private.objVal}')
    
    # for v in private.getVars():
    #     print(f'name: {v.VarName}, value: {v.x}')

    inspection = [x_param[i] for i in x_param.keys()]
    maintenance = [i.x for i in private.getVars() if i.VarName[0] == "x"]
    performance = [i.x for i in private.getVars() if i.VarName[0] == "v"]

    PPP_metrics = {"Inspection": inspection,
                        "Maintenance": maintenance,
                        "Performance": performance}
    df = pd.DataFrame(PPP_metrics, columns = PPP_metrics.keys())

    fig = plt.figure(figsize =(20, 10))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    corr = 0.4
    Inspection = {t:(t,df["Inspection"][t]) for t in range(df.shape[0]) if df["Inspection"][t] == 1}
    Maintenance = {t:(t,df["Maintenance"][t]) for t in range(df.shape[0]) if df["Maintenance"][t] == 1}
    ax.plot(range(df.shape[0]), df["Performance"], 'k--', linewidth = 1.5, label = "Performance")
    ax.plot([Inspection[t][0] for t in Inspection], [Inspection[t][1] for t in Inspection], 'rs', label = "Inspection actions")
    ax.plot([Maintenance[t][0] for t in Maintenance], [Maintenance[t][1] for t in Maintenance], 'b^', label = "Maintenance actions")
    ax.set_xlabel("Period", size=15)
    ax.set_ylabel("Road's Performance", size=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    plt.suptitle("Private's perspective" , fontsize=15)
    plt.grid(True)
    #plt.savefig('Leader perspective.png')
    plt.show()
    # plt.close(fig)


def HPR_PPP(I):
    
    HPR = Model('HPR_PPP')
    
    #HPR.setParam("NumericFocus",True)
    '''
    LEADER VARIABLES
    '''
    q = {t:HPR.addVar(vtype=GRB.BINARY, name="q_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    #r = {(k,l):HPR.addVar(vtype=GRB.BINARY, name="r_"+str((k,l))) for k in I.K for l in I.L}                             # Whether reward k in K is given at service level l in L
    '''
    FOLLOWER VARIABLES
    '''
    x = {t:HPR.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    y = {t:HPR.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}					             # Number of periods after last restoration
    b = {(t,tau):HPR.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}    # Whether yt=tau
    z = {(t,l):HPR.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in I.T for l in I.L}		      # Whether system is at service level l at t
    v = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in I.T}							# Performance at t
    pplus = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in I.T}				# Earnings at t
    pminus = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in I.T}				# Expenditures at t
    pdot = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in I.T}				# Money at t
    w = {t:HPR.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}							# Linearization of y*x
    u = {t:HPR.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in I.T}						# Lineartization for v*x
    #m = {(k,l,t):HPR.addVar(vtype=GRB.BINARY, name="m_"+str((k,l,t))) for k in I.K for l in I.L for t in I.T}                             # Linearization of z*r
    aux = {(t,l):HPR.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in I.T for l in I.L}		      # variable for linearization ztl*qt
    
    '''
    OBJECTIVE
    '''
    #Leader objective
    HPR.setObjective(-quicksum(I.g[l]*z[t,l] for l in I.L for t in I.T) + quicksum(q[t]*I.c_sup_i for t in I.T) + quicksum(aux[t,l]*I.bond[l] for t in I.T for l in I.L) + I.a*len(I.T), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    HPR.addConstr(y[0] == 0, "iniY") 
    HPR.addConstr(w[0] == 0, "iniW") 	
    HPR.addConstr(u[0] == 0, "iniU") 
    HPR.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            HPR.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            HPR.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            HPR.addConstr(w[t] >= y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            HPR.addConstr(w[t] <= len(I.T)*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            HPR.addConstr(u[t] <= v[t], "linU1_"+str(t))
            HPR.addConstr(u[t] >= v[t] - (1-x[t]), "linU2_"+str(t))
            HPR.addConstr(u[t] <= x[t], "linU3_"+str(t))
            
            # Update available cash
            HPR.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        HPR.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        HPR.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        HPR.addConstr(v[t] == quicksum(I.gamma[tau]*b[t,tau] for tau in I.T), "quantV_"+str(t))
        
        # Linearization for service level
        HPR.addConstr(v[t] <= quicksum(I.xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        HPR.addConstr(v[t] >= quicksum(I.xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        HPR.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        HPR.addConstr(pminus[t] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t], "spend_"+str(t))
        HPR.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return
    HPR.addConstr(quicksum(pplus[t] for t in I.T) >= (1+I.epsilon)*quicksum(pminus[t] for t in I.T), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in I.T:
        for l in I.L:
            HPR.addConstr(aux[t,l] <= q[t], name = "binaux1_"+str((t,l)))
            HPR.addConstr(aux[t,l] <= z[t,l], name = "binaux2_"+str((t,l)))
            HPR.addConstr(aux[t,l] >= q[t] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        HPR.addConstr(pplus[t] == I.a + quicksum(aux[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    
    '''
    LEADER CONSTRAINTS
    '''
    
    #Leader budget
    #HPR.addConstr(quicksum(q[t]*I.c_sup_i for t in I.T) <= I.Beta, "Leader_budget")
    
    #Minimum social profit
    for t in I.T:
        HPR.addConstr(quicksum(I.g[l]*z[t,l] for l in I.L) >= I.g_star, name = "social_profit_" + str(t))
    
    
    HPR.update()

    return HPR

def HPR_PPP_std(I):
    
    HPR_std = Model('HPR_PPP_std')
    
    #HPR.setParam("NumericFocus",True)
    '''
    LEADER VARIABLES
    '''
    q = {t:HPR_std.addVar(vtype=GRB.BINARY, name="q_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    #r = {(k,l):HPR.addVar(vtype=GRB.BINARY, name="r_"+str((k,l))) for k in I.K for l in I.L}                             # Whether reward k in K is given at service level l in L
    '''
    FOLLOWER VARIABLES
    '''
    x = {t:HPR_std.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in I.T}                             # Whether a maintenance action is applied at t
    y = {t:HPR_std.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in I.T}					             # Number of periods after last restoration
    b = {(t,tau):HPR_std.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in I.T for tau in I.T}    # Whether yt=tau
    z = {(t,l):HPR_std.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in I.T for l in I.L}		      # Whether system is at service level l at t
    v = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in I.T}							# Performance at t
    pplus = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in I.T}				# Earnings at t
    pminus = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in I.T}				# Expenditures at t
    pdot = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in I.T}				# Money at t
    w = {t:HPR_std.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in I.T}							# Linearization of y*x
    u = {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in I.T}						# Lineartization for v*x
    #m = {(k,l,t):HPR.addVar(vtype=GRB.BINARY, name="m_"+str((k,l,t))) for k in I.K for l in I.L for t in I.T}                             # Linearization of z*r
    aux = {(t,l):HPR_std.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in I.T for l in I.L}		      # variable for linealization zlt*qt
    
    '''
    OBJECTIVE
    '''
    #Leader objective
    HPR_std.setObjective(-quicksum(I.g[l]*z[t,l] for l in I.L for t in I.T) + quicksum(q[t]*I.c_sup_i for t in I.T) + quicksum(aux[t,l]*I.bond[l] for t in I.T for l in I.L) + I.a*len(I.T), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    HPR_std.addConstr(y[0] == 0, "iniY") 
    HPR_std.addConstr(w[0] == 0, "iniW") 	
    HPR_std.addConstr(u[0] == 0, "iniU") 
    HPR_std.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    
    for t in I.T:
        if t>0:   
            # Restoration inventory
            HPR_std.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW1_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW2_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linW3_"+str(t))}
            HPR_std.update()
            HPR_std.addConstr(w[t] + HPR_std.getVarByName("s_linW1_"+str(t)) == y[t-1], "linW1_"+str(t))
            HPR_std.addConstr(w[t] - HPR_std.getVarByName("s_linW2_"+str(t)) == y[t-1] - len(I.T)*(1-x[t]), "linW2_"+str(t))
            HPR_std.addConstr(w[t] + HPR_std.getVarByName("s_linW3_"+str(t)) == len(I.T)*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU1_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU2_"+str(t))}
            {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_linU3_"+str(t))}
            HPR_std.update()
            HPR_std.addConstr(u[t] + HPR_std.getVarByName("s_linU1_"+str(t)) == v[t], "linU1_"+str(t))
            HPR_std.addConstr(u[t] - HPR_std.getVarByName("s_linU2_"+str(t)) == v[t] - (1-x[t]), "linU2_"+str(t))
            HPR_std.addConstr(u[t] + HPR_std.getVarByName("s_linU3_"+str(t)) == x[t], "linU3_"+str(t))
            
            # Update available cash
            HPR_std.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= I.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        HPR_std.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in I.T), "binY1_"+str(t))
        HPR_std.addConstr(quicksum(b[t,tau] for tau in I.T) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        HPR_std.addConstr(v[t] == quicksum(I.gamma[tau]*b[t,tau] for tau in I.T), "quantV_"+str(t))
        
        # Linearization for service level
        HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_rangeU_"+str(t))
        HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_rangeL_"+str(t))
        HPR_std.update()
        HPR_std.addConstr(v[t] + HPR_std.getVarByName("s_rangeU_"+str(t)) == quicksum(I.xi_U[l]*z[t,l] for l in I.L), "rangeU_"+str(t))
        HPR_std.addConstr(v[t] - HPR_std.getVarByName("s_rangeL_"+str(t)) == quicksum(I.xi_L[l]*z[t,l] for l in I.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        HPR_std.addConstr(quicksum(z[t,l] for l in I.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == I.alpha + I.f[t] + quicksum((I.d[l,t+1]+I.k[t])*z[t,l] for l in I.L), "earn_"+str(t))
        HPR_std.addConstr(pminus[t] == (I.cf[t]+I.cv[t])*x[t]-I.cv[t]*u[t], "spend_"+str(t))
        HPR_std.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return
    HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_return")
    HPR_std.update()
    HPR_std.addConstr(quicksum(pplus[t] for t in I.T) - HPR_std.getVarByName("s_return") == (1+I.epsilon)*quicksum(pminus[t] for t in I.T), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in I.L) linealization
    for t in I.T:
        for l in I.L:
            
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux1_"+str((t,l)))}
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux2_"+str((t,l)))}
            {(t,l):HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_binaux3_"+str((t,l)))}
            HPR_std.update()
            HPR_std.addConstr(aux[t,l] + HPR_std.getVarByName("s_binaux1_"+str((t,l))) == q[t], name = "binaux1_"+str((t,l)))
            HPR_std.addConstr(aux[t,l] + HPR_std.getVarByName("s_binaux2_"+str((t,l))) == z[t,l], name = "binaux2_"+str((t,l)))
            HPR_std.addConstr(aux[t,l] - HPR_std.getVarByName("s_binaux3_"+str((t,l))) == q[t] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in I.T:
        HPR_std.addConstr(pplus[t] == I.a + quicksum(aux[t,l]*I.bond[l] for l in I.L), name = "Agents_earnings_"+str(t))
    
    '''
    LEADER CONSTRAINTS
    '''
    
    #Leader budget
    #HPR.addConstr(quicksum(q[t]*I.c_sup_i for t in I.T) <= I.Beta, "Leader_budget")
    
    #Minimum social profit
    for t in I.T:
        {t:HPR_std.addVar(vtype=GRB.CONTINUOUS, name="s_social_profit_"+str(t))}
        HPR_std.update()
        HPR_std.addConstr(quicksum(I.g[l]*z[t,l] for l in I.L) - HPR_std.getVarByName("s_social_profit_"+str(t)) == I.g_star, name = "social_profit_" + str(t))
    
    
    HPR_std.update()
    
    return HPR_std

