# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:40:53 2020

@author: n.torres11
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def performance(T,S,v,minP):
    plt.plot(T, [np.mean([v[s][t] for s in S]) for t in T], label = "Maintenance",color = 'g', linewidth = 1)
    plt.plot(T, np.repeat(minP,len(T)), label = "$\gamma^*$", color = 'k', linewidth = 1)
    plt.xlim(0.0,len(T))
    plt.ylim(-0.02,1.02)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Performance")
    plt.title("Life-cycle of the system")
    plt.show()

def cashFlow(T,S,earnings,expeditures):
    plt.bar(T, [np.mean([earnings[s][t] for s in S]) for t in T], 0.5, color = 'g', label = 'Earnings')
    plt.bar(T, [np.mean([-expeditures[s][t] for s in S]) for t in T], 0.5, color = 'r', label = 'Expenditures')
    plt.xlim(0.0-0.5,len(T)+0.5)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Cash ($)")
    plt.title("Cash-flow of the project")
    plt.show()

def meanPerformance(T,S,v,minP):
    plt.bar(S, [np.mean(v[s]) for s in S], color = 'b')
    plt.plot(S, np.repeat(minP,len(S)), linewidth = 0.75, color = 'k')
    plt.ylim(0.0,len(S))
    plt.ylim(0.0,1.0)
    plt.xlabel("Scenario")
    plt.ylabel("Performance")
    plt.title("Average performance")
    plt.show()
    
def budget(T,S,cash):
    plt.boxplot({t:[cash[s][t] for s in S] for t in T}.values())
    plt.xlim(0.0,len(T)+0.5)
    plt.xlabel("Time (years)")
    plt.xticks(list(range(0,len(T)+1,5)),list(range(0,len(T)+1,5)))
    plt.ylabel("Cash ($)")
    plt.title("Available budget of the project")
    plt.show()

def solutions(T,solRN,solRA,minP):
    plt.plot(T, [solRN[t] for t in T], label = "Risk $\it{neutral}$",color = 'darkgreen', linewidth = 1)
    plt.plot(T, [solRA[t] for t in T], label = "Risk $\it{averse}$",color = 'darkred', linewidth = 1)
    plt.plot(T, np.repeat(minP,len(T)), label = "$\gamma^*$", color = 'k', linewidth = 1)
    plt.xlim(0.0,len(T))
    plt.ylim(-0.02,1.02)
    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Performance")
    plt.show()