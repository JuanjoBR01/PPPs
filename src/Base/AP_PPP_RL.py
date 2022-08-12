'''
An RL-Environment for the PPP

Our RL-agent is the "Agent" in the PA-Problem:
- acts through time horizon T
- state S \in (0,1] (may be discretized; 5/20 or 20/5)
- at each time:
	- a stochastic realization of deterioration occurs
	- a maintenance decision is made (x\in{0,1} OR x\in[0, 1-S_t])
	- a randomized inspection policy is executed (r<p_i)
	- "rewards" are computed for the agent (and the system)


'''
from math import log, exp
from random import random
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class PPP():

	def __init__(self):
		# Stoch Prog
		self.states = [i for i in range(20)]
		self.S = [0,1,2] 			# [age, perf, bud]
		self.X = [0,1] 				# available actions
		self.W = [False, True] 		# [shock?, inspection?] (currently deterministic)

		# General settings
		self.T = 30									# Planning horizon
		self.threshold = .6							# Performance threshold
		self.t_reach_half = 10						# Periods to midlife
		self.L = range(5)							# Discrete performance levels
		self.Q = range(self.t_reach_half)				# Set of periods to midlife	
		self.Lambda = -log(.5)/-self.t_reach_half 	# det rate		

		# Costs and earnings
		self.c_f = 10		# fixed cost of maintenance
		self.c_v = 1		# unit cost of maintenance
		self.a = 0			# agent's salary		
		self.epsilon = .3	# target agent's profit

		# DP
		self.gamma = .9
		self.f_W = [(1,[False, True])]


	'''
		function that governs the benefit and incentive rules
	'''
	def incentive(self, S, W, choice='sigmoid', show=False):
		age = S[0]
		perf = S[1]

		if choice=='sigmoid':
			rate, offset = 10, self.threshold
			incent = 1/( 1 + exp(-rate*(perf-offset)) )			

		elif choice=='linear':
			slope, offset = 1, 0
			incent = offset + slope*perf

		if show==True:			
			shape = [self.incentive(S=[0, i/100.0, 0], W=[], show=False) for i in range(100)]			
			plt.plot([offset, offset], [0,1], '--', color='k')
			plt.plot([i/100.0 for i in range(100)], shape)
			plt.suptitle("PPP - Incentive Scheme")
			plt.xlabel("Observed performance")
			plt.ylabel("Portion of cake transferred")
			plt.show()
		return incent
	

	'''
		function that governs system deterioration
	'''
	def deteriorate(self, S, W, show=False):
		age = S[0]		
		shock = W[0]

		Lambda = -log(self.threshold)/10   	# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt x
		
		if shock:
			delta_perf = 3*random()
		else:
			delta_perf = exp(-Lambda*age) - exp(-Lambda*(age-1))

		if show==True:			
			shape = [exp(-Lambda*t) for t in range(self.T)]			
			plt.plot([0,self.T], [self.threshold, self.threshold], '--', color='k')
			plt.plot(shape)
			plt.suptitle("System - Deterioration Model (null intervention)")
			plt.xlabel("Time")
			plt.ylabel("Performance")
			plt.show()

		return delta_perf

	'''
		functions for MDP "cost" and "transition"
	'''
	def cost(self, S, X, W):
		# pendiente que incentivo sea dependiente de inspeccion en W		
		return -self.c_f*X - self.c_v*X + 7*self.incentive(S, W)

	def transition(self, S, X, W):
		if X:
			self.S[0] = 0
			self.S[1] = 1

		delta_perf = self.deteriorate(S, W)
		bud = self.cost(S, X, W)

		self.S = [S[0]+1, S[1]+delta_perf, S[2]+bud]


	'''
		function to test fixed policies (arbitrary)
	'''
	def fixed_action_rule(self, kind='random_20'):
		if kind[:6]=='random':		
			print(int(kind[7:]))
			return round( random() < int(kind[7:])/100.0)
		elif kind=='always':
			return 1
		elif kind=='never':
			return 0

	'''
		function that simulates iterations of the system
	'''
	def run(self):
		performance = []
		cashflow = []
		for t in range(self.T):			
			X = self.fixed_action_rule('random_20')			
			self.transition(self.S, X, self.W)			
			performance.append(self.S[1])
			cashflow.append(self.S[2])

		plt.subplot(2,1,1)
		plt.plot(performance)
		plt.subplot(2,1,2)
		plt.plot(cashflow)
		plt.show()

	'''
		functions to apply Bellman's recursion
	'''
	def V(self, pi, S):		

		SM, C = self.transition, self.cost 		# for short		

		if type(pi)==dict:
			x = pi[S]
		else:
			x = pi(S)

		value = sum( w[0]*(C(S, x, w[1]) + self.gamma*self.v_hat[ SM(S, x, w[1]) ] ) for w in self.f_W )
		return value


	def V_bw(self, S, argmin):		

		SM, C = self.transition, self.cost 		# for short

		def Ew_Bellman(x):
			return sum( w[0]*( C(S, x, w[1]) + self.gamma*self.v_hat[ SM(S, x, w[1]) ] ) for w in self.f_W )
		
		cand = [(x, Ew_Bellman(x)) for x in self.X]

		x_best, value = min(cand, key = lambda h: h[1])
		
		if argmin:
			return x_best
		else:
			return value

	
myPPP = PPP()

myPPP.run()




