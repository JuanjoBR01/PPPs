'''
PPP Environment

A principal-agent setting for the maintenance of a deteriorating system

- transition: updates 'time since last maintenance' based on deterioration
- 

'''

from math import exp, log
from random import random
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class PPP():

	def __init__(self, Tau): #Lambda=, offset=, rate=
		self.S = 0 											# current state (for when running)
		self.X = [0, 1]										# available actions
		self.L = range(5)									# discrete levels
		self.T = range(Tau) 								# time horizon (if any)

		self.thres = 3										# performance threshold (discrete)
		self.fail = .2										# failure threshold (continuous)
		self.ttf = 10.0										# time to failure
		
		self.Lambda = -log(self.fail)/self.ttf				# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
		
		self.c_m = 1 										# cost of a maintenance action
		self.rate = 3 										# "slope" of sigmoidal benefit-performance function
		self.offset = 3 									# "offset" of sigmoidal benefit-performance function
		
		self.f_W = [(1, [False, False])] 					# scenarios: (prob_s, [shock, inspection])

		self.gamma = .9										# discount factor
		self.v_hat = {i:0 for i in range(int(self.ttf))} 	# state value estimation

		self.alpha = .5		
	

	'''
		show a summary of main models: deterioration and incentives
	'''
	def show(self):
		plt.subplot(211)		
		plt.plot([self.fail for t in self.T], '--', color='k')
		plt.plot([exp(-self.Lambda*t) for t in self.T], color='r')
		plt.title("Deterioration model of the physical system")
		plt.ylabel("Performance")
		plt.xlabel("Time (years)")

		plt.subplot(212)
		plt.plot([self.offset*(.2) for level in self.L], '--', color='k')
		plt.plot([level/10.0 for level in range(41)],[1/( 1 + exp(-self.rate*(level/10.0-self.offset)) ) for level in range(41)], color='r')
		plt.title("Economic model of the contractual relationship")
		plt.ylabel("Benefit")
		plt.xlabel("Performance")

		plt.show()


	'''
		get (0,1) performance based on state (time since last maintenance)
	'''
	def get_perf(self, S):
		perf = exp(-self.Lambda*S)
		return perf


	'''
		get discrete state (level) from (0,1) performance
	'''
	def get_level(self, perf):

		if perf < .2:
			return 0
		elif perf < .4:
			return 1
		elif perf < .6:
			return 2
		elif perf < .8:
			return 3
		else:
			return 4


	'''
		get sigmoid-based incentive from discrete state (level)
	'''
	def get_incentive(self, level):		
		incent = 1/( 1 + exp(-self.rate*(level-self.offset)) )
		return incent


	'''
		apply maintenance OR increase time since last maintenance
	'''
	def transition(self, S, X=0, W=[False, False]):
		
		if X > .5:
			S_prime = 0	
		elif S < int(self.ttf)-1:
			S_prime += 1

		return S_prime
		

	'''
		compute cost (incentive - maintenance_cost)
	'''
	def cost(self, S, X=0, W=[False, False]):

		perf = self.get_perf(S)
		level = self.get_level(perf)
		incentive = self.get_incentive(level)
		
		total = (self.alpha)*incentive - self.c_m*X
		return -total


	'''
		function to test fixed policies (arbitrary)
	'''
	def fixed_action_rule(self, S, kind='random_20'):
		if kind[:6]=='random':			
			return round( random() < int(kind[7:])/100.0)
		elif kind=='always':
			return 1
		elif kind=='never':
			return 0
		elif kind[:5]=='reach':			
			if S>=int(kind[6:]):
				return 1
			else:
				return 0


	'''
		Bellman's recursion for a GIVEN policy
	'''
	def V(self, pi, S):		

		SM, C = self.transition, self.cost 		# for short		

		if type(pi)==dict:
			x = pi[S]
		else:
			x = pi(S)

		value = sum( w[0]*(C(S, x, w[1]) + self.gamma*self.v_hat[ SM(S, x, w[1]) ] ) for w in self.f_W )
		return value


	'''
		Bellman's recursion to find a policy
	'''
	def V_bw(self, S, arg):		

		SM, C = self.transition, self.cost 		# for short

		def Ew_Bellman(x):
			return sum( w[0]*( C(S, x, w[1]) + self.gamma*self.v_hat[ SM(S, x, w[1]) ] ) for w in self.f_W )
		
		cand = [(x, Ew_Bellman(x)) for x in self.X]

		x_best, value = min(cand, key = lambda h: h[1])
		
		if arg:
			return x_best
		else:
			return value


	'''
		policy iteration
	'''
	def policy_iteration(self):
		self.v_hat = {i:0 for i in range(int(self.ttf))}
		policy_is_changing = True
		current_policy = {i:0 for i in range(int(self.ttf))}

		while policy_is_changing:

			# Cycle to EVALUATE current policy
			while True:
				error = 0
				for i in range(int(self.ttf)):
					old_val = self.v_hat[i]
					self.v_hat[i] = self.V(current_policy, i)
					error = max(error, abs(self.v_hat[i]-old_val) )
				if error < 1e-12:
					break
			
			# Cycle to UPDATE current policy
			policy_is_changing = False
			for i in range(int(self.ttf)):
				old_action = current_policy[i]
				current_policy[i] = self.V_bw(i, True) 	# True: returns x_best
													# The trick: V_bw uses a new 'v_hat'
				if old_action != current_policy[i]:
					policy_is_changing = True
		print (current_policy)
		return current_policy


	'''
		function that simulates iterations of the system
	'''
	def run(self, opt=True):
		pi = self.policy_iteration()
		performance = []
		cashflow = []
		total = 0
		for t in self.T:
			if opt:
				X = pi[self.S]
			else:
				X = self.fixed_action_rule(S=self.S, kind='reach_8')

			self.S = self.transition(self.S, X)
			performance.append( self.S )
			c = self.cost(self.S, X)
			total -= c
			cashflow.append( -c )

		print("Cumulative earning: " + str(total))

		plt.subplot(2,1,1)
		plt.plot(performance)
		plt.title("performance over time")

		plt.subplot(2,1,2)
		plt.plot(cashflow)
		plt.title("cashflow over time")
		plt.show()



myPPP = PPP()
myPPP.show()
#myPPP.run(opt=True)

