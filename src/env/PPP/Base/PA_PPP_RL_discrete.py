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

		'''
		Modelo de ciclos de deterioro
		self.t_reach_half = 12
		self.Q = range(self.t_reach_half)
		'''

		'''
		Parámetros necesarios para modelar el deterioro
		'''

		self.thres = 3										# performance threshold (discrete)
		self.fail = .2										# failure threshold (continuous)
		self.ttf = 10.0										# time to failure
		
		self.Lambda = -log(self.fail)/self.ttf				# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
		

		'''
		Épsilon del retorno
		Benefit and target profit
        self.epsilon = 1.4*100

		Parámetro f (¿?)
		Revisar paper exacto  }
        self.f ={0:0,1:135,2:0,3:0,4:0,5:83.8243786129859,6:0,7:0,8:0,9:0,10:52.0483440729868,11:0,12:0,13:0,14:0,15:32.3179266648371,16:0,17:0,18:0,19:0,20:20.0668897832594,21:0,22:0,23:0,24:0,25:12.4599597539036,26:0,27:0,28:0,29:0,30:7.73665469565768}

		Modelado del VDT - Discount Factor
		self.d = {}
        for i in range(1,6):
            b = 10.0-2*i
            for j in range(1,31):
                self.d[(i,j)]=(b/(1+0.1)**(j-1))
		

		# Minimum performance 
        self.minP = .6
		'''
		self.c_m = 1 										# cost of a maintenance action
		self.rate = 3 										# "slope" of sigmoidal benefit-performance function
		self.offset = 3 									# "offset" of sigmoidal benefit-performance function
		
		self.f_W = [(1, [False, False])] 					# scenarios: (prob_s, [shock, inspection])

		self.gamma = .9										# discount factor
		self.v_hat = {i:0 for i in range(int(self.ttf))} 	# state value estimation

		self.alpha = .5		

		'''
		Beneficio social asociado al nivel de performance. g_star es el deseado
		'''
		self.g = {5:2, 4:47, 3:500, 2:953, 1:998}
		# Earnings target
		self.g_star = 595

		'''
		Fixed income from the principal to the agent
		# self.a = 50
		'''


		'''
		Leader budget
        self.Beta = 5e8
		'''

		'''
		Queremos usar el bond o el sigmoidal del incentivo?
		        bond = {1: {5:-29.65, 4:-27.4, 3:-4.75, 2:17.9, 1:20.15},
                2:{5:-148.25, 4:-137, 3:-23.75, 2:89.5, 1:100.75},
                3:{5:-296.5, 4:-274, 3:-47.5, 2:179, 1:201.5},
                4:{5:-444.75, 4:-411, 3:-71.25, 2:268.5, 1:302.25}}

		1 va por default
        self.bond = bond[1]

		'''

		'''
		Fixed cost of inspection
        self.c_sup_i = 1
        c_sup_i = {1:50, 2:250, 3:70}
        self.c_sup_i = c_sup_i[INS]
		'''
	

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

		'''
			Samuel: recordar ajustar en el MIP
		'''

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
		S_prime = 0
		if X > .5:
			S_prime = 0	
		# TODO: Preguntar a qué hace referencia este condicional
		#elif S < int(self.ttf)-1:
		else:
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
		return current_policy


	'''
		function that simulates iterations of the system
	'''
	def run(self, opt=True):
		pi = self.policy_iteration()
		print(pi)
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



myPPP = PPP(30)
myPPP.run()
#myPPP.run(opt=True)

