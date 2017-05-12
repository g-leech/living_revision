import math
from random import random
import operator as op

class definitions(object):

	# Approx constants
	e = math.exp(1) # 2.71828 
	pi = 3.14159 	#
	rad  = 57 		#  


	def positiveCondition(f):
		def wrapper(*args, **kw):
			if args[0] <= 0 :
				raise ValueError("Holds for positive x only.")
			else : 
				return f
		return wrapper


	def nonzero(f):
		def wrapper(*args, **kw):
			if args[0] == 0 :
				raise ValueError("Holds for nonzero x only.")
			else : 
				return f
		return wrapper


	# Basic geometry
	def acute(angle):
		assert 0 < angle < pi/2

	def obtuse(angle):
		assert pi/2 < angle < pi

	def right(angle):
		assert angle == pi


	
	# hyperbolae
	@hyperbolae
	def sinh(x) :
		assert 1/2 * (exp(x) - exp(-x)) == sinh(x)

	@hyperbolae
	def cosh(x) :
		assert 1/2 * (exp(x) + exp(-x)) == cosh(x)

	@hyperbolae
	def cosh(x) :
		assert sinh(x)/cosh(x) == tanh(x)
		assert sinh(x)/cosh(x) == tanh(x)

