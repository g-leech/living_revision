# Calculus
from sympy import *

f = Function('f')
g = Function('g')
h = Function('h')


def is_differential(eq) :
	def recur(sub, isFound) :
		for term in sub.args :
			if isinstance(term, Derivative) :
				return True
			elif term.args :			
				isFound = recur(term, isFound)
		return isFound
	return recur(eq, False) 



def search_expr(expr, target, isFound) :
	for term in expr.args :
		if isinstance(term, target) :
			return True
		elif term.args :			
			isFound = search_expr(term, target, isFound)

	return isFound


# TODO
# That is, linear in y.
def is_first_order(eq) :
	for term in eq.lhs.args:
		print(term.coeff(x))
	return

# TODO
def is_linear(eq) :
	#LHS
	eq.lhs
	#RHS

	return 

# TODO
def is_homogeneous() :

	return


# Linear examples
x, y = Symbols('x y')
isLinear = Eq(Derivative(y, x) - (x**2)*y, x**3)
isntLinear = Eq(Derivative(y, x), x * y**2)
assert is_linear(isLinear)
assert not is_linear(isntLinear)

# TODO
# linear
def integrating_factor( eq ):
	if not is_linear(eq):
		raise ValueError("Integrating factor only works on LDEs")

	return 



