from math import *
import operator as op

scoldString = "Holds only if x is "


def condition(i, op, comparand):
	def decor(f):
		def wrapper(*args, **kwargs):
			if op(args[i], comparand) :
				return f(*args, **kwargs)
			else : 
				raise ValueError(scoldString + op.__name__ + " "+ str(comparand))
		
		return wrapper
	
	return decor


@condition(0, op.gt, 0)
def subtract_logs(x, y, base) :
	log_b_X = log(x, base)
	log_b_Y = log(y, base)
	assert log(x/y, base) == (log_b_X - log_b_Y)


# On Exponentiation. 
def exponents(f):
	f.exponentiation = True
	return f


@exponents
def identity_exp(base) :
	assert ( base ** 1 ) == ( base )


# y-intercept of exp(x), exp(0)
@exponents
def zero_exp(base) :
	assert ( base ** 0 ) == 1


@exponents
def neg_exp(base, exponent) :
	assert ( base **-exponent ) == ( 1 / base ** exponent )


@exponents
def add_exps(x, y, base) :
	assert (base**x * base**y) == (base**(x+y))


@exponents
def subtract_exps(x, y, base) :
	assert (base**x / base**y) == (base**(x-y))


@exponents
def multiply_exps(x, y, base) :
	assert (base**x)** y == (base**(x*y))


# The log laws. 
@logarithms
def add_logs(x, y, base) :
	log_b_X = log(x, base)
	log_b_Y = log(y, base)
	assert log(x*y, base) == (log_b_X + log_b_Y)


@logarithms
@positiveCondition
def subtract_logs(x, y, base) :
	log_b_X = log(x, base)
	log_b_Y = log(y, base)
	assert log(x/y, base) == (log_b_X - log_b_Y)


@logarithms
#@positiveCondition
@condition("arg[0] > 0")
def scalar_log(x, n, base) :
	log_b_Xn = log(x**n, base)
	assert log_b_Xn == n * log(x, base)


@logarithms
def identity_log(base) :
	assert log(base, base) == 1


# x-intercept of log(x) is 1
@logarithms
def zero_log(base) :
	assert log(1, base) == 0

@logarithms
def ln(x) :
	assert log(x) == log(x, exp(1))
