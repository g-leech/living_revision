import math
from random import random

# Approx constants
e = 2.71828
pi = 3.14159

# Operations on exponents
def neg_exp(a, x) :
assert ( a**-x ) == ( 1 / a**x )

def add_exps(a, x, y) :
assert (a**x * a**y) == (a**(x+y))

def subtract_exps(a, x, y) :
assert (a**x / a**y) == (a**(x-y))

def multiply_exps(a, x, y) :
assert (a**x)** y == (a**(x*y))


# Inefficient but surprising estimator for pi. n > 50m plz.
# Sample the unit square; increment if it lands in the circle sector (~79% chance)
def pi(n):  
    hits = sum([ _unit_monte_carlo() for i in range(1, n+1) ])
    return 4 * hits / n

def _unit_monte_carlo():
    x = random() * 2 - 1
    y = random() * 2 - 1 
    return 1 if (x**2 + y**2  < 1) else 0



def run_all( module ):
    import module
    for i in dir(module):
        item = getattr(module,i)
        if callable(item):
            item()

if __name__ == '__main__':
#run_all( math_defs )

