# Various functions to numerically optimise various things.
# There's a relatively low limit to raw Python's bignums/precision
# (see `approx_exp` for instance)
# so don't use this to go to space or do surgery or whatnot.


from math import sqrt, floor, exp
import re
from locale import localeconv
import numpy as np
import matplotlib.pyplot as plt



def pythagoras_distance(x1, y1, x2, y2) :
    squaredXDiff = (x1 - x2)**2
    squaredYDiff = (y1 - y2)**2
    
    return sqrt(squaredXDiff + squaredYDiff)


def quadratic_equation(a, b, c) :
    if a == 0: raise Exception("Can't have zero x^2 coefficient")
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0: raise Exception("No real solutions")
    
    firstRoot = (-b + sqrt(discriminant)) / (2 * a)
    secondRoot = (-b - sqrt(discriminant)) / (2 * a)
    
    return firstRoot, secondRoot


def count_decimal_places(foo) :
    foo = str(foo)
    dec_pt = localeconv()['decimal_point']
    decrgx = re.compile("\d+(%s\d+)?e(-|\+)(\d+)" % dec_pt)
    if decrgx.search(foo):
        raise NotImplementedError( "e notation not implemented")
    else:
        return len(foo.split(dec_pt)[-1])
    

def relative_error( exact, estimate ) :
    return abs( (estimate - exact) / exact )


def round_relative_error_bound( precision ) :
    return 5 * 10**-(precision)



#####
# 1.1.2

def plot_func_and_interval(a, b, f, trim=True, plotNow=True) :
    x = np.linspace(a/2,b*2, 100)
    ax = plt.axes()
    ax.plot(x,  f(x) )
    if trim:
        ax.set_ylim(ymin=0)
        
    arbitrary = 32
    
    ax.plot([a]*arbitrary, range(arbitrary), linestyle="dotted")
    ax.plot([b]*arbitrary, range(arbitrary), linestyle="dotted")
    
    if plotNow:
        plt.show()


# Changed sign between c and d: therefore crossed y=0
def are_opposite_signs(c,d) :
    return floor(c) ^ floor(d) < 0


def binary_search(a, b, f, precision=6) :
    fA, fB = f(a), f(b)
    if not are_opposite_signs(fA, fB) :
        print("Bad start, sign of f({}) == sign of f({})".format(a, b))
        return a, b
        
    mid = (a+b) / 2
    halfIntervals = [(a, mid), (mid, b)]
    fMid = f(mid)
    
    if round(fMid, precision) == 0 :
        return a,b
    if are_opposite_signs(fA, fMid) :
        return binary_search(a, mid, f)
    if are_opposite_signs(fB, fMid) :
        return binary_search(mid, b, f)    


def resolve_last_place(l, r, f) :
    if l != r :
        mid = (l+r) / 2
        fMid = f(mid)
        if are_opposite_signs(f(l), fMid) :
            return l
        if are_opposite_signs(fMid, f(r)) :
            return r

    return l


def how_many_bisections(l, r, precision) :
    return (np.log10(r - l) + precision) \
            / np.log10(2) \


# k 
# +2 for the initial evals
# +1 for the final rounded check
def bisection_effort(l, r, precision) :
    evalsPerBisection = 1
    bisects = how_many_bisections(l,r,precision)
    
    return evalsPerBisection * bisects + 3
    
    

################
# SIMPLE ITERATION
#
def simply_converge_on_precision(f, it, precision) :
    nex = 0
    while round(it, precision+1) != round(nex, precision+1) :
        it, nex = f(it), it
        print(nex)
    print()
    return nex


def run_fixed_simple_iterations(f, estimate, iters) :
    for i in range(iters):
        print(estimate)
        estimate = f(estimate)
    print()
    

def simple_it(g, x, tol, error=100):
    i = 0
    while error >= tol :
        old = x
        x = g(x)
        error = x - old
        i += 1
        #print(x)
    print("Took", i, "iterations")
    
    return x
    
    
###############################
# NEWTON
# 
def newton_raphson_delta(f, df, x0, error=1e-5):
    def dx(f, x):
        return abs(0 - f(x))

    delta = dx(f, x0)
    
    while delta > error :
        x0 = x0 - f(x0) / df(x0)
        delta = dx(f, x0)
    
    return x0, f(x0)


def newton_raphson(it, f, fPrime, precision=3) :
    nex = 0
    while round(it, precision) != round(nex, precision) :
        step = it - (f(it) / fPrime(it))
        it, nex = step, it
        print(it)
    
    return it



# Recursive convergent series for exp
# Kinda clumsy
# e^x = 1 + x/1! + x^2/2! + x^3/3! ...
def approx_exp(x, i=1) :
    if x > 710 :
        raise Exception("Raw Python can't handle nums that big")
    
    precision = x * 10 if x < 90 else 900

    if i < precision :
        return x**i / factorial(i) \
            + approx_exp(x, i+1)
    return 1



def add_nx(f, fPrime, it, precision=3) :
    # Only an approximation anyway so round
    n = -round(fPrime(it), 2)
    nex = 0 if it != 0 else 0.1
    
    print("Iterative scheme: x_+ = 0.35 - f({})/{}".format(it, abs(n)))
    
    while round(it, precision+1) != round(nex, precision+1) :
        it, nex = it + f(it) / n, it
        print(it)
    
    return it




##########################
# Stoppping criteria

def error_bound(places) :
    return 5 * 10**-(places)


# Then take max?
def get_l(gPrime, a, b) :
    return (abs(gPrime(a)), abs(gPrime(b)) )


# precision: n-decimal-place accuracy
# L: grad at root
def abs_tolerance( precision, L ) :
    bound = error_bound(precision+1)
    if L >= 1 :
        raise Exception("This might not converge!")
    if L <= 0.5 :
        return bound
    else :
        return ( (1-L)/ L ) * bound


# precision: n-sig-fig accuracy
# L : grad at root
def relative_tolerance( precision, L ) :
    bound = error_bound(precision)
    if L >= 1 :
        raise Exception("This might not converge!")
    if L <= 0.5 :
        return bound
    else :
        return ( (1-L)/ L ) * bound


# 4.1
def is_onestep_converged(xr, xr1, L, precision=5):
    eps = abs_tolerance(precision, L)
    
    return abs( xr1 - xr ) <= eps


# 4.2
def is_relatively_onestep_converged(xr, xr1, L, precision=5):
    eps = relative_tolerance(precision, L)
    relError = abs(xr1 - xr) / abs(xr1)
    
    return relError <= eps


# 4.4
def is_small_residual(f, xr, precision=2, delta=None): 
    if not delta:
        delta = error_bound(precision)
        print(delta)
    #print("Is residual less than", delta, "?")
    
    return abs( f(xr) ) <= delta


def has_converged(error, tol, f, xr) :
    return error <= tol \
        and is_small_residual(f, xr)