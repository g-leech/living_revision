import numpy as np
import pandas as pd
import numpy.linalg as lin
from itertools import permutations


# non-singular == invertible
def is_non_singular(M) :
    return round(lin.det(M), 6) != 0


# m_{row}{col} = a_{row}{col} / a_{col}{col}
def multiplier(A, row, col):
    denom = A[col-1][col-1]
    assert(denom != 0)
    
    return A[row-1][col-1] / denom


def get_indices_to_zero_out(A):
    rows = list(range(1, A.shape[0]) )
    cols = list( range(A.shape[1]-1) )
    perms = [list(zip(rows, p)) for p in permutations(cols)]
    flat = [el for sublist in perms for el in sublist]
    bottomTriangle = [i for i in flat if i[0] > i[1] ]
    
    return sorted(set([p for p in bottomTriangle if p[0] !=p[1] ]))


def zero_out_element_at(M, row, col) :
    if M[(row-1, col-1)] != 0 :
        m = multiplier(M, row, col)
        M[row-1] = M[row-1] - (m * M[col-1])
    
    return M[:,:-1], M[:,-1]


def check_pivots(A):
    return [i for i in range(len(A)) \
    		if A[i][i] == 0]


def check_and_swap(A, b) :
    zeroPivots = check_pivots(A)
    if zeroPivots :
        next_ = zeroPivots[0]
        A, b = interchange(A, b, next_, next_+1)
        print("Changing row", next_, "with", next_+1 )
        
    return A, b


def convert_to_upper_triangular(A, b, show=True) :
    idx = get_indices_to_zero_out(A)
    
    for row, col in idx :
        A, b = zero_out_element_at(np.column_stack((A, b)), \
        							row+1, col+1)
        A, b = check_and_swap(A, b)
        
        if show :
            print(A)
            print(b)
    
    return A, b



# Checks for equivalence class of solution set
# Attempts row substitutions if pivots are zero
def gaussian_elimination(A, b) :
    typ = solution_class(A, b)
    if typ != "unique set" :
        raise Exception(typ + " of solutions")
    
    upTriangularA, b = convert_to_upper_triangular(A, b)
    
    return backSubstitute(upTriangularA, b)


def get_last_value(a, b) :
    coeff = a[-1][-1]
    
    return round(b[-1] / coeff, 4)


def get_offset(A, i, d, n) :
    terms = 0
    for j in range(1,i):
        next_ = "x"+ str(n-j)
        terms += A[-i,-j] * d[next_]

    return terms


def backSubstitute(A, b) :
    n = len(A) + 1
    d = {}

    for i in range(1, n):
        key = "x" + str(n - i)
        terms = get_offset(A, i, d, n)

        val = (b[-i] - terms) / A[-i][-i]
        d[key] = val
        print()

    return d



def interchange(A, b, rowI, rowJ) :
    A[[rowI,rowJ]] = A[[rowJ,rowI]]
    b[[rowI,rowJ]] = b[[rowJ,rowI]]
    
    return A, b



def solution_class(A, b) :
    if is_non_singular(A) :
        return "unique set"
    else :
        return type_of_brokenness(A, b)

    

def get_sequence_of_matrices(A, b) :
    As, bs = [A], [b]
    idx = get_indices_to_zero_out(A)
    
    for row, col in idx :
        A, b = check_and_swap(A, b)
        try :
            A, b = zero_out_element_at(np.column_stack((A, b)), \
            							row+1, col+1)
        except AssertionError :
            pass
        
        As += A
        bs += b
    
    return As, bs


def type_of_brokenness(A, b) :
    A, b = convert_to_upper_triangular(A, b, show=False)
    lastValueZero = b[-1] == 0
    lastPivotZero = A[-1][-1] == 0
    
    if lastPivotZero and lastValueZero :
        return "infinite set"
    elif lastPivotZero :
        return "empty set"
    else :
        raise Exception("Nonsingular but not simply")

