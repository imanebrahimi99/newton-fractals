# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:54:39 2022

@author: Iman Ebrahimi, Ruolin Wang, Ruth Risberg, Sebastian Westerlund,
Ayush Chakraborty 
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import types


class fractal2D:
    # it was suggested to use a tolerance to look for roots and distinguish between them
    # Task 1
    def __init__(self, function, derivative, tolerance=5e-8, max_iter=100):
        self.func = function
        if isinstance(derivative, types.FunctionType):
            self.dfunc = derivative
            self.approx = False
        else:
            self.df_step = derivative
            self.approx = True
        self.tol = np.abs(
            tolerance
        )  # to prevent mistakes for any correct implementation
        self.maxi = int(np.abs(max_iter))  
        # to prevent mistakes for any correct implementation
        self.simple = False
        self.zeroes = []



    def df_approx(self, p): #Approximates the Jacobian at a point p

        self.df = np.zeros(shape=(2, 2), dtype=np.float64)
        
        #Approximate and save derivatives wrt x
        fr = self.func(np.array([p[0] + self.df_step, p[1]]))
        fl = self.func(np.array([p[0] - self.df_step, p[1]]))
        self.df[0, 0] = (fr[0] - fl[0]) / (2*self.df_step)
        self.df[1, 0] = (fr[1] - fl[1]) / (2*self.df_step)

        #Approximate and save derivatives wrt y
        fr = self.func(np.array([p[0], p[1] + self.df_step]))
        fl = self.func(np.array([p[0], p[1] - self.df_step]))
        self.df[0, 1] = (fr[0] - fl[0]) / (2*self.df_step)
        self.df[1, 1] = (fr[1] - fl[1]) / (2*self.df_step)
        return

    def is_simple(self, simple):  #Task 5
        if isinstance(simple, bool):
            self.simple = simple
    """
    def newton(self, guess):  #Task 2
        old = guess
        try:
            if self.approx == False:
                J1 = inv(
                    self.dfunc(old)
                )  # reordered evaluation of elements to update inv(Jacobian)
                # only for simple==True case
            else:  # Task 5
                self.df_approx(old)
                J1 = inv(self.df)
        except FloatingPointError:
            return None  # We might face ill conditioned matrices for some points
          

        for i in range(self.maxi):
            chg = J1 @ self.func(old)
            chg_max_norm = np.abs(chg)
            new = old - chg
            if self.tol > max(chg_max_norm[0],
                              chg_max_norm[1]):  #np.allclose(new, old):
                return new
            else:
                old = new
                if self.simple == False:
                    try:  # still we might get ill conditioned matrices...

                        if self.approx == False:
                            J1 = inv(self.dfunc(old))
                        else:
                            self.df_approx(old)
                            J1 = inv(self.df)
                    except FloatingPointError:
                        return new
        else:
            return None
    """
    def newton_i(self, guess):
        old = guess
        try:
            if self.approx == False: #If the derivative is a function:
                J1 = inv(self.dfunc(old))  #This is the first jacobian
                # reordered evaluation of elements to update inv(Jacobian)
                # only for simple==True case
            else:
                self.df_approx(old) # Approximating it 
                J1 = inv(self.df)   # Then store it as our first jacobian matrix
        except FloatingPointError:
            return None, 0  # We might face ill conditioned matrices for some points
            # generated for mesh and we must prevent bad calculations

        for i in range(self.maxi):
            chg = J1 @ self.func(old)
            chg_max_norm = np.abs(chg)
            new = old - chg
            new_max_norm = np.abs(self.func(new))
            if (not self.simple) and self.tol > max(chg_max_norm[0],
                              chg_max_norm[1]):  #alternative: np.allclose(new, old):
                return new, i + 1
            elif self.simple and self.tol > max(new_max_norm[0], 
                                                new_max_norm[1]):
                return new, i+1
            else:
                old = new
                if self.simple == False:
                    try:  # still we might get ill conditioned matrices...

                        if self.approx == False:
                            J1 = inv(self.dfunc(old))
                        else:
                            self.df_approx(old)
                            J1 = inv(self.df)
                    except FloatingPointError:
                        return new, i + 1
        else:
            return None, 0
    """
    def getzero(self, guessx, guessy):
        newzero = self.newton(np.array([guessx, guessy]))
        if isinstance(newzero, type(None)):
            return -1

        for i, val in enumerate(self.zeroes):
            if np.allclose(newzero, val):
                return i

        self.zeroes.append(newzero)
        return (len(self.zeroes) - 1)
    """
# %%
    def getzero_i(self, guessx, guessy):
        newzero, cnt = self.newton_i(np.array([guessx, guessy]))
        if isinstance(newzero, type(None)): #If the newton method did not converge to a value, then:
            return -1, 0 #Assigned -1 to divergence. 

        for i, val in enumerate(self.zeroes): #i = index, val = value
        #Going through the list and checking if the zero we found is already there or not
        #if it is, we return the index.
            if np.allclose(newzero, val):
                return i, cnt

        self.zeroes.append(newzero)
        return (len(self.zeroes) - 1), cnt #Returning The Last Index of the zeroes. 
# %%
    """
    def plot(self, N, a, b, c, d, simple=False): #Task 4
        print('Plotting...') # Just to show that the program works not bugging
        x = np.linspace(a, b, N)
        y = np.linspace(c, d, N)
        X, Y = np.meshgrid(x, y)
        self.is_simple(simple)
        vgetzeros = np.vectorize(self.getzero)
        A = vgetzeros(X, Y)
        plt.pcolor(x, y, A, shading="auto")
        plt.show()
        print('Done!')
    """
    def plot_i(self, N, a, b, c, d, simple=False, counting=False): # Task 7
        x = np.linspace(a, b, N)
        y = np.linspace(c, d, N)

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(x, y)
        self.is_simple(simple)
        vgetzeros = np.vectorize(self.getzero_i)
        A, C = vgetzeros(X, Y)

        """
        for zero in self.zeroes:
            i1 = round(N*(zero[0]-a)/(b-a))
            i2 = round(N*(zero[1]-a)/(b-a))
            A[i1,i2] = 10
        """

        colors = ax.pcolor(x, y, A, shading="auto", picker=1)

        ticks = [i for i in range(np.amin(A), np.amax(A)+1)]
        fig.colorbar(colors, ax=ax, ticks=ticks)

        fig2, ax2 = plt.subplots()
        colors2 = ax2.pcolor(x, y, C, shading="auto", picker=1)

        ticks2 = [i for i in range(np.amin(C), np.amax(C), 4)]
        ticks2.append(np.amax(C))
        fig2.colorbar(colors2, ax=ax2, ticks=ticks2)

        def onpick(event):
            ind = event.ind
            row, col = int(ind[0] / N), ind[0] % N
            print('Root index: %ld, Location: (%lf, %lf), Iterations: %ld' %
                  (A[row, col], x[row], y[col], C[row, col]))

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()


# %%
def f(X):
    x, y = X
    return np.array([x**3 - 3 * x * y**2 - 1, 3 * x**2 * y - y**3])


def df(X):
    x, y = X
    J = np.array([[3 * x**2 - 3 * y**2, -6 * x * y],
                  [6 * x * y, 3 * x**2 - 3 * y**2]])
    return (J)


# %%

# %%



# %%
def f1(X):
    x, y = X
    return np.array(
        [x**3 - 3 * x * y**2 - 2 * x - 2, 3 * x**2 * y - y**3 - 2 * y])

def df1(X):
    x, y = X
    return np.array(
        [[ 3 * x**2 - 3 * y**2 - 2,      -6 * x * y],
         [ 6 * x * y,                    3 * x**2 - 3 * y**2 - 2]]
    )


# %%
def f2(X):
    x, y = X
    return np.array([
        x**8 - 28 * x**6 * y**2 + 70 * x**4 * y**4 + 15 * x**4 -
        28 * x**2 * y**6 - 90 * x**2 * y**2 + y**8 + 15 * y**4 - 16,
        8 * x**7 * y - 56 * x**5 * y**3 + 56 * x**3 * y**5 + 60 * x**3 * y -
        8 * x * y**7 - 60 * x * y**3
    ])

def df2(X):
    x, y = X
    return np.array(
        [[8*x**7-168*x**5*y**2+280*x**3*y**4+60*x**3-56*x**1*y**6-180*x**1*y**2,
          -56*x**6*y**1+280*x**4*y**3-168*x**2*y**5-180*x**2*y**1+8*y**7+60*y**3],
         
         [56*x**6*y-280*x**4*y**3+168*x**2*y**5+180*x**2*y-8*1*y**7-60*1*y**3,
          8*x**7*y-168*x**5*y**2+280*x**3*y**4+60*x**3*y-56*x*y**6-180*x*y**2]]
    )

def test(X):
    x, y = X
    return np.array([
        np.sin(x)+3*y, np.cos(y) + 4*np.sin(x)
    ])

def dtest(X):
    x,y = X
    return np.array([
        [np.cos(x), 3],
        [4*np.cos(x), -np.sin(y)]
    ])

obj = fractal2D(f2,0.01)

a = 5
obj.plot_i(50, -a, a, -a, a, simple = False)  #Try onclick in automatic interactive backend

#obj.plot_i(50, -a, a, -a, a, simple = True)
plt.show()


def differentiate(func, var): # differentiates polynomials on a nice format, if powers are 9 or less
    func = list(func)
    func.append('+')
    out = []
    term = []
    for i in func:
        if i == ' ':
            continue
        elif i in '+-':
            coeff = ''
            rest = []
            for j in range(len(term)):
                if term[j] in '+-':
                    continue
                if term[j] in '1234567890':
                    coeff += term[j]
                else:
                    rest = term[j:]
                    break
            if coeff == '':
                coeff = '1'
                rest = ['*'] + rest
            coeff = int(coeff)
            if term[0] == '-':
                coeff *= -1
            #print(coeff)
            found = False
            for j in range(len(rest)):
                if rest[j] == var:
                    #print(j)
                    found = True
                    for k in range(j+1, len(rest)):
                        #print(rest[k])
                        if rest[k] == '*':
                            continue
                        if rest[k] in '1234567890':
                            coeff *= int(rest[k])
                            rest[k] = str(int(rest[k])-1)
                            break
                        else:
                            rest[j] = '1'
                            break
                    break
            if not found:
                term = [i]
            else:
                if coeff >= 0:
                    out.append('+')
                out.append(str(coeff))
                out.append(''.join(rest))
                term = [i]
        else:
            term.append(i)
    return(''.join(out))