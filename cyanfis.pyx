from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from cython cimport view
from cpython cimport array
from libc.math cimport exp, sqrt
from libc.stdlib cimport malloc, free
import itertools

@cython.cdivision(True)
cdef double gaussmf(double x, double mu, double sigma) nogil:
    return exp(-((x - mu)**2.) / (2.*sigma**2))

@cython.cdivision(True)
cdef double partial_dMF(double x, double mu, double sigma, int alpha) nogil:

    # alpha = 1: derivative of gaussmf wrt sigma
    # alpha = 0: derivative of gaussmf wrt mu
    if alpha:
        return (mu - x)**2*exp(-(mu - x)**2/(2*sigma**2))/sigma**3
    else:
        return (x - mu)*exp(-(mu - x)**2/(2*sigma**2))/sigma**2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double sum_double(double[:] arr) nogil:
    cdef double out = 0
    cdef int i
    for i in xrange(arr.shape[0]):
        out += arr[i]
    return out


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] divide_double(double[:] arr, double num):
    cdef double[:] out = arr.copy()
    cdef int i
    for i in xrange(arr.shape[0]):
        out[i] = arr[i] / num
    return out

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] mult_double(double[:] arr, double num):
    cdef:
        double[:] out = arr.copy()
        int i
    for i in xrange(arr.shape[0]):
        out[i] = arr[i] * num
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:,:,:] sum_2_3d_arrays(double[:,:,:] a, double[:,:,:] b):
    cdef:
        double[:,:,:] out = a.copy()
        int i,j,k
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            for k in xrange(a.shape[2]):
                out[i,j,k] = a[i,j,k] + b[i,j,k]
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:,:,:] mult_2_3d_arrays(double[:,:,:] a, double[:,:,:] b):
    cdef:
        double[:,:,:] out = a.copy()
        int i,j,k
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            for k in xrange(a.shape[2]):
                out[i,j,k] = a[i,j,k] * b[i,j,k]
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:,:,:] mult_double3d(double[:,:,:] arr, double num):
    cdef:
        double[:,:,:] out = arr.copy()
        int i, j, k
    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            for k in xrange(arr.shape[2]):
                out[i,j,k] = arr[i,j,k] * num
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double sum_elements_3d_array(double[:,:,:] arr) nogil:
    cdef:
        double out = 0
        int i,j,k

    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            for k in xrange(arr.shape[2]):
                out = out + arr[i,j,k]
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double mult_elements_1d_array(double[:] arr) nogil:
    cdef:
        double out = 1
        int i = 0
    for i in xrange(arr.shape[0]):
        out = out * arr[i]
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:,:] evaluateMF(double[:,:,:] MFList, double[:] rowInput):

    cdef double[:,:] out = np.empty((rowInput.shape[0], MFList.shape[1]))
    cdef int i, k
    for i in xrange(rowInput.shape[0]):
        for k in xrange(MFList[i].shape[0]):
            out[i,k] = gaussmf(rowInput[i], MFList[i,k,0], MFList[i,k,1])

    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int npwhere_first(int[:] arr, int num) nogil:
    cdef int i
    for i in xrange(arr.shape[0]):
        if arr[i] == num:
            break
    return i

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double[:] npflatten(double[:,:] arr):
    cdef:
        double[:] out = np.empty(arr.size)
        int i, j
        int outindex = 0

    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            out[outindex] = arr[i,j]
            outindex += 1
    return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int[:] npwhere(int[:] arr, int num):
    """Equivalent to np.where(), but it only works with int objects and array (1-dim)
    to scalar equivalences."""
    cdef int i
    cdef int outindex = 0
    cdef int[:] out = arr.copy()

    for i in xrange(arr.shape[0]):
        if arr[i] == num:
            out[outindex] = i
            outindex += 1
    return out[:outindex]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef short isin(int num, int[:] arr) nogil:
    cdef:
        int i
        short out = 0
    for i in xrange(arr.shape[0]):
        if arr[i] == num:
            out = 1
            break
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int[:] npdelete(int[:] arr, int num):
    """Equivalent to np.delete(), but it only works with int objects and array (1-dim)
    to scalar equivalences."""

    cdef int i
    cdef int outindex = 0
    # cdef int[:] out = np.empty(length - 1, np.dtype("i"))
    cdef int[:] out = np.empty(arr.shape[0] - 1, dtype=np.int32)
    for i in xrange(arr.shape[0]):
        if i != num:
            out[outindex] = arr[i]
            outindex += 1
    return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double[:,:] backprop(double[:,:] X,
                          double[:] Y,
                          double[:,:,:] memFuncs,
                          int[:,:] rules,
                          double[:] consequents,
                          int columnX,
                          int[:] columns,
                          double[:] theWSum,
                          double[:,:] theW,
                          double[:] theLayerFive):

    cdef:
        int nrules = rules.shape[0]
        int nmfuncs = memFuncs.shape[0]
        int MF, rowX, colY, consequent, lower, upper, i, r, c, rowindex, colindex
        int xcols = X.shape[1] + 1
        short alpha = 0
        int[:] adjCols = npdelete(columns,columnX)
        int[:] rulesWithAlpha = np.empty(memFuncs.shape[1]**2,dtype=np.int32)
        double varToTest, sum1, acum, prod, fConsequent, dW_dAplha_sum, senSit, thewsum_r
        double[:] temparr
        double[:] tmpRow = np.empty(nmfuncs)
        double[:] dW_dAplha = np.empty(memFuncs.shape[1]**2)
        double[:] parameters = np.empty(2)
        double[:] bucket3 = np.empty(X.shape[0])
        double[:] ics = np.empty(xcols)
        double[:] bucket1 = np.empty(nrules)
        double[:,:] evalmf
        double[:,:] X_inner = np.append(X,np.ones((X.shape[0],1)),axis=1)
        double[:,:] paramGrp = np.empty((memFuncs.shape[1],2))

    for MF in xrange(memFuncs.shape[1]):
        rulesWithAlpha = npwhere(rules[:,columnX], MF)

        for alpha in xrange(2):
            # bucket3 = np.empty(X.shape[0])
            for rowX in xrange(X.shape[0]):
                varToTest = X[rowX,columnX]
                tmpRow[:] = varToTest
                evalmf = evaluateMF(memFuncs, tmpRow)
                thewsum_r = theWSum[rowX]

                ics = X_inner[rowX,:]

                senSit = partial_dMF(
                            X[rowX,columnX],
                            memFuncs[columnX,MF,0],    # mean
                            memFuncs[columnX,MF,1],    # sigma
                            alpha)
                # produces d_ruleOutput/d_parameterWithinMF
                for r in xrange(rulesWithAlpha.shape[0]):
                    prod = 1.0
                    rowindex = rulesWithAlpha[r]
                    for c in xrange(adjCols.shape[0]):
                        colindex = adjCols[c]
                        prod *= evalmf[colindex, rules[rowindex, colindex]]

                    dW_dAplha[r] = prod * senSit

                dW_dAplha_sum = sum_double(dW_dAplha)

                for consequent in xrange(nrules):
                    acum = 0
                    if isin(consequent, rulesWithAlpha):
                        acum = ((dW_dAplha[npwhere_first(rulesWithAlpha,consequent)] * thewsum_r) - theW[rowX, consequent] * dW_dAplha_sum) / (thewsum_r * thewsum_r)
                    temparr = consequents[xcols * consequent:(xcols * consequent) + xcols]

                    fConsequent = 0
                    for i in xrange(xcols):
                        fConsequent += ics[i] * temparr[i]

                    bucket1[consequent] = acum * fConsequent
                bucket3[rowX] = sum_double(bucket1) * (Y[rowX]-theLayerFive[rowX])*(-2)
            parameters[alpha] = sum_double(bucket3)

        paramGrp[MF] = parameters

    return paramGrp

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef forwardHalfPass(double[:,:,:] memFuncs,
                    int[:,:] rules,
                    double[:,:] Xs):

    cdef:
        int pattern, x, row, z
        double[:] wSum = np.empty(Xs.shape[0])
        double[:] layerTwo = np.empty(rules.shape[0])
        double[:] layerThree = layerTwo.copy()
        double[:,:] layerOne = np.empty((Xs.shape[0], rules.shape[1]))
        double[:,:] miAlloc = np.empty((rules.shape[0], rules.shape[1]))
        double[:,:] w = np.empty((Xs.shape[0], layerTwo.shape[0]))
        double[:,:] X_inner = np.append(Xs,np.ones((Xs.shape[0],1)),axis=1)
        double[:,:] temparr = np.empty((layerThree.shape[0], X_inner.shape[1]))
        double[:,:] layerFour = np.empty((Xs.shape[0],rules.shape[0]*X_inner.shape[1]))
        double[:] rowHolder = np.empty(rules.shape[0]*Xs.shape[1] + 1)

    for pattern in xrange(Xs.shape[0]):
        #layer one
        layerOne = evaluateMF(memFuncs,Xs[pattern,:])

        #layer two
        for row in xrange(rules.shape[0]):
            for x in xrange(rules.shape[1]):
                miAlloc[row, x] = layerOne[x,rules[row,x]]
            layerTwo[row] = mult_elements_1d_array(miAlloc[row])

        w[pattern] = layerTwo

        #layer three
        wSum[pattern] = sum_double(layerTwo)

        #prep for layer four (bit of a hack)
        layerThree = divide_double(layerTwo, wSum[pattern])

        for z in xrange(layerThree.shape[0]):
            temparr[z] = mult_double(X_inner[pattern],layerThree[z])
        rowHolder = npflatten(temparr)

        layerFour[pattern] = rowHolder
    return layerFour, wSum, w

cpdef return_first_error(ANFISObj):
    return ANFISObj.errors[0]

cdef class ANFIS:
    """Class to implement an Adaptive Network Fuzzy Inference System: ANFIS"

    REFORMULATED: IT ONLY WORKS UNDER THE FOLLOWING CONDITIONS
    - ONLY GAUSSMF
    - ONLY ONE OUTPUT
    - ALL INPUTS HAVE THE SAME NUMBER OF MFs

    Attributes:
        X
        Y
        XLen
        memFuncs
        memFuncsByVariable
        rules
        consequents
        errors


    rewritten by Gabriele Iacono
    """

    cdef:
        double[:,:] X
        double[:] Y
        double[:,:,:] memFuncs
        int XLen, epochs
        int[:,:] rules
        double[:] consequents
        public double[:] errors
        double[:] fittedValues
        double[:] residuals

    def __cinit__(self, x, y, mflist, epochs):
        self.X = x
        self.Y = y
        self.XLen = x.shape[1]
        self.memFuncs = mflist #memFuncs and MFList are used as synonims (for the time being)

        memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*memFuncsByVariable)),dtype=np.int32)
        self.consequents = np.empty(len(self.rules) * (self.X.shape[1] + 1))
        self.epochs = epochs
        self.errors = np.empty(epochs)
        self.fittedValues = np.empty(y.shape[0])
        self.residuals = np.empty(y.shape[0])

    # def __init__(self, x, y, mflist, epochs):
    #     self.X = x
    #     self.Y = y
    #     self.XLen = x.shape[1]
    #     self.memFuncs = mflist #memFuncs and MFList are used as synonims (for the time being)
    
    #     memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
    #     self.rules = np.array(list(itertools.product(*memFuncsByVariable)),dtype=np.int32)
    #     self.consequents = np.empty(len(self.rules) * (self.X.shape[1] + 1))
    #     self.epochs = epochs
    #     self.errors = np.empty(epochs)
    #     self.fittedValues = np.empty(y.shape[0])
    #     self.residuals = np.empty(y.shape[0])

    cdef double[:] LSE(self,
                       double[:,:] A,
                       double[:] B,
                       double initialGamma = 1000):
        cdef:
            double[:,:] S = np.eye(A.shape[1])*initialGamma
            double[:] x = np.empty(A.shape[1]) # need to correct for multi-dim B
            double[:] a = x.copy()
            double b

        for i in xrange(A.shape[0]):
            a = A[i,:]
            b = B[i]
            S -= np.dot(np.dot(np.dot(S,a),a),S)/(1+np.dot(np.dot(S,a),a))
            x += np.dot(S,np.dot(a,(b-np.dot(a,x))))
        return x


    # cdef double[:] LSE(self, double[:,:] A, double[:] B, double initialGamma = 1000):
    # # cdef double[:] LSE(self, np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=1] B, double initialGamma = 1000):
    #
    #     # cdef:
    #     #     double[:,:] S = np.eye(A.shape[1])*initialGamma
    #     #     double[:] x = np.zeros(A.shape[1]) # need to correct for multi-dim B
    #     #     double[:] a = x.copy()
    #     #     double b
    #     #     int i
    #     #
    #     # for i in xrange(1,A.shape[0]):
    #     #     a = A[i,:]
    #     #     b = B[i]
    #     #     S[i] = S[i-1] - np.dot(np.dot(np.dot(S[i-1],a),a.T),S[i-1])/(1+np.dot(np.dot(a.T,S[i-1]),a))
    #     #     x[i] = x[i-1] +np.dot(S[i],np.dot(a,(b-np.dot(a.T,x[i-1]))))
    #
    #     cdef:
    #         double[:] x = np.empty(A.shape[1])
    #     x = np.dot(np.linalg.inv(np.dot(A.T,A)), np.dot(A.T, B))
    #     return x


    def trainHybridJangOffLine(self, double tolerance=1e-5, double initialGamma=1000, double k=0.01):
        cdef:
            int errindex = 0
            # int i
            int epoch = 1
            int[:] cols = np.array(xrange(self.XLen), dtype=np.int32)
            short convergence = 0
            double error, t0, eta
            double[:] wSum = np.empty(self.X.shape[0])
            double[:] layerFive = np.empty(self.X.shape[0])
            double[:,:] layerFour = np.empty((self.X.shape[0],self.rules.shape[0]*(self.X.shape[1] + 1)))
            double[:,:] w = np.empty((self.X.shape[0], self.rules.shape[0]))
            double[:,:, :] dE_dAlpha = np.empty((self.X.shape[0], self.memFuncs.shape[1], self.memFuncs.shape[2]))

        while (epoch <= self.epochs) and convergence != 1:

            #layer four: forward pass
            [layerFour, wSum, w] = forwardHalfPass(self.memFuncs, self.rules, self.X)

            #layer five: least squares estimate
            self.consequents = self.LSE(layerFour,self.Y,initialGamma)
            layerFive = np.dot(layerFour,self.consequents)

            #error
            error = sum_double((np.subtract(self.Y,layerFive)**2))
            self.errors[errindex] = error
            # print("current error: {0}".format(error))
            if error < tolerance:
                convergence = 1

            # backpropagation
            if convergence != 1:
                # t0 = time()
                for colX in xrange(self.X.shape[1]):
                    dE_dAlpha[colX,:,:] = backprop(self.X,
                                                   self.Y,
                                                   self.memFuncs,
                                                   self.rules,
                                                   self.consequents,
                                                   colX,
                                                   cols,
                                                   wSum,
                                                   w,
                                                   layerFive)
                # print("Backprop: {0}".format(time()-t0))

                if errindex >= 4:
                    if all([(self.errors[errindex] < self.errors[errindex-1]),
                            (self.errors[errindex-2] < self.errors[errindex-2]),
                            (self.errors[errindex-2] < self.errors[errindex-3]),
                            (self.errors[errindex-4] > self.errors[errindex-3])]):
                        k *= 0.9
                if errindex >= 3:
                    if (self.errors[errindex - 3] > self.errors[errindex - 2] > self.errors[errindex - 1] > self.errors[errindex]):
                        k *= 1.1

                errindex += 1
                try:
                    eta = k / sqrt(sum_elements_3d_array(mult_2_3d_arrays(dE_dAlpha,dE_dAlpha)))
                except ZeroDivisionError:
                    eta = k

                self.memFuncs = sum_2_3d_arrays(self.memFuncs,mult_double3d(dE_dAlpha, -eta))

                epoch += 1

        self.fittedValues = self.predict(self.X)
        self.residuals = np.subtract(self.Y, self.fittedValues)


        return self.fittedValues


    cpdef double[:] predict(self, double[:,:] varsToTest):
        cdef double[:] layerfive

        [layerFour, wSum, w] = forwardHalfPass(self.memFuncs, self.rules, varsToTest)

        layerfive = np.dot(layerFour,self.consequents)

        return layerfive