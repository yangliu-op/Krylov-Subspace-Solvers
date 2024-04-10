# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:46:50 2022

Preconditioned MINRES with Positive semi-definite preconditioners

Note that this code is experimental for the paper below and subject to change

Reference:
Obtaining Pseudo-inverse Solutions With MINRES
https://arxiv.org/abs/2309.17096
Authors: Yang Liu, Andre Milzarek, Fred Roosta


@author: Yang Liu
"""
import torch
from torch import conj_physical as conj
# from FCR import myCR
from FCG import CCG
from MinresQLP import MinresQLP
from tool_functions import showFigure
import os
from torch import pinverse as pinv # for testing purposes

class MINRES(object):
    def __init__(self, A, b, rtol=1E-4, maxit=100, M=None, shift=0, inner_rtol=None, 
                 reorth=False, npc=False, inexactness="relres", eps=1E-10):
        '''
        Parameters
        ----------
        A : d x d Symmetrix, Hermitian or Complex Symmetric matrices.
        b : d x 1 vector. Same datetype as A.
        rtol : Scalar, optional. 
            The default is 1E-4.
        maxit : Int, optional
            Maximum iteration. The default is 100.
        M : Positive semedefinite Preconditioner operator, optional
            The default is None.
        shift : Scalar, optional
            Perturbation on A; min || b - (A + shift eye) x ||^2. The default is 0.
        inner_rtol : Scalar, optional
            Tolerance of inner relative residual termination for preconditioners.
        reorth : Boolean, optional
            Reorthogonalization. The default is False.
        npc : Boolean, optional
            Non-positive curvature (NPC) detection. The default is False.
            Will terminate if NPC direction being detected.
        inexactness : TYPE, optional
            Termination condition. 
            The default is "relHres" ==> || H r || \leq rtol || H x ||.
            "relres" ==> || r || \leq rtol || b ||
            Subject to change.
        eps : Scalar, optional
            Termination Tolerance of numerical error, better to set large if 
            reorth==False for large scale problem. 
            The default is 1E-10.

        Returns
        -------
        None.

        '''
        self.A = A
        self.b = b
        self.dim = len(b)
        
        # Check input data type
        if torch.is_complex(b):
            if torch.norm(A.T - A) < 1e-12*self.dim**2:
                self.Atype = 'CS' # Complex Symmetric
            else:
                self.Atype = 'H' # Hermitian
        else:
            self.Atype = 'S' # Real Symmetric
            
        self.r = b.clone()
        self.device = b.device
        self.rtol = rtol
        self.maxit = maxit
        self.shift = shift
        self.flag = -1
        self.iters = 0
        self.reorth = reorth
        self.npc = npc
        self.dtype = "SOL"
        self.eps = eps
        self.inexactness = inexactness
        self.theAdb = pinv(self.A) @ b
        
        # Initialize Preconditioner
        if M is None:
            self.precon = False
            self.beta1 = b.norm()
            vn = b.clone()
        else: # v --> w, but || w || \neq 1
            self.precon = True
            if inner_rtol is None:
                self.inner_rtol = 1e-10 # solving more exactly increases stability
            else:
                self.inner_rtol = inner_rtol
            if self.Atype == "CS":
                self.M = conj(M)
            else:
                self.M = M
            vn = CCG(self.M, b, self.inner_rtol, self.maxit)[0]
            # vn = myCR(self.M, b, self.inner_rtol, self.maxit)[0]
            
        # Compute beta_1
        if self.Atype == "CS":
            self.beta1 = torch.sqrt(b.T.dot(conj(vn)))
        else:            
            self.beta1 = torch.sqrt(torch.vdot(vn, b))
        self.beta1 = self.beta1.real # beta is always real
            
        self.hr = vn.clone()
        self.vn = vn/self.beta1 # vn = wn/betan
        self.zn = b.clone()/self.beta1 # store zn = zn/betan
        self.z = torch.zeros_like(b)
            
        self.betan = self.beta1        
        self.cs = -self.beta1/self.beta1
        self.sn = 0*self.beta1
        self.tau = 0
        self.delta1 = 0
        self.epsilonn = 0
        self.phi = self.beta1
        self.relres = 1
        self.x = torch.zeros_like(b)
        self.d = torch.zeros_like(b)
        self.dl = torch.zeros_like(b)
        self.x_lifted = torch.zeros_like(b)
        
        # recording for experimental purposes
        self.record = torch.tensor([1, 1], device=b.device,
                                   dtype=torch.float64).reshape(1,-1)
        
        if self.reorth: 
            self.V = torch.outer(self.zn, conj(self.vn))

    def run(self):
        while self.flag == -1:            
            self._1run() # perform 1 iteration of MINRES
            
            # recording for experimental purposes
            tmp = torch.tensor([(self.theAdb - self.x).norm()/self.theAdb.norm(), 
                              (self.theAdb - self.x_lifted).norm()/self.theAdb.norm()], 
                             device = self.record.device, dtype=torch.float64)
            self.record = torch.cat((self.record, tmp.reshape(1,-1)), axis=0)
            
            if self.iters >= self.maxit and self.flag == -1:                
                self.flag = 4
                print("Maximum iteration reached")
                self.dtype = "MAX"
                break
        return self.x, self.relres, self.iters, self.r, self.dtype

    def _1run(self):
        # Lanczos
        if self.Atype == "CS":
            self._lanczos(conj(self.vn))
        else:
            self._lanczos(self.vn)
        self.iters += 1
            
        # QR decomposition
        self.epsilon = self.epsilonn
        self._qr()
        
        self.gamma2 = torch.sqrt(torch.abs(self.gamma1)**2 + self.betan**2)
        # self.cs, self.sn, self.gamma2 = self._symgivens(self.gamma1, self.betan) 
        
        # Inexactness checking, subject to change
        if self.inexactness == "relres":
            inexactnesscheck = (self.relres < self.rtol)
        elif self.inexactness == "Arnorm":
            inexactnesscheck = (self.phi*torch.sqrt(
                self.gamma1**2 + self.delta1**2) < self.rtol)
        else:
            inexactnesscheck = (self.phi*torch.sqrt(
                self.gamma1**2 + self.delta1**2) < self.rtol*torch.sqrt(
                                                  self.beta1**2-self.phi**2))
        if inexactnesscheck:
            self.flag = 1  ## trustful least-squares solution
            print("Inexactness reached")
            return
        
        # non-positive curvature (NPC) detection
        npc = - self.cs * self.gamma1
        if self.Atype != "CS" and self.npc and npc < self.eps:
            # NPC detection for Hermitian system
            self.dtype = 'NPC'
            print("NPC detected")
            return
        
        # Update
        if self.gamma2 > self.eps:
            self.cs = self.gamma1/self.gamma2
            self.sn = self.betan/self.gamma2
            if self.Atype == "CS":
                self._updates(conj(self.v))
            else:
                self._updates(self.v)                
            self.relres = self.phi/self.beta1
            if self.betan < self.eps:
                self.flag = 0
                self.x_lifted = self.x # ||r|| = 0, No lifting being needed
                print('Exact soltion')
                return
        else:
            ## gamma1 = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            self.cs = 0
            self.sn = 1
            self.gamma2 = 0  
            self.flag = 2
            self._lifting() # Exact pseudo-inverse solution will be found
            print('Exact least-squred soltion')
            return

    def _lanczos(self, cvn):
        # zn = vn, z = v for non-preconditioned system
        # cvn = vn, for Hermitian system
        Av = self._Av(cvn)        
        if self.shift == 0:
            pass
        else:
            Av = Av + self.shift*cvn
            
        # compute alpha
        self.alpha = torch.vdot(self.vn, Av)
        if self.Atype != "CS":
            self.alpha = self.alpha.real # the Hermitian alpha in real
            
        zn = Av - self.alpha*self.zn - self.betan*self.z # pLanczos
        
        # if self.reorth:
        #     # Only do reorthogonalization on vn is sufficient
        #     zn = zn - torch.mv(self.V, zn)
        self.z = self.zn.clone() # backup
        self.v = self.vn.clone() # backup
        
        # preconditioning
        if self.precon:
            vn = CCG(self.M, zn, self.inner_rtol, self.maxit)[0]
        else:
            vn = zn
            
        # reorthogonalization
        if self.reorth: 
            # the reorthogonalization for CS is indentical to H if vn = conj(wn)
            vn = vn - torch.mv(self.V.H, vn)
            if not self.precon:
                zn = vn
            
        # compute beta
        if self.Atype == "CS":
            # betan=sqrt(<wn, conj(zn)>)=sqrt(<conj(wn), zn>) for CS
            betan2 = zn.T.dot(conj(vn))
        else:
            betan2 = torch.vdot(zn, vn)
        if torch.abs(betan2) < 1e-15: # to avoid numerical error
            self.betan = 0*betan2.real
        else:
            self.betan = torch.sqrt(betan2).real # all betan in real
        if torch.isnan(self.betan):
            print('M is indefinite!? Double Check!', torch.vdot(zn, vn))
        
        self.vn = vn/self.betan
        self.zn = zn/self.betan
        if self.reorth: # updates reorthogonalization matrix
            self.V = self.V + torch.outer(self.zn, conj(self.vn))
        
    def _qr(self):
        # self.cs is real for Hermitian cases
        self.delta2 = conj(self.cs)*self.delta1 + self.sn*self.alpha
        self.gamma1 = self.sn*self.delta1 - self.cs*self.alpha
        self.epsilonn = self.sn*self.betan
        self.delta1 = -self.cs*self.betan

    def _updates(self, v):
        # self.cs is real in Hermitian case
        self.tau = conj(self.cs)*self.phi 
        self.phi = self.sn*self.phi

        tmp = (v - self.epsilon*self.dl - self.delta2*self.d)/self.gamma2
        self.dl = self.d.clone()
        self.d = tmp
        self.x = self.x + self.tau*self.d
        self.r = self.sn **2 * self.r - self.phi * conj(self.cs) * self.zn # r_check
        self.hr = self.sn **2 * self.hr - self.phi * conj(self.cs) * self.vn # r_hat
        
        self._lifting()
    
    def _lifting(self):
        # Lifting & Correction to obtain pseudo-inverse solution approxmation
        # It will be exact when MINRES being terminated
        if self.precon: # Lifting for preconditoned system
            if self.Atype == "CS":
                self._lifting_formula(conj(self.r), conj(self.hr))
            else:
                self._lifting_formula(self.r, self.hr)
        else:
            if self.Atype == "CS":
                self._lifting_formula(conj(self.r), conj(self.r))
            else:
                self._lifting_formula(self.r, self.r)
    
    def _lifting_formula(self, r1, r2):
        self.x_lifted = self.x - torch.vdot(r1, self.x)/torch.vdot(r2, r1)*r2
    
    def _symgivens(self, a, b):
        # cs, sn, gamma2 computation from [Choi, 2006]MINRES-QLP
        a_ = torch.abs(a)
        if a_ == 0:
            c = 0
            s = 1
            r = b
        elif b == 0:
            s = 0
            c = a / a_
            r = a_
        elif b >= a_:
            t = a_ / b
            s = 1 / torch.sqrt(1 + t ** 2)
            c = s / b * a
            r = b / s
        else:
            t = b / a_
            s = t / torch.sqrt(1 + t ** 2)
            c = s / b * a
            if torch.is_complex(a):
                r = b / s
            else:
                r = a / c
        return c, s, r  

    def _Av(self, x):
        if callable(self.A):
            Ax = self.A(x)
        else:
            Ax = torch.mv(self.A, x)
        return Ax

def tocomp(a): # real to complex
    return torch.complex(a, torch.zeros_like(a))


def CompSymm(n, r=None, device=None):
    '''
    Randomly generate n x n complex symmetric matrix
    Parameters
    ----------
    n : Integer
        Size.
    r : Integer leq n, optional
        Rank. The default is None.
    device : cpu or cuda, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A : Complex128, n x n
        A n x n complex symmetric matrix with rank r.
    U : Complex128, n x r
        U matrix for SVD.
    s : torch.float64, n
        Singular values.

    '''
    if r > n:        
        raise ValueError('Rank can not execeed dimension!')
    if device is None:
        device = "cpu"        
    mydtypec = torch.complex128
    W1 = torch.randn(n, n, device=device, dtype=mydtypec)
    
    A = W1.T @ W1
    U, s, V = torch.svd(A)
    if r is None:
        r = n
    else:
        s[r:] = 0
    A = U @ torch.diag(tocomp(s)) @ U.t()
    return A, U, s, U.t()

def Symm(n, r=None, device=None, PSD=False):
    '''
    Randomly generate n x n symmetric matrix
    Parameters
    ----------
    n : Integer
        Size.
    r : Integer leq n, optional
        Rank. The default is None.
    device : cpu or cuda, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A : Complex128, n x n
        A n x n complex symmetric matrix with rank r.
    U : Complex128, n x r
        U matrix for SVD.
    s : torch.float64, n
        Singular values.

    '''
    if r > n:        
        raise ValueError('Rank can not execeed dimension!')
    if device is None:
        device = "cpu"        
    mydtypec = torch.float64
    W1 = torch.randn(n, n, device=device, dtype=mydtypec)
    
    A = W1.T @ W1
    U, s, V = torch.svd(A)
    if not PSD:
        s[0] = -1
    if r is None:
        r = n
    else:
        s[r:] = 0
    A = U @ torch.diag(s) @ U.T
    return A, U, s, U.T

def Herm(n, r=None, device=None, PSD=False):
    '''
    Randomly generate n x n Hermitian matrix
    Parameters
    ----------
    n : Integer
        Size.
    r : Integer leq n, optional
        Rank. The default is None.
    device : cpu or cuda, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A : Complex128, n x n
        A n x n complex symmetric matrix with rank r.
    U : Complex128, n x r
        U matrix for SVD.
    s : torch.float64, n
        Singular values.

    '''
    if r > n:        
        raise ValueError('Rank can not execeed dimension!')
    if device is None:
        device = "cpu"        
    mydtypec = torch.complex128
    W1 = torch.randn(n, n, device=device, dtype=mydtypec)
    
    A = W1.H @ W1
    U, s, V = torch.svd(A)
    if not PSD:
        s[0] = -1
    if r is None:
        r = n
    else:
        s[r:] = 0
    A = U @ torch.diag(tocomp(s)) @ U.H
    return A, U, s, U.H

def SkewSymm(n, r=None, device=None):
    '''
    Randomly generate n x n complex symmetric matrix
    Parameters
    ----------
    n : Integer
        Size.
    r : Integer leq n, optional
        Rank. The default is None.
    device : cpu or cuda, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A : Complex128, n x n
        A n x n complex symmetric matrix with rank r.
    U : Complex128, n x r
        U matrix for SVD.
    s : torch.float64, n
        Singular values.

    '''
    if r > n:        
        raise ValueError('Rank can not execeed dimension!')
    if device is None:
        device = "cpu"        
    mydtypec = torch.float64
    W1 = torch.randn(n, n, device=device, dtype=mydtypec)
    
    A = W1.T @ W1
    for i in range(n):
        for j in range(n):
            if i < j:
                A[i,j] *= -1
            if i == j:
                A[i,i] = 0
    if r is None:
        r = n
    else:
        U, s, V = torch.svd(A)
        s[r:] = 0
        A = U @ torch.diag(s) @ V.t()
    return A, U, s, V.t()

def SkewHerm(n, r=None, device=None):
    if device is None:
        device = "cpu"        
    mydtypec = torch.complex128
    W1 = torch.randn(n, n, device=device, dtype=mydtypec)
    
    A = W1.T @ W1
    for i in range(n):
        for j in range(n):
            if i < j:
                A[i,j].real *= -1
            if i == j:
                A[i,i].real = 0
                A[i,i].imag = 1
    if r is None:
        r = n
    else:
        U, s, V = torch.svd(A)
        s[r:] = 0
        A = U @ torch.diag(tocomp(s)) @ V.H
    return A, U, s, V.H
    
def pseudoinverse_recovery():
    n = 20
    d = 15
    A1, U1, s1, V1 = Herm(n,d)
    A2, U2, s2, V2 = CompSymm(n,d)
    # A3, U3, s3, V3 = SkewSymm(n,d)
    b = torch.ones(n, dtype=torch.complex128)
    rtol = 1E-15
    maxit = n
    print('')
    record_all = []  
    methods_all = []
    # reorth = False
    reorth = True
    PA1 = MINRES(A1, b, rtol, maxit, reorth=reorth)
    PA1.run()
    methods_all.append('Hermitian')
    record_all.append(PA1.record)
    PA2 = MINRES(A2, b, rtol, maxit, reorth=reorth)
    PA2.run()
    methods_all.append('Complex Symmetric')
    record_all.append(PA2.record)
    mypath = 'PMINRES_pseudo_n_%s' % (n)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    showFigure(methods_all, record_all, mypath)

def MINRES_vs_MINRESQLP():
    n = 20
    d = 15
    A1, U1, s1, V1 = Symm(n,d)
    b = torch.ones(n, dtype=torch.float64)
    rtol = 1E-15
    eps = 1e-2
    maxit = n
    print('')
    record_all = []  
    methods_all = []
    PA1 = MINRES(A1, b, rtol, maxit, reorth=True, eps = eps)
    PA1.run()
    methods_all.append('MINRES')
    record_all.append(PA1.record)
    PA2, record2 = MinresQLP(A1, b, rtol, maxit, reOrth=True, Adb= pinv(A1) @ b)
    methods_all.append('MINRES-QLP')
    record_all.append(record2)
    mypath = 'PMINRES_pseudo_n_%s' % (n)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    showFigure(methods_all, record_all, mypath)

def verification(): 
    """Run an example of minresQLP."""
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n = 20
    m = n
    R = torch.randn(n, n, device=device, dtype=torch.float64)
    B = R + R.T
    U, s, V = torch.svd(B)
    d = n
    prob = ['Symm', 'Herm', 'CS']
    
    for i in range(3):
        if prob[i] == 'Symm':
            print('Symmetric')
            A = Symm(n, d)[0]
            b = torch.randn(n, device=device, dtype=torch.float64)
            S = torch.randn(n, m, device=device, dtype=torch.float64)
            tA = S.T @ A @ S
            tb = S.T @ b
        if prob[i] == 'Herm':
            print('Hermitian')
            A = Herm(n, d)[0]
            b = torch.randn(n, device=device, dtype=torch.complex128)
            S = torch.randn(n, m, device=device, dtype=torch.complex128)
            tA = S.H @ A @ S
            tb = S.H @ b
        if prob[i] == 'CS':
            print('Complex Symmetric')
            A = CompSymm(n, d)[0]
            b = torch.randn(n, device=device, dtype=torch.complex128)
            S = torch.randn(n, m, device=device, dtype=torch.complex128)
            tA = S.T @ A @ S
            tb = S.T @ b
        M = torch.mm(S, S.H)
        rtol = 1e-7
        maxit = n
        reorth = True
        # MINRES on preconditioned system
        MINRES_PS = MINRES(tA, tb, rtol, maxit, reorth=reorth)
        MINRES_PS.run()
        # preconditioned MINRES
        PMINRES = MINRES(A, b, rtol, maxit*3, M=M, reorth=reorth)
        PMINRES.run()
        
        print('Verification', MINRES_PS.phi, PMINRES.phi)
        print('Preconditioned MINRES == MINRES on preconditioned system',
              (S @ MINRES_PS.x - PMINRES.x).norm())
        print('True solution recovery', 
              (S @ MINRES_PS.x - S @ pinv(tA) @ tb).norm(), '\n')
            
    
if __name__ == '__main__':
    ### To test/verify implementation, run:
    verification()
    
    ### To generate pseudo-inverse recovery experiments between Hermitian systems  
    ### and Complex-symmetric systems, run:
    # pseudoinverse_recovery() 
    
    ### To generate pseudo-inverse recovery experiments between MINRES-QLP and 
    # MINRES run:
    # MINRES_vs_MINRESQLP()
