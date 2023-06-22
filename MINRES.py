# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:46:50 2022

@author: Yang Liu
"""
import torch
from torch import pinverse as pinv

class MINRES(object):
    def __init__(self, A, b, rtol=1E-4, maxit=100, M=None, shift=0, 
                 reorth=True, npc=True, inexactness="relres", 
                 eps=1E-8):
        '''
        Parameters
        ----------
        A : TYPE 
            Symmetrix
        b : TYPE 
            Same datetype as A.
        rtol : TYPE, optional
            DESCRIPTION. The default is 1E-4.
        maxit : TYPE, optional
            Maximum iteration. The default is 100.
        M : TYPE, optional
            Positive semedefinite Preconditioner for H and CS. The default is None.
        shift : TYPE, optional
            Perturbation. The default is 0.
        reorth : TYPE, optional
            Reorthogonalization. The default is True.
        npc : TYPE, optional
            Non-positive curvature detection. The default is True.
        inexactness : TYPE, optional
            Termination condition. 
            The default is "relHres" ==> || H r || \leq rtol || H x ||.
            "relres" ==> || r || \leq rtol || b ||
        eps : TYPE, optional
            Termination condition of machine error, can be large. 
            The default is 1E-2.
        '''
        # Initialize methods
        self.A = A
        self.b = b
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
        if M is None:
            self.precon = False
            self.beta1 = b.norm()
            self.vn = b.clone()/self.beta1
            self.v = torch.zeros_like(b)
            self.r = b.clone() # to aviod stupid bugs in pytorch
            self.hr = self.r
        else: # v represent w, but || w || \neq 1
            self.precon = True
            self.M = M
            if reorth == True:
                self.reorth = False
                self.preorth = True
            else:
                self.preorth = False
            self.zn = b
            self.z = torch.zeros_like(b)
            vn = torch.mv(M, self.zn)
            self.beta1 = torch.sqrt(torch.vdot(vn, self.zn))
            self.r = b.clone()
            self.vn = vn/self.beta1
            self.zn = b/self.beta1 # store zn = zn/betan
            self.hr = vn.clone()
        self.betan = self.beta1 # can be 0 for implementation purpuse
        self.cs = -self.beta1/self.beta1
        self.sn = 0*self.beta1
        self.tau = 0
        self.delta1 = 0
        self.epsilonn = 0
        self.phi = self.beta1 # intitialize residual M-norm
        self.relres = 1
        self.x = torch.zeros_like(b)
        self.d = torch.zeros_like(b)
        self.dl = torch.zeros_like(b)
        self.Adb = torch.zeros_like(b) # A^{\dagger} b --- Pseudo-inverse solution

    def run(self):
        while self.flag == -1:
            self._1run() # preform a (precondtioned) MINRES step
                 
            # More termination condition
            if self.iters >= self.maxit and self.flag == -1:                
                self.flag = 4
                self.dtype = "MAX"
                break
        return self.x, self.relres, self.iters, self.r, self.dtype

    def _1run(self):
        # Lanczos process
        self._lanczos()         
        self.iters += 1
        
        # QR decomposition
        self.epsilon = self.epsilonn
        self._qr() 
        
        # non-positive curvature detection
        npc = - self.cs * self.gamma1 
        a_ = torch.abs(self.gamma1)
        self.gamma2 = torch.sqrt(a_**2 + self.betan**2)
        if self.npc and npc < self.eps/10: # NPC detection for Hermitian system
            self.flag = 3
            self.dtype = 'NPC'
            print("NPC detected")
            return
            
        # inexactness checking
        if self.inexactness == "relres":
            inexactnesscheck = (self.relres < self.rtol)
        else:
            inexactnesscheck = (self.phi*torch.sqrt(
                self.gamma2**2 + self.delta1**2) < self.rtol*torch.sqrt(
                                                  self.beta1**2-self.phi**2))
        if inexactnesscheck:
            self.flag = 1  ## trustful least square solution
            # print("Inexactness reached")
            return
        
        # update
        if self.gamma2 > self.eps:
            self.cs = self.gamma1/self.gamma2
            self.sn = self.betan/self.gamma2
            self._updates()
            self.relres = self.phi/self.beta1
            if self.betan < self.eps:
                self.flag = 0
                if not self.precon:
                    self.Adb = self.x
                # print('Exact soltion')
                return
        else:
            ## gamma1 = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            self.cs = 0
            self.sn = 1
            self.gamma2 = 0  
            self.flag = 2
            print('Least-squares terminated, residual nonzero')
            return

    def _lanczos(self):
        q = self._Av(self.vn)   
        
        if self.shift == 0:
            pass
        else:
            q = q + self.shift*self.vn # A --> (A + shift*eye)
        self.alpha = torch.vdot(self.vn, q)
        if self.precon:
            zn = q - self.alpha*self.zn - self.betan*self.z # pLanczos
            self.z = self.zn.clone()
            self.v = self.vn.clone()
            if self.preorth and self.iters > 0:
                zn = zn - torch.mv(self.V, zn)
            vn = torch.mv(self.M,zn)
            if self.preorth and self.iters > 0:
                vn = vn - torch.mv(self.V.T, vn)
            self.betan = torch.sqrt(torch.vdot(vn, zn))
            self.vn = vn/self.betan # redefine vn = vn/betan
            self.zn = zn/self.betan # redefine zn = zn/betan
            if self.preorth:
                if self.iters == 0:
                    self.V = torch.outer(self.zn, self.vn)
                else:
                    self.V = self.V + torch.outer(self.zn, self.vn)
        else:
            vn = q - self.alpha*self.vn - self.betan*self.v # Lanczos
            self.v = self.vn.clone()
            self.betan = vn.norm()  
            self.vn = vn/self.betan 
            if self.reorth: # reorthogonalization
                if self.iters == 0:
                    self.V = self.v.reshape(-1, 1)
                    self.vn = self.vn - torch.vdot(self.v, self.vn)*self.v
                else:
                    self.vn = self.vn - torch.mv(self.V, torch.mv(self.V.T, self.vn))
                self.V = torch.cat((self.V, self.vn.reshape(-1, 1)), axis=1)
                
        
    def _qr(self):
        self.delta2 = self.cs*self.delta1 + self.sn*self.alpha
        self.gamma1 = self.sn*self.delta1 - self.cs*self.alpha
        self.epsilonn = self.sn*self.betan
        self.delta1 = -self.cs*self.betan

    def _updates(self):
        self.tau = self.cs*self.phi
        self.phi = self.sn*self.phi

        tmp = (self.v - self.delta2*self.d - self.epsilon*self.dl)/self.gamma2
        self.dl = self.d.clone()
        self.d = tmp
        self.x = self.x + self.tau*self.d
        
        self.hr = self.sn **2 * self.hr - self.phi * self.cs * self.vn
        if self.precon:
            self.r = self.sn **2 * self.r - self.phi * self.cs * self.zn
        else:
            self.r = self.hr
        # print("rr", self.r, self.b - self.A @ self.x, self.phi)

    def _Av(self, x):
        if callable(self.A):
            Ax = self.A(x)
        else:
            Ax = torch.mv(self.A, x)
        return Ax

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
    A : torch.float64, n x n
        A n x n complex symmetric matrix with rank r.
    U : torch.float64, n x r
        U matrix for SVD.
    s : torch.float64, n
        Singular values.

    '''
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

def main(): 
    """Run an example of precondtioned MINRES."""
    # torch.manual_seed(2)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n = 5
    m = 5
    b = torch.ones(n, device=device, dtype=torch.float64)
    E, U, s, _ = Symm(n)
    # d = n
    d = n-1
    s[d:] = -1 ## let A indefinite manully
    A = U @ torch.diag(s) @ U.T
    S = torch.randn(n, m, device=device, dtype=torch.float64)
    M = torch.mm(S, S.T)
    maxit = n
    rtol = 0
    
    # The preconditioned MINRES ||b - Ax|| with M = S S.T is mathematically equivelent 
    # to apply MINRES to the preconditioned system || S.T b - S.T A S tx || x = S tx
    AA = S.T @ A @ S
    bb = S.T @ b
    
    MRP = MINRES(AA, bb, rtol, maxit, npc=False)
    MRP.run()
    print("Verify MINRES implementation", MRP.relres)
    PMR = MINRES(A, b, rtol, maxit, M=M, npc=False)
    PMR.run()
    print("Verify preconditioning implementation", (S @ MRP.x - PMR.x).norm(), MRP.relres)
    
    MRPNPC = MINRES(AA, bb, rtol, maxit, npc=True)
    MRPNPC.run()
    print("NPC detection in MINRES on preconditioned LS", MRPNPC.r @ (AA @ MRPNPC.r))
    PMRNPC = MINRES(A, b, rtol, maxit, M=M, npc=True)
    PMRNPC.run()
    print("NPC detection in precondtioned MINRES", PMRNPC.hr @ (A @ PMRNPC.hr), 
          "not necessary NPC", PMRNPC.r @ (A @ PMRNPC.r))  
    print("Some Equivlency, har(r) = M r", (PMRNPC.hr - M @ PMRNPC.r).norm(), (PMRNPC.hr - S @ MRPNPC.r).norm())
    
if __name__ == '__main__':
    main()