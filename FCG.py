import torch
    
def FCG(A, b, tol, maxiter, M=None, shift = 0):
    """
    Flexible Conjugate Gradient methods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
        M: preconditioner, function handle on v, returns (approx) M^{\dagger} v
        shift: input = A + shift * eye
    OUTPUT:
        x: best solution x
        rel_res: relative residual
        T: iterations
    """
    x = torch.zeros_like(b)
    r = b
    bnorm = b.norm() 
    T = 0
    if M is not None:
        p = precond(M, r)
    else:
        p = r
    z = p.clone()
    rTz = r @ z
    
    while T < maxiter:
        Ap = Ax(A, p) + shift*p
        T += 1 # Matrix-vector product being made
        pTAp = p @ Ap
        if pTAp <= 0:
            print('pTAp =', pTAp)
            raise ValueError('pTAp <= 0 in myCG')
        alpha = rTz/pTAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = torch.norm(r)/bnorm      
        if rel_res < tol:
            return x, rel_res, T
        zl = z.clone()
        if M is not None:
            z = precond(M, r)
        else:
            z = r
        rTzl = rTz
        rTz = r @ z
        beta = (rTz - r@zl)/rTzl # Flexible CG
        # beta = rTz/rTzl # Standard CG
        p = z + beta*p
    return x, rel_res, T
        

def myCG(A, b, tol, maxiter, shift = 0):
    """
    Conjugate gradient methods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
        shift: input = A + shift * eye
    OUTPUT:
        x: best solution x
        rel_res: relative residual
        T: iterations
    """
    x = torch.zeros_like(b)
    r = b
    bnorm = b.norm() 
    T = 0
    p = r.clone()
    rTr = r.norm()**2
    
    while T < maxiter:
        Ap = Ax(A, p) + shift*p
        T += 1 # Matrix-vector product being made
        pTAp = p @ Ap
        if pTAp <= 0:
            print('pTAp =', pTAp)
            raise ValueError('pTAp <= 0 in myCG')
        alpha = rTr/pTAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = torch.norm(r)/bnorm      
        if rel_res < tol:
            return x, rel_res, T
        rTrl = rTr
        rTr = r @ r
        beta = rTr/rTrl # Flexible CG
        # beta = rTz/rTzl # Standard CG
        p = r + beta*p
    return x, rel_res, T

def Ax(A, v):
    if callable(A):
        Ax = A(v)
    else:
        Ax =torch.mv(A, v)
    return Ax

def precond(M, v):
    if callable(M):
        h = M(v)
    else:
        h = torch.mv(torch.pinverse(M), v)
    return h

def main(): 
# =============================================================================
    # torch.manual_seed(1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n = 5
    A = torch.randn(n, n, device=device, dtype=torch.float64)
    C = torch.randn(n, n, device=device, dtype=torch.float64)
    b = torch.randn(n, device=device, dtype=torch.float64)
    B = A.T @ A
    x = FCG(B, b, 0, n, M = torch.eye(n, device=device, dtype=torch.float64)) # M = eye
    x2 = FCG(B, b, 0, n, M = C.T @ C) # M \succ \zero
    x3 = myCG(B, b, 1E-9, n) # CG
    PIS = torch.pinverse(B) @ b
    print(x[0] - PIS, x2[0] - PIS, x3[0] - PIS)
    print(' ')
    
if __name__ == '__main__':
    main()