import torch
    
def FCR(A, b, tol, maxiter, M=None, shift = 0):
    """
    Conjugate residual mathods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
        M: preconditioner function handle on v, returns M^{\dagger} v
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
    Az = Ax(A, z) + shift*z
    Ap = Az.clone()
    if M is not None:
        MAp = precond(M, Ap)
    else:
        MAp = Ap
    zTAz = z @ Az
    
    while T < maxiter:
        if zTAz <= 0:
            print('zTAz =', zTAz)
            raise ValueError('zTAz <= 0 in myCR')
        alpha = zTAz/(Ap @ MAp)
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = torch.norm(r)/bnorm      
        if rel_res < tol:
            return x, rel_res, T
        z = z - alpha*MAp
        Azl = Az.clone()
        Az = Ax(A,z)        
        T += 1 # Matrix-vector product being made
        zTAzl = zTAz
        zTAz = z @ Az
        beta = (zTAz - z@Azl)/zTAzl # Flexible CR
        # beta = zTAz/zTAzl # Standard CR
        p = z + beta*p
        Ap = Az + beta*Ap
        if M is not None:
            MAp = precond(M, Ap)
        else:
            MAp = Ap
    return x, rel_res, T
        
    
def myCR(A, b, tol, maxiter, shift = 0):
    """
    Conjugate residual mathods. Solve Ax=b for PD matrices.
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
    p = r
    Ar = Ax(A, r)
    Ap = Ar.clone()
    rTAr = r @ Ar
    
    while T < maxiter:
        # print(T)
        if rTAr <= 0:
            print('rTAr =', rTAr)
            raise ValueError('rTAr <= 0 in myCR')
        alpha = rTAr/Ap.norm()**2
        x = x + alpha*p
        r = r - alpha*Ap
        Ar = Ax(A,r)        
        T += 1 # Matrix-vector product being made
        rel_res = torch.norm(r)/bnorm      
        if rel_res < tol:
            return x, rel_res, T
        rTArl = rTAr
        rTAr = r @ Ar
        beta = rTAr/rTArl # Standard CR
        p = r + beta*p
        Ap = Ar + beta*Ap
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
    x = FCR(B, b, 0, n, M = torch.eye(n, device=device, dtype=torch.float64)) # M=eye
    x2 = FCR(B, b, 0, n, M = C.T @ C) # M \succ \zero
    x3 = myCR(B, b, 0, n) # CR
    PIS = torch.pinverse(B) @ b
    print(x[0] - PIS, x2[0] - PIS, x3[0] - PIS)
    print(' ')
    
if __name__ == '__main__':
    main()