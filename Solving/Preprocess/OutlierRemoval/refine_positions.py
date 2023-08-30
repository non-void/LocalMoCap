import numpy as np
import scipy
import scipy.sparse
import time


class Info():
    pass


def sparse(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return scipy.sparse.csr_matrix((v, (i, j)), shape=(m, n))


def objfun2(X, S, dd):
    Xij = X @ S
    normXij = np.sqrt(np.sum(Xij ** 2, axis=0))
    objval = np.linalg.norm(normXij - dd, ord=2) ** 2
    return objval


def gradfun2(X, S, dd, Sg):
    ne = dd.shape[0]
    Xij = X @ S
    normXij = np.sqrt(np.sum(Xij ** 2, axis=0)) + np.finfo(np.float64).eps
    tmp = 1 - dd / normXij
    G = Xij @ scipy.sparse.spdiags(2 * tmp, 0, ne, ne)
    G = G @ Sg
    return G


def refine_positions(X0, anchor, DD, maxit=500, tol=1e-9):
    start_time = time.time()
    _, n = X0.shape
    if n >= 1000:
        tol = 1e-7
    _, m = anchor.shape
    D1 = DD[0:n, 0:n]
    II, JJ = np.where(np.triu(D1))
    sortedIIJJ = np.vstack([II, JJ])[:, np.lexsort((II, JJ))]
    II, JJ = sortedIIJJ[0], sortedIIJJ[1]
    dd = D1[II, JJ]
    ne = II.shape[0]
    S = sparse(II, np.arange(ne), np.ones(ne), n + m, ne) - \
        sparse(JJ, np.arange(ne), np.ones(ne), n + m, ne)
    if m > 0:
        D2 = DD[0:n, n:n + m]
        I2, J2 = np.where(D2)
        sortedI2J2 = np.vstack([I2, J2])[:, np.lexsort((I2, J2))]
        I2, J2 = sortedI2J2[0], sortedI2J2[1]
        d2 = D2[I2, J2].reshape((-1, 1))
        # if d2.shape[0] < d2.shape[1]:
        #     d2 = d2.transpose()
        ne2 = I2.shape[0]
        S2 = sparse(I2, np.arange(ne2), np.ones(ne2), n + m, ne2) - \
             sparse(n + J2, np.arange(ne2), np.ones(ne2), n + m, ne2)
        S = scipy.sparse.hstack([S, S2])
        dd = np.hstack([dd, d2.reshape((-1))])
    Sg = S.transpose().tocsr()
    Sg = scipy.sparse.hstack([Sg[0:dd.shape[0], 0:n], sparse([], [], [], dd.shape[0], m)])
    X0 = np.hstack([X0, anchor])
    obj = objfun2(X0, S, dd)
    gradx = gradfun2(X0, S, dd, Sg)

    info = Info()
    info.objective = np.zeros(maxit + 1)
    info.grad_norm = np.zeros(maxit + 1)
    info.step_length = np.zeros(maxit + 1)
    info.objective[0] = obj
    info.grad_norm[0] = np.sqrt(np.max(np.sum(gradx ** 2)))

    Xiter = X0
    objold = obj
    for iter in range(maxit):
        objnew = float("inf")
        gradx = gradfun2(Xiter, S, dd, Sg)
        alpha = 0.2
        count = 0
        while (objnew > objold) and count < 20:
            alpha = 0.5 * alpha
            count = count + 1
            Xnew = Xiter - alpha * gradx
            objnew = objfun2(Xnew, S, dd)
        Xiter = Xnew
        info.objective[iter + 1] = objnew
        info.grad_norm[iter + 1] = np.sqrt(np.max(np.sum(gradx ** 2)))
        info.step_length[iter + 1] = alpha
        err = np.abs(objnew - objold) / (1 + abs(objold))
        if err < tol:
            break
        objold = objnew
    if m > 0:
        Xiter = Xiter[:, 0:n]
    info.cpu_time = (time.time() - start_time)
    info.objective = info.objective[0:iter]
    info.grad_norm = info.grad_norm[0:iter]
    info.step_length = info.step_length[0:iter]

    return Xiter, info
