import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.linalg
import scipy.io as scio
import time

from .dcroot import dcroot
from .refine_positions import refine_positions


def SNLExamples(problem_name, noise_type, n, m, nf, Range):
    class Params():
        pass

    def randistance_qi(P0, PP, Radius, nf, noise_type):
        dim, npts = PP.shape
        nfix = P0.shape[1]

        D0 = np.zeros((npts, nfix))
        DD = np.zeros((npts, npts))

        for j in range(npts):
            if nfix > 0:
                tmp = np.matmul(PP[:, j].reshape((dim, 1)), np.ones((1, nfix))) - P0
                rr = np.sqrt(np.sum(tmp ** 2, axis=0))
                idx = np.where(rr < Radius)[0]
                rr = rr[rr < Radius]
                if idx.shape[0]:
                    if noise_type == 'multiplicative':
                        rand_scale = 1 + norm.ppf(np.random.rand(1, idx.shape[0])) * nf
                        # rand_scale = 1
                        rr = rr * rand_scale
                    D0[j, idx] = rr
            if j > 0:
                tmp = np.matmul(PP[:, j].reshape((dim, 1)), np.ones((1, j))) - PP[:, 0:j]
                rr = np.sqrt(np.sum(tmp ** 2, axis=0))
                idx = np.where(rr < Radius)[0]
                rr = rr[rr < Radius]
                if idx.shape[0]:
                    if noise_type == 'multiplicative':
                        rand_scale = 1 + norm.ppf(np.random.rand(1, idx.shape[0])) * nf
                        # rand_scale = 1
                        rr = rr * rand_scale
                    DD[idx, j] = rr
        DD = np.triu(DD, 1) + np.triu(DD, 1).transpose()
        Dall = np.hstack([DD, D0])
        return Dall

    if problem_name == 'BLTWY06_Inner':
        dim = 2
        m = 4
        s = n - m
        B = 0.2 * np.asarray([[1, 1, -1, -1],
                              [1, -1, 1, -1]])
        PA = B
        PS = -0.5 + np.random.random((s, 2)).transpose()
        PP = np.hstack([PA, PS])
    elif problem_name == "3DSNLsymetric":
        dim = 3
        m = 8
        # PA = 0.1 * np.asarray([[3, 7, 7, 3, 3, 7, 7, 3],
        #                        [3, 3, 7, 7, 3, 3, 7, 7],
        #                        [3, 3, 3, 3, 7, 7, 7, 7]])
        PA = np.asarray([[3, 7, 7, 3, 3, 7, 7, 3],
                         [3, 3, 7, 7, 3, 3, 7, 7],
                         [3, 3, 3, 3, 7, 7, 7, 7]]) * 10
        PS = np.random.random((n - m, 3)).transpose() * 100
        PP = np.hstack([PA, PS])

    D = randistance_qi(PA, PS, Range, nf, noise_type)
    if m > 0:
        D0 = squareform(pdist(PA.transpose()))
        D = np.vstack([np.hstack([D0, D[:, n - m:n].transpose()]), \
                       np.hstack([D[:, n - m:n], D[0:n - m, 0:n - m]])])

    pars = Params()
    pars.m = m
    pars.range = Range
    pars.PP = PP
    pars.draw = True

    return D, dim, pars


def JXJ(X):
    nX = X.shape[0]
    Xe = np.sum(X, axis=1).reshape((nX, 1))
    eXe = np.sum(Xe)
    JXJ = np.tile(Xe, (1, nX))
    JXJt = np.tile(Xe.transpose(), (nX, 1))
    JXJ = -(JXJ + JXJt) / nX
    JXJ = JXJ + X + eXe / (nX ** 2)
    return JXJ


def ProjKr(A, r):
    JAJ = JXJ(A)
    JAJ = (JAJ + JAJ.transpose()) / 2
    P0, V0 = scipy.sparse.linalg.eigs(JAJ, k=r, which="LM")
    P0 = np.diag(P0.real)
    V0 = V0.real
    Z0 = np.matmul(V0, np.matmul(np.maximum(np.zeros_like(P0), P0), \
                                 V0.transpose())) + A - JAJ
    return Z0


def FNorm(x):
    return np.linalg.norm(x, ord="fro") ** 2


def WNorm(x):
    return np.linalg.norm(x, ord=1)


def procrustes_qi(A, P):
    m0 = A.shape[1]
    a0 = np.sum(A, axis=1, keepdims=True) / m0
    p0 = np.sum(P, axis=1, keepdims=True) / m0
    A = A - a0[:, np.zeros((m0), dtype=int)]
    P1 = P - p0[:, np.zeros((m0), dtype=int)]
    P = np.matmul(P1, A.transpose())
    U0, _, V = np.linalg.svd(P)
    V = V.transpose()
    Q = np.matmul(U0, V.transpose())
    P = np.matmul(Q.transpose(), P1) + a0[:, np.zeros((m0), dtype=int)]
    return Q, P, a0, p0


def procrustes_zhou(m0, PP, D0):
    r0 = PP.shape[0]
    n0 = D0.shape[1]
    JDJ = JXJ(D0)
    JDJ = -(JDJ + JDJ.transpose()) / 4
    D1, U1 = scipy.sparse.linalg.eigs(JDJ, k=r0, which="LM")
    U1 = U1.real
    D1 = np.diag(D1.real)
    # U1[:, 0] *= -1

    # D1_full, U1_full = scipy.linalg.eig(JDJ)
    # D1 = np.diag(D1_full[:r0].real)
    # U1 = U1_full[:, :r0].real
    X0 = np.matmul(D1 ** (1 / 2), U1.transpose())
    if m0 > 0:
        A = PP[:, 0:m0]
        Q, _, a0, p0 = procrustes_qi(A, X0[:, 0:m0])
        Z0 = np.matmul(Q.transpose(), X0 - p0[:, np.zeros(n0, dtype=int)]) + \
             a0[:, np.zeros(n0, dtype=int)]
        Xs = Z0[:, m0:n0]
        Xa = Z0[:, 0:m0]
        Xs = Xs * np.max([1.0, np.sum(Xa * A) / np.linalg.norm(Xa, "fro") ** 2])
    return Xs


def PREEEDM(D, dim, pars):
    class Out():
        pass

    def get_parameters(nD, pars):
        m = 0
        itmax = 2000
        Otol = np.log(nD) * 1e-4
        Etol = 1e-2

        if "Otol" in dir(pars):
            Otol = pars.Otol
        if "Etol" in dir(pars):
            Etol = pars.Etol
        if "m" in dir(pars):
            m = pars.m

        return m, itmax, Etol, Otol

    def plot_SNL(X, RMSD, refine, pars, ax=None):
        PP = pars.PP
        r, n = PP.shape
        if "m" in dir(pars):
            m = pars.m
        Ta = slice(m)
        Ts = slice(m, n)
        plt.rcParams['axes.titlesize'] = 10
        if r == 2:
            if refine:
                plt.title("rRMSD {}".format(RMSD))
            else:
                plt.title("RMSD {}".format(RMSD))
            plt.scatter(PP[0, Ts], PP[1, Ts], marker="o", s=9, facecolors="none", edgecolors="b")
            plt.scatter(PP[0, Ta], PP[1, Ta], marker="s", s=10,
                        facecolors="none", edgecolors="g")
            plt.scatter(X[0, :], X[1, :], c="m", marker="*", s=3)
            for i in range(n - m):
                plt.plot([X[0, i], PP[0, m + i]], [X[1, i], PP[1, m + i]],
                         c="b", linewidth=0.5)
        elif r == 3:
            if refine:
                plt.title("rRMSD {}".format(RMSD))
            else:
                plt.title("RMSD {}".format(RMSD))
            if ax:
                ax.scatter(PP[0, Ts], PP[1, Ts], PP[2, Ts], marker="o", s=9, facecolors="none", edgecolors="b")
                ax.scatter(X[0, :], X[1, :], X[2, :], c="m", marker="*", s=3)
                for i in range(n - m):
                    ax.plot([X[0, i], PP[0, m + i]], [X[1, i], PP[1, m + i]],
                            [X[2, i], PP[2, m + i]], c="b", linewidth=0.5)

        return

    start_time = time.time()
    n = D.shape[0]
    nD = np.where(D)[0].shape[0]
    rate = nD / (n * n)
    m, itmax, Eigtol, Objtol = get_parameters(nD, pars)

    if rate < 0.9:
        DD = D ** 2
    # fSD = scio.loadmat("fSD_3DSNLsymetric_seed2_scale100.mat")["fSD"]
    fSD = scio.loadmat("fSD_BLTWY06_Inner_seed1.mat")["fSD"]
    scale = np.max(fSD)
    Do = D
    if scale <= 10:
        scale = 1
    else:
        D /= scale
        fSD /= scale

    DD = D ** 2
    H = np.zeros_like(D)
    H[np.where(D)] = 1
    r = dim
    T = slice(0, m)
    if m > 0:
        H[T, T] = 0

    Z = fSD ** 2
    UB = n * np.max(Z)
    L = np.zeros(n)
    U = UB * np.ones(n)

    if "range" in dir(pars):
        H1 = 1 - H
        rs = (pars.range / scale) ** 2
        L = L * H + H1 * rs
        U = U * H1 + H * rs

    L[T, T] = Z[T, T]
    U[T, T] = Z[T, T]

    np.fill_diagonal(L, 0)
    np.fill_diagonal(U, 0)

    Z = np.minimum(U, np.maximum(L, Z))
    rho = np.sqrt(n) * rate * np.max(D)

    Hr = H / rho
    TH = np.where(Hr > 0)
    PZ = ProjKr(-Z, r)
    frZ = WNorm(np.sqrt(Z[TH]) - D[TH]) + \
          (rho / 2) * FNorm(Z + PZ)
    OErr = np.zeros((itmax, 1))
    EErr = np.zeros((itmax, 1))

    for iter in range(1, itmax):
        Z = dcroot(-PZ, Hr, TH, DD, L, U)
        PZ = ProjKr(-Z, r)

        gZ = FNorm(Z + PZ)
        ErrEig = gZ / FNorm(JXJ(-Z))
        frZo = frZ
        frZ = WNorm(np.sqrt(Z[TH]) - D[TH]) + (rho / 2) * gZ
        ErrObj = np.abs(frZo - frZ) / (1 + rho + frZo)
        print("Iter: {}, ErrEig: {}, ErrObj {}".format(iter, ErrEig, ErrObj))

        if iter >= 5 and ErrEig < Eigtol and ErrObj < Objtol:
            break

        OErr[iter - 1, 0] = ErrObj
        EErr[iter - 1, 0] = ErrEig

        if ErrEig > Eigtol and ErrObj < Objtol / 5 and \
                iter >= 10 and np.var(EErr[iter - 10:iter, 0]) / ErrObj < 1e-4:
            rho *= 1.25
            Hr = H / rho
            Objtol = np.min(Objtol, np.max(Objtol / 1.1, ErrObj))
        if ErrObj > Objtol and ErrEig < Eigtol / 5:
            if iter <= 5:
                rho = 0.5 * rho
                Hr = H / rho
            elif iter > 5 and np.var(OErr[iter - 5:iter, 0]) / ErrObj < 1e-4:
                rho = 0.75 * rho
                Hr = H / rho

    if "PP" in dir(pars):
        T1 = slice(m, n)
        PPs = pars.PP / scale
        Xs = procrustes_zhou(m, PPs, Z)
        out = Out()
        out.Time = time.time() - start_time
        rX, refine_info = refine_positions(Xs, PPs[:, T],
                                           np.hstack([D[T1, T1], D[T1, T]]))
        out.rTime = refine_info.cpu_time
        out.Z = Z * scale ** 2
        out.X = Xs * scale
        out.rX = rX * scale
        out.Time = out.Time + out.rTime
        out.RMSD = np.sqrt(FNorm(pars.PP[:, T1] - out.X) / (n - m))
        out.rRMSD = np.sqrt(FNorm(pars.PP[:, T1] - out.rX) / (n - m))

        print("Time {} s".format(out.Time))
        print("Refinement Time {} s".format(out.rTime))
        print("RMSD {}".format(out.RMSD))
        print("rRMSD {}".format(out.rRMSD))

        if "draw" in dir(pars):
            fig = plt.figure()
            if dim == 2:
                plt.subplots_adjust(wspace=0, hspace=0.3)
                plt.subplot(2, 1, 1)
                plot_SNL(out.X, out.RMSD, False, pars)
                plt.subplot(2, 1, 2)
                plot_SNL(out.rX, out.rRMSD, True, pars)
                plt.show()
            elif dim == 3:

                plt.subplots_adjust(wspace=0, hspace=0.3)
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                plot_SNL(out.X, out.RMSD, False, pars, ax)
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                plot_SNL(out.rX, out.rRMSD, True, pars, ax)
                plt.show()

if __name__=="__main__":
    np.random.seed(1)

    pro = 1
    n = 500
    nf = 0.1

    problem = ['BLTWY06_Inner', "3DSNLsymetric", 'EDM']
    noisetype = 'multiplicative'

    if pro < 3:
        # R = 50
        R = 0.2
        m = 4
        D, dim, pars = SNLExamples(problem[pro - 1], noisetype, n, m, nf, R)

    point_pos = pars.PP
    dist_mat = squareform(pdist(point_pos.transpose()))
    Xs = procrustes_zhou(m, point_pos, dist_mat ** 2)
    # print("RMSD {}".format(np.sqrt(FNorm(point_pos[:, m:] - Xs) / (n - m))))

    # out = PREEEDM(D, dim, pars)
    out = PREEEDM(dist_mat, dim, pars)
