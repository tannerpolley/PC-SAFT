import numpy as np
import numdifftools as nd
from scipy.optimize import root, least_squares
from gekko import GEKKO
import pyomo.environ as pyo

def PC_SAFT_GEKKO(T, z, m, σ, ϵ_k, k_ij, η=None, phase='liquid', P_sys=None, κ_AB=None, ϵ_AB_k=None):

    model = GEKKO()

    a_ni = model.Const(np.array([[0.9105631445, -0.3084016918, -0.0906148351],
                             [0.6361281449, 0.1860531159, 0.4527842806],
                             [2.6861347891, -2.5030047259, 0.5962700728],
                             [-26.547362491, 21.419793629, -1.7241829131],
                             [97.759208784, -65.255885330, -4.1302112531],
                             [-159.59154087, 83.318680481, 13.776631870],
                             [91.297774084, -33.746922930, -8.6728470368]]).T)

    b_ni = model.Const(np.array([[0.7240946941, -0.5755498075, 0.0976883116],
                     [2.2382791861, 0.6995095521, -0.2557574982],
                     [-4.0025849485, 3.8925673390, -9.1558561530],
                     [-21.003576815, -17.215471648, 20.642075974],
                     [26.855641363, 192.67226447, -38.804430052],
                     [206.55133841, -161.82646165, 93.626774077],
                     [-355.60235612, -165.20769346, -29.666905585]]).T)


    kb = model.Const(1.380649e-23)  # J/K
    N_A = model.Const(6.0221e23)  # 1/mol
    R = model.Const(8.314)  # J/mol-K
    π = model.Const(np.pi)

    # Parameters
    T = model.Param(T)
    T_og = model.Param(T)
    z = model.Param(z)
    z_og = model.Param(z)
    m = model.Param(m)
    k = model.Param(len(σ))
    σ = model.Param(σ)
    ϵ_k = model.Param(ϵ_k)
    κ_AB = model.Param(κ_AB)
    ϵ_AB_k = model.Param(ϵ_AB_k)
    η_diff = model.Param(False)
    T_diff = model.Param(False)
    x_diff = model.Param(False)
    d_static = model.Param(σ * (1 - .12 * np.exp(-3 * ϵ_k / T)))


    if κ_AB is None:
        κ_AB = model.Param(np.zeros(k))

    if ϵ_AB_k is None:
        ϵ_AB_k = model.Param(np.zeros(k))

    # --------------------------------------- Intermediates --------------------------------------- #

    σ_ij = model.Intermediate(np.array([[1 / 2 * (σ[i] + σ[j]) for j in range(k)] for i in range(k)]))
    ϵ_ij = model.Intermediate(np.array([[(ϵ_k[i] * ϵ_k[j]) ** (1 / 2) * (1 - k_ij[i][j]) for j in range(k)] for i in range(k)]))
    κ_AB_ij = model.Intermediate(np.array([[(κ_AB[i] * κ_AB[j]) ** (1 / 2) * ((σ[i] * σ[j]) / (1 / 2 * (σ[i] * σ[j]))) ** 3 for j in range(k)] for i in range(k)]))
    ϵ_AB_ij = model.Intermediate(np.array([[(ϵ_AB_k[i] + ϵ_AB_k[j]) / 2 for j in range(k)] for i in range(k)]))

    if P_sys is not None:
        P_sys = model.Param(P_sys)
        phase = model.Param(phase)
        model.find_η()
    elif P_sys is None and η is not None:
        model.η = η
    elif P_sys is None and η is None:
        if phase == 'liquid':
            model.η = .4
        elif phase == 'vapor':
            model.η = .01
        print(
            'Warning, a default η had to be defined based on the given phase since so system pressure was given to iteratively find η')


    def m_bar(model):
        z = model.z
        m = model.m

        return sum(z * m)


    def d(model):
        T = model.T
        σ = model.σ
        ϵ_k = model.ϵ_k

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))


    def d_og(model):
        T = model.T_og
        σ = model.σ
        ϵ_k = model.ϵ_k

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))


    def ρ(model):
        z = model.z_og
        η = model.η
        d = model.d_static
        m = model.m
        k = model.k
        return 6 / model.π * η * (sum([z[i] * m[i] * d[i] ** 3 for i in range(k)])) ** (-1)


    def v(model):
        ρ = model.ρ()
        return model.N_A * 10 ** -30 / ρ


    def ξ(model):
        z = model.z
        d = model.d()
        ρ = model.ρ()
        m = model.m
        return np.array([model.π / 6 * ρ * np.sum([z[i] * m[i] * d[i] ** n for i in range(model.k)]) for n in range(4)])


    def g_hs_ij(model):
        d = model.d()
        ξ = model.ξ()

        return np.array([[(1 / (1 - ξ[3])) +
                          ((d[i] * d[j] / (d[i] + d[j])) * 3 * ξ[2] / (1 - ξ[3]) ** 2) +
                          ((d[i] * d[j] / (d[i] + d[j])) ** 2 * 2 * ξ[2] ** 2 / (1 - ξ[3]) ** 3)
                          for j in range(model.k)]
                         for i in range(model.k)])


    def d_ij(model):
        d = model.d()
        return np.array([[1 / 2 * (d[i] + d[j]) for j in range(model.k)] for i in range(model.k)])


    def Δ_AB_ij(model):
        T = model.T
        d_ij = model.d_ij()
        g_hs_ij = model.g_hs_ij()
        return np.array([[d_ij[i][j] ** 3 * g_hs_ij[i][j] * model.κ_AB_ij[i][j] * (np.exp(model.ϵ_AB_ij[i][j] / T) - 1)
                          for j in range(model.k)] for i in range(model.k)])


    def a_hs(model):
        ξ = model.ξ()
        return 1 / ξ[0] * (3 * ξ[1] * ξ[2] / (1 - ξ[3]) + ξ[2] ** 3 / (ξ[3] * (1 - ξ[3]) ** 2) + (
                ξ[2] ** 3 / ξ[3] ** 2 - ξ[0]) * np.log(1 - ξ[3]))


    def a_hc(model):
        z = model.z

        k = model.k
        m = model.m
        m_bar = model.m_bar()
        g_hs_ij = model.g_hs_ij()
        a_hs = model.a_hs()

        return m_bar * a_hs - sum([z[i] * (m[i] - 1) * np.log(g_hs_ij[i][i]) for i in range(k)])


    def a_disp(model):
        T = model.T
        z = model.z
        η = model.ξ()[-1]

        a_ni = model.a_ni
        b_ni = model.b_ni
        π = model.π
        k = model.k
        ρ = model.ρ()
        m = model.m
        m̄ = model.m_bar()
        ϵ_ij = model.ϵ_ij
        σ_ij = model.σ_ij

        a = a_ni[0] + (m̄ - 1) / m̄ * a_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * a_ni[2]
        b = b_ni[0] + (m̄ - 1) / m̄ * b_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * b_ni[2]

        I1 = [a[i] * η ** i for i in range(7)]
        I2 = [b[i] * η ** i for i in range(7)]

        I1 = np.sum(I1)
        I2 = np.sum(I2)

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_1 = Σ_i

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) ** 2 * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_2 = Σ_i

        C1 = (1 + m̄ * (8 * η - 2 * η ** 2) / (1 - η) ** 4 + (1 - m̄) * (
                20 * η - 27 * η ** 2 + 12 * η ** 3 - 2 * η ** 4) / ((1 - η) * (2 - η)) ** 2) ** -1

        return -2 * π * ρ * I1 * Σ_1 - π * ρ * m̄ * C1 * I2 * Σ_2


    def a_assoc(model):
        z = model.z
        k = model.k
        ρ = model.ρ()
        ϵ_AB_k = model.ϵ_AB_k
        Δ_AB_ij = model.Δ_AB_ij()

        def XA_find(XA_guess, n, Δ_AB_ij, ρ, z):
            # m = int(XA_guess.shape[1] / n)
            AB_matrix = np.asarray([[0., 1.],
                                    [1., 0.]])
            Σ_2 = np.zeros((n,), dtype='float_')
            XA = np.zeros_like(XA_guess)

            for i in range(n):
                Σ_2 = 0 * Σ_2
                for j in range(n):
                    Σ_2 += z[j] * (XA_guess[j, :] @ (Δ_AB_ij[i][j] * AB_matrix))
                XA[i, :] = 1 / (1 + ρ * Σ_2)

            return XA

        a_sites = 2  # 2B association?
        i_assoc = np.nonzero(ϵ_AB_k)
        n_assoc = len(i_assoc)
        if n_assoc == 0 or n_assoc == 1:
            return 0
        XA = np.zeros((n_assoc, a_sites), dtype='float_')

        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, n_assoc, Δ_AB_ij, ρ, z[i_assoc])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten()

        return sum([z[i] * sum([np.log(XA[j] - 1 / 2 * XA[j] + 1 / 2) for j in range(len(XA))]) for i in range(k)])


    def a_ion(model):
        return 0


    def a_res(model):
        a_hc = model.a_hc()
        a_disp = model.a_disp()
        a_assoc = model.a_assoc()
        a_ion = model.a_ion()

        return a_hc + a_disp + a_assoc + a_ion


    def da_dη(model):
        η = model.η

        model.η_og = model.η

        def f(η_diff):
            model.η = η_diff
            return model.a_res()

        model.η = model.η_og

        return nd.Derivative(f)(η)


    def da_dx(model):
        z = model.z
        model.z_og = model.z
        da_dx = []
        for k in range(len(z)):

            def f(zk):

                z_new = []
                for i in range(len(z)):
                    if i == k:
                        z_new.append(zk)
                    else:
                        z_new.append(z[i])

                model.z = z_new

                return model.a_res()

            model.z = model.z_og

            da_dx.append(nd.Derivative(f)(z[k]))

        model.z = model.z_og

        return np.array(da_dx)


    def da_dT(model):
        T = model.T
        model.T_og = model.T

        def f(T_diff):
            model.T = T_diff
            return model.a_res()

        model.T = model.T_og
        return nd.Derivative(f)(T)


    def Z(model):
        η = model.η

        da_dη = model.da_dη()

        model.η = model.η_og

        return 1 + η * da_dη


    def P(model):
        T = model.T
        Z = model.Z()
        ρ = model.ρ()
        kb = model.kb

        return Z * kb * T * ρ * 10 ** 30


    def find_η(model):
        def f(ηg):
            model.η = float(ηg)
            P = model.P()
            P_sys = model.P_sys
            return (P - P_sys) / 100000

        phase = model.phase
        if phase == 'liquid':
            ηg = .5
        elif phase == 'vapor':
            ηg = 10e-10
        else:
            print('Phase spelling probably wrong or phase is missing')
            ηg = .01
        η = root(f, np.array([ηg])).x[0]

        model.η = η


    def h_res(model):
        T = model.T
        model.T_og = model.T

        Z = model.Z()
        da_dT = model.da_dT()

        model.T = model.T_og

        return -T * da_dT + (Z - 1)


    def s_res(model):
        T = model.T
        model.T_og = model.T

        a_res = model.a_res()
        Z = model.Z()
        da_dT = model.da_dT()

        model.T = model.T_og

        return -T * (da_dT + a_res / T) + np.log(Z)


    def g_res(model):
        a_res = model.a_res()
        Z = model.Z()
        return a_res + (Z - 1) - np.log(Z)


    def μ_res(model):
        z = model.z
        T = model.T

        a_res = model.a_res()
        Z = model.Z()
        da_dx = model.da_dx()

        Σ = np.sum([z[j] * da_dx[j] for j in range(len(z))])

        μ_res = [(a_res + (Z - 1) + da_dx[i] - Σ) * model.kb * T for i in range(len(z))]

        return np.array(μ_res)


    def φ(model):
        T = model.T
        μ_res = model.μ_res()
        Z = model.Z()

        return np.exp(μ_res / model.kb / T - np.log(Z))


def flash(x, y, T, P, m, σ, ϵ_k, k_ij, flash_type='Bubble_T', κ_AB=None, ϵ_AB_k=None):
    def eqs_to_solve(x, y, T, P, flash_type):

        mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

        φ_l = mix_l.φ()
        φ_v = mix_v.φ()

        eqs = [(y[i] * φ_v[i] - x[i] * φ_l[i]) for i in range(len(y))]
        if flash_type[:-2] == 'Bubble':
            eqs.append(1 - np.sum([y[i] for i in range(len(y))]))
        elif flash_type[:-2] == 'Dew':
            eqs.append(1 - np.sum([x[i] for i in range(len(x))]))
        else:
            print('Wrong Flash Type')

        return eqs

    def f(w):
        if flash_type == 'Bubble_T':
            return eqs_to_solve(x, w[:-1], T, w[-1], flash_type)
        elif flash_type == 'Bubble_P':
            return eqs_to_solve(x, w[:-1], w[-1], P, flash_type)
        elif flash_type == 'Dew_T':
            return eqs_to_solve(w[:-1], y, T, w[-1], flash_type)
        elif flash_type == 'Dew_P':
            return eqs_to_solve(w[:-1], y, w[-1], P, flash_type)
        else:
            print('Wrong Flash Type')
            return None

    if flash_type == 'Bubble_T':
        guesses = list(y) + [P]
    elif flash_type == 'Bubble_P':
        guesses = list(y) + [T]
    elif flash_type == 'Dew_T':
        guesses = list(x) + [P]
    elif flash_type == 'Dew_P':
        guesses = list(x) + [T]
    else:
        print('Wrong Flash Type')
        guesses = []
    options = {'xtol': 1e-4, }
    ans = root(f, np.array([guesses]), options=options).x
    return ans[:-1], ans[-1]


def BPT_gekko(x, y, T, P, m, σ, ϵ_k, k_ij, κ_AB=None, ϵ_AB_k=None):
    def fun(z):
        y1, y2, y3, P = z
        y = np.array([y1, y2, y3])

        mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

        φ_l = mix_l.φ()
        φ_v = mix_v.φ()

        eqs = [(y[i] * φ_v[i] - x[i] * φ_l[i]) for i in range(len(y))]
        eqs.append(1 - np.sum([y[i] for i in range(len(y))]))

        return eqs

    bnds = ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, np.inf])
    guesses = np.array([y[0], y[1], y[2], P])
    res = least_squares(fun, guesses, bounds=bnds, ftol=1e-3, xtol=1e-3)
    ans = res.x

    return ans[:-1], ans[-1]
