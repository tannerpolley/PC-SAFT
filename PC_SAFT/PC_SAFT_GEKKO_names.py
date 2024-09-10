import numpy as np
from gekko import GEKKO


def PC_SAFT_bubble_T(x, yg, T, Pg, m_seg, σ, ϵ_k, k_ij, κ_AB=None, ϵ_AB_k=None):

    def get_i_name(name_i):
        names = [m._intermediates[i].name for i in range(len(m._intermediates))]
        if name_i in names:
            count = sum([name_i in name for name in names])
            name_i += ' - ' + str(count)
        return name_i

    def f_m_bar(m, z):
        return m.Intermediate(sum([z[i] * m_seg[i] for i in range(k)]), name=get_i_name('m_bar'))

    def f_v(m, z, eta, T):
        ρ = f_ρ(m, z, eta, T)
        return N_A * 10 ** -30 / ρ

    def f_d(m, T):

        return [m.Intermediate(σ[i] * (1 - .12 * exp(-3 * ϵ_k[i] / T)), name=get_i_name(f'd_{i + 1}')) for i in
                range(k)]

    def f_ρ(m, z, eta, T):
        d = f_d(m, T)
        name = get_i_name('rho')
        return m.Intermediate(6 / π * eta * sum([(z[i] * m_seg[i] * d[i] ** 3) for i in range(k)]) ** (-1), name=name)

    def f_ξ(m, z, T, ρ):
        d = f_d(m, T)
        return [m.Intermediate(π / 6 * ρ * sum([z[i] * m_seg[i] * d[i] ** n for i in range(k)]),
                               name=get_i_name(f'xi{n + 1}')) for n in range(4)]

    def f_g_hs_ij(m, z, T, ρ):
        d = f_d(m, T)
        ξ = f_ξ(m, z, T, ρ)
        return [[m.Intermediate((1 / (1 - ξ[3])) +
                                ((d[i] * d[j] / (d[i] + d[j])) * 3 * ξ[2] / (1 - ξ[3]) ** 2) +
                                ((d[i] * d[j] / (d[i] + d[j])) ** 2 * 2 * ξ[2] ** 2 / (1 - ξ[3]) ** 3),
                                name=get_i_name(f'g_hs_{i + 1}{j + 1}'))
                 for j in range(k)]
                for i in range(k)]

    def f_d_ij(m, d):
        return [[m.Intermediate(1 / 2 * (d[i] + d[j]), name=get_i_name(f'd_{i + 1}{j + 1}')) for j in range(k)] for i in
                range(k)]

    # Δ_AB_ij = m.Intermediate(
    #     [[d_ij[i][j] ** 3 * g_hs_ij[i][j] * m.κ_AB_ij[i][j] * (exp(m.ϵ_AB_ij[i][j] / T) - 1)
    #                for j in range(k)] for i in range(k)])

    def f_a_hs(m, z, T, ρ):
        ξ = f_ξ(m, z, T, ρ)
        return m.Intermediate(1 / ξ[0] * (3 * ξ[1] * ξ[2] / (1 - ξ[3]) + ξ[2] ** 3 / (ξ[3] * (1 - ξ[3]) ** 2) + (
                ξ[2] ** 3 / ξ[3] ** 2 - ξ[0]) * log(1 - ξ[3])), name=get_i_name('a_hs'))

    def f_a_hc(m, z, T, ρ):
        m_bar = f_m_bar(m, z)
        g_hs_ij = f_g_hs_ij(m, z, T, ρ)
        a_hs = f_a_hs(m, z, T, ρ)
        return m.Intermediate(m_bar * a_hs - sum([z[i] * (m_seg[i] - 1) * log(g_hs_ij[i][i]) for i in range(k)]),
                              name=get_i_name('a_hc'))

    def f_a_disp(m, z, T, ρ):
        m_bar = f_m_bar(m, z)
        eta = f_ξ(m, z, T, ρ)[-1]

        a = [
            a_ni[0, i] + (m_bar - 1) / m_bar * a_ni[1, i] + (m_bar - 1) / m_bar * (m_bar - 2) / m_bar * a_ni[2, i]
            for i in range(7)]
        b = [
            b_ni[0, i] + (m_bar - 1) / m_bar * b_ni[1, i] + (m_bar - 1) / m_bar * (m_bar - 2) / m_bar * b_ni[2, i]
            for i in range(7)]

        I1 = sum([a[i] * eta ** i for i in range(7)])
        I2 = sum([b[i] * eta ** i for i in range(7)])

        Σ_1 = sum([sum([z[i] * z[j] * m_seg[i] * m_seg[j] * (ϵ_ij[i][j] / T) * σ_ij[i][j] ** 3
                        for j in range(k)])
                   for i in range(k)])
        Σ_2 = sum([sum([z[i] * z[j] * m_seg[i] * m_seg[j] * (ϵ_ij[i][j] / T) ** 2 * σ_ij[i][j] ** 3
                        for j in range(k)])
                   for i in range(k)])

        C1 = (1 + m_bar * (8 * eta - 2 * eta ** 2) / (1 - eta) ** 4 +
              (1 - m_bar) * (20 * eta - 27 * eta ** 2 + 12 * eta ** 3 - 2 * eta ** 4) / (
                          (1 - eta) * (2 - eta)) ** 2) ** -1
        return m.Intermediate(-2 * π * ρ * I1 * Σ_1 - π * ρ * m_bar * C1 * I2 * Σ_2, name=get_i_name('a_disp'))

    def f_a_res(m, z, T, ρ):

        a_hc = f_a_hc(m, z, T, ρ)
        a_disp = f_a_disp(m, z, T, ρ)
        return m.Intermediate(a_hc + a_disp, name=get_i_name('a_res'))

    def f_da_res_deta(m, z, eta, T):
        δ = .0001
        h = eta * δ
        eta1 = eta - 2 * h
        eta2 = eta - 1 * h
        eta3 = eta + 1 * h
        eta4 = eta + 2 * h
        a_res_1 = f_a_res(m, z, T, f_ρ(m, z, eta1, T))
        a_res_2 = f_a_res(m, z, T, f_ρ(m, z, eta2, T))
        a_res_3 = f_a_res(m, z, T, f_ρ(m, z, eta3, T))
        a_res_4 = f_a_res(m, z, T, f_ρ(m, z, eta4, T))
        return m.Intermediate((a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h),
                              name=get_i_name('da_res_deta'))

    def f_da_res_dT(m, z, eta, T):
        δ = .0001
        h = T * δ
        ρ = f_ρ(m, z, eta, T)
        a_res_1 = f_a_res(m, z, T - 2 * h, ρ)
        a_res_2 = f_a_res(m, z, T - 1 * h, ρ)
        a_res_3 = f_a_res(m, z, T + 1 * h, ρ)
        a_res_4 = f_a_res(m, z, T + 2 * h, ρ)

        return m.Intermediate((a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h), name=get_i_name('da_res_dT'))

    def f_da_res_dz(m, z, eta, T, j):
        δ = .0001
        h = z[j] * δ

        diff = [-2 * h, -h, h, 2 * h]
        z_new = m.Array(m.Param, (4, 3))
        for n in range(4):
            for i in range(len(z)):
                if i == j:
                    z_new[n, i] = z[i] + diff[n]
                else:
                    z_new[n, i] = z[i]

        a_res_1 = f_a_res(m, z_new[0], T, f_ρ(m, z, eta, T))
        a_res_2 = f_a_res(m, z_new[1], T, f_ρ(m, z, eta, T))
        a_res_3 = f_a_res(m, z_new[2], T, f_ρ(m, z, eta, T))
        a_res_4 = f_a_res(m, z_new[3], T, f_ρ(m, z, eta, T))
        return m.Intermediate((a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h), name=get_i_name('da_res_dz'))

    def f_Z(m, z, eta, T):
        da_res_deta = f_da_res_deta(m, z, eta, T)
        return m.Intermediate(1 + eta * da_res_deta, name=get_i_name('z'))

    def f_P(m, z, eta, T):
        Z = f_Z(m, z, eta, T)
        ρ = f_ρ(m, z, eta, T)
        return m.Intermediate(Z * kb * T * ρ * 10 ** 30, name=get_i_name('p_model'))

    def find_η(z, Pg, T, phase):

        if phase == 'liquid':
            ηg = .5
        elif phase == 'vapor':
            ηg = 10e-10
        else:
            print('Phase spelling probably wrong or phase is missing')
            ηg = .01
        m2 = GEKKO(remote=False)
        η = m2.Var(value=ηg)
        if not isinstance(z[0], float):
            z = [z[i].VALUE.value for i in range(len(z))]
        P_sys_2 = m2.Intermediate(f_P(m2, z, η, T))
        Pg_fixed = Pg.VALUE.value
        m2.Equation(0 == (Pg_fixed - P_sys_2) / 100000)
        m2.solve(disp=False)

        return η.value[0]

    def f_h_res_RT(m, z, eta, T):
        da_res_dT = f_da_res_dT(m, z, eta, T)
        Z = f_Z(m, z, eta, T)

        return m.Intermediate(-T * da_res_dT + (Z - 1), name=get_i_name('h_res_RT'))

    def f_s_res_RT(m, z, eta, T):
        ρ = f_ρ(m, z, eta, T)
        da_res_dT = f_da_res_dT(m, z, eta, T)
        a_res = f_a_res(m, z, T, ρ)
        Z = f_Z(m, z, eta, T)

        return m.Intermediate(-T * (da_res_dT + a_res / T) + log(Z), name=get_i_name('s_res_RT'))

    def f_g_res_RT(m, z, eta, T):
        ρ = f_ρ(m, z, eta, T)
        a_res = f_a_res(m, z, T, ρ)
        Z = f_Z(m, z, eta, T)
        name = get_i_name('g_res_RT')
        return m.Intermediate(a_res + (Z - 1) - log(Z), name=get_i_name('g_res_RT'))

    def f_μ_res_kT(m, z, eta, T):
        ρ = f_ρ(m, z, eta, T)
        a_res = f_a_res(m, z, T, ρ)
        Z = f_Z(m, z, eta, T)
        da_res_z = [f_da_res_dz(m, z, eta, T, i) for i in range(len(z))]
        Σ = sum([z[i] * da_res_z[i] for i in range(len(z))])
        μ_res = [m.Intermediate((a_res + (Z - 1) + da_res_z[i] - Σ), name=get_i_name(f'mu_res_kt_{i + 1}')) for i in
                 range(len(z))]

        return μ_res

    def f_φ(m, z, P, T, phase):
        eta = find_η(z, P, T, phase)
        μ_res_kT = f_μ_res_kT(m, z, eta, T)
        Z = f_Z(m, z, eta, T)

        return [m.Intermediate(exp(μ_res_kT[i] - log(Z)), name=get_i_name(f'phi_{i + 1}')) for i in range(3)]

    a_ni = np.array([[0.9105631445, -0.3084016918, -0.0906148351],
                     [0.6361281449, 0.1860531159, 0.4527842806],
                     [2.6861347891, -2.5030047259, 0.5962700728],
                     [-26.547362491, 21.419793629, -1.7241829131],
                     [97.759208784, -65.255885330, -4.1302112531],
                     [-159.59154087, 83.318680481, 13.776631870],
                     [91.297774084, -33.746922930, -8.6728470368]]).T

    b_ni = np.array([[0.7240946941, -0.5755498075, 0.0976883116],
                     [2.2382791861, 0.6995095521, -0.2557574982],
                     [-4.0025849485, 3.8925673390, -9.1558561530],
                     [-21.003576815, -17.215471648, 20.642075974],
                     [26.855641363, 192.67226447, -38.804430052],
                     [206.55133841, -161.82646165, 93.626774077],
                     [-355.60235612, -165.20769346, -29.666905585]]).T

    k = len(x)

    σ_ij = [[1 / 2 * (σ[i] + σ[j]) for j in range(k)] for i in range(k)]
    ϵ_ij = [[(ϵ_k[i] * ϵ_k[j]) ** (1 / 2) * (1 - k_ij[i][j]) for j in range(k)] for i in range(k)]
    # κ_AB_ij = [[(κ_AB[i] * κ_AB[j]) ** (1 / 2) * ((σ[i] * σ[j]) / (1 / 2 * (σ[i] * σ[j]))) ** 3 for j in range(k)]
    #     for i in range(k)]
    # ϵ_AB_ij = [[(ϵ_AB_k[i] + ϵ_AB_k[j]) / 2 for j in range(k)] for i in range(k)]

    # if κ_AB is None:
    #     κ_AB = np.zeros(k)
    #
    # if ϵ_AB_k is None:
    #     ϵ_AB_k = np.zeros(k)

    kb = 1.380649e-23  # J/K
    N_A = 6.0221e23  # 1/mol
    π = np.pi
    m = GEKKO(remote=False)

    # Gekko Functions
    exp = m.exp
    log = m.log
    sum = np.sum

    y = [m.Var(value=yg[i], lb=0, ub=1, name=f'y_{i + 1}') for i in range(len(yg))]
    P = m.Var(value=5e5, lb=0, ub=1e7, name=f'P')
    φv = [m.Intermediate(f_φ(m, y, P, T, 'vapor')[i]) for i in range(len(yg))]
    φl = [m.Intermediate(f_φ(m, x, P, T, 'liquid')[i]) for i in range(len(yg))]

    m.Equation([y[i] * φv[i] == x[i] * φl[i] for i in range(3)])
    m.Equation(1 == sum(y))
    m.options.IMODE = 1
    m.options.SOLVER = 3
    # m.open_folder()
    m.solve(disp=False)
    return φl[0].value
