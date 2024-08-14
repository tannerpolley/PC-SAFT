from numpy import exp, log, array
from scipy.optimize import minimize, root
from gekko import GEKKO

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation


def solve_ChemEQ(Cl_0, Tl, guesses=array([.005, 1800, 39500, 1300, 1300, 20])):

    a1, b1, c1 = 233.4, -3410, -36.8
    a2, b2, c2 = 176.72, -2909, -28.46

    K1 = exp(a1 + b1/Tl + c1*log(Tl))/1000 # kmol -> mol
    K2 = exp(a2 + b2/Tl + c2*log(Tl))/1000 # kmol -> mol

    def f(Cl):

        # Kee1 = log(Cl[3]) + log(Cl[4]) - log(Cl[0]) - 2*log(Cl[1])
        # Kee2 = log(Cl[3]) + log(Cl[5]) - log(Cl[0]) - log(Cl[1]) - log(Cl[2])

        Kee1 = Cl[3]*Cl[4]/(Cl[0]*Cl[1]**2)
        Kee2 = Cl[3]*Cl[5]/(Cl[0]*Cl[1]*Cl[2])

        eq1 = Kee1 - K1
        eq2 = Kee2 - K2
        eq3 = Cl_0[0] - (Cl[0] + Cl[3])
        eq4 = Cl_0[1] - (Cl[1] + Cl[3] + Cl[4])
        eq5 = Cl_0[2] - (Cl[2] + Cl[3] - Cl[4])
        eq6 = Cl[3] - (Cl[4] + Cl[5])

        eqs = array([eq1, eq2, eq3, eq4, eq5, eq6])
        # print(eqs)
        return eqs

    return array(root(f, guesses).x).astype('float')

def solve_ChemEQ_gekko(Cl_0, Tl, guesses=array([.005, 1800, 39500, 1300, 1300, 20])):

    m = GEKKO()

    Cl_CO2_0 = m.Param(Cl_0[0])
    Cl_MEA_0 = m.Param(Cl_0[1])
    Cl_H2O_0 = m.Param(Cl_0[2])

    Cl_CO2 = m.Var(guesses[0], lb=0)
    Cl_MEA = m.Var(guesses[1], lb=0)
    Cl_H2O = m.Var(guesses[2], lb=0)
    Cl_MEAH = m.Var(guesses[3], lb=0)
    Cl_MEACOO = m.Var(guesses[4], lb=0)
    Cl_HCO3 = m.Var(guesses[5], lb=0)

    # print(type(Cl_CO2.value[0]))

    Tl = m.Param(Tl)

    a1, b1, c1 = m.Param(233.4), m.Param(-3410), m.Param(-36.8)
    a2, b2, c2 = m.Param(176.72), m.Param(-2909), m.Param(-28.46)

    K1 = m.Intermediate(m.exp(a1 + b1/Tl + c1*m.log(Tl))/1000) # kmol -> mol
    K2 = m.Intermediate(m.exp(a2 + b2/Tl + c2*m.log(Tl))/1000) # kmol -> mol

    Kee1 = m.Intermediate(Cl_MEAH*Cl_MEACOO/(Cl_CO2*Cl_MEA**2))
    Kee2 = m.Intermediate(Cl_MEAH*Cl_HCO3/(Cl_CO2*Cl_MEA*Cl_H2O))

    m.Equation(Kee1 == K1)
    m.Equation(Kee2 == K2)
    m.Equation(Cl_CO2_0 == (Cl_CO2 + Cl_MEAH))
    m.Equation(Cl_MEA_0 == (Cl_MEA + Cl_MEAH + Cl_MEACOO))
    m.Equation(Cl_H2O_0 == (Cl_H2O + Cl_MEAH - Cl_MEACOO))
    m.Equation(Cl_MEAH == (Cl_MEACOO + Cl_HCO3))

    m.solve(disp=False)


    Cl = array([Cl_CO2.value[0], Cl_MEA.value[0], Cl_H2O.value[0], Cl_MEAH.value[0], Cl_MEACOO.value[0], Cl_HCO3.value[0]])

    return Cl


# Cl_0 = [1270.492077, 4553.735043, 38329.94835]
# Tl = 320
# Cl = solve_ChemEQ_gekko(Cl_0, Tl)
# print(Cl)


