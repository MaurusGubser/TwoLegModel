from sympy import symbols, Function, Matrix, diff, cos, sin, simplify, \
    init_printing
from IPython.display import display

# two-leg-model parameters
s0, s1, s2, s3 = symbols('s_0 s_1 s_2 s_3')
l0, l1, l2, l3 = symbols('l_0 l_1 l_2 l_3')
g = symbols('g')
t = symbols('t')

# state functions
x = Function('x_0')(t)
y = Function('x_1')(t)
phi0 = Function('phi_0')(t)
phi1 = Function('phi_1')(t)
phi2 = Function('phi_2')(t)
phi3 = Function('phi_3')(t)

# Rotation matrices for limb-aligned coordinate systems
Rphi0 = Matrix([[cos(phi0), sin(phi0)], [-sin(phi0), cos(phi0)]])
Rphi1 = Matrix([[cos(phi0 + phi1), sin(phi0 + phi1)],
                [-sin(phi0 + phi1), cos(phi0 + phi1)]])
Rphi2 = Matrix([[cos(phi2), sin(phi2)], [-sin(phi2), cos(phi2)]])
Rphi3 = Matrix([[cos(phi2 + phi3), sin(phi2 + phi3)],
                [-sin(phi2 + phi3), cos(phi2 + phi3)]])

# position of imus and pressure sensors

# left leg
r0 = Matrix([x + s0 * sin(phi0), y - s0 * cos(phi0)])
r1 = Matrix([x + l0 * sin(phi0) + s1 * sin(phi0 + phi1),
             y - l0 * cos(phi0) - s1 * cos(phi0 + phi1)])
r4 = Matrix([x + l0 * sin(phi0) + l1 * sin(phi0 + phi1),
             y - l0 * cos(phi0) - l1 * cos(phi0 + phi1)])
# right leg
r2 = Matrix([x + s2 * sin(phi2), y - s2 * cos(phi2)])
r3 = Matrix([x + l2 * sin(phi2) + s3 * sin(phi2 + phi3),
             y - l2 * cos(phi2) - s3 * cos(phi2 + phi3)])
r5 = Matrix([x + l2 * sin(phi2) + l3 * sin(phi2 + phi3),
             y - l2 * cos(phi2) - l3 * cos(phi2 + phi3)])

# first and second derivatives
r0dot = simplify(Rphi0 * diff(r0, t))
r0ddot = simplify(Rphi0 * (diff(r0, t, 2) + Matrix([0, g])))
r1dot = simplify(Rphi1 * diff(r1, t))
r1ddot = simplify(Rphi1 * (diff(r1, t, 2) + Matrix([0, g])))
r4dot = simplify(diff(r4, t))
r4ddot = simplify(diff(r4, t, 2))

r2dot = simplify(Rphi2 * diff(r2, t))
r2ddot = simplify(Rphi2 * (diff(r2, t, 2) + Matrix([0, g])))
r3dot = simplify(Rphi3 * diff(r3, t))
r3ddot = simplify(Rphi3 * (diff(r3, t, 2) + Matrix([0, g])))
r5dot = simplify(diff(r5, t))
r5ddot = simplify(diff(r5, t, 2))

# print first and second derivatives
init_printing()
rs = [r0dot, r0ddot, r1dot, r1ddot, r4dot, r4ddot, r2dot, r2ddot, r3dot, r3ddot,
      r5dot, r5ddot]
for r in rs:
    display(r)
