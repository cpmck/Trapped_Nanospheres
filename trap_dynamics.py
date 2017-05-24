
import scipy
from scipy.constants import codata
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import minimize
from qutip import *
import time
from mpl_toolkits.mplot3d import Axes3D

from paul_trap import *

def make_sytem(m1,m2,r1,r2,q1,q2,B,V,r):

	sphere_1 = Nanosphere(q1, r1, m1)
	sphere_2 = Nanosphere(q2, r2, m2)
	trap_1 = Harmonic_Paul_trap(V, r)
	B_field_1 = B_field(B)

	
	system_1 = System(trap_1,sphere_1,sphere_2,B_field_1)
	

	positions = system_1.equilibrium()
	mass_1 = system_1.sphere_1.mass()
	mass_2 = system_1.sphere_2.mass()
	charge_1 = system_1.sphere_1.charge()
	charge_2 = system_1.sphere_2.charge()
	coupling = system_1.J().item((0.1))
	omegaT = system_1.omegaT()*1e-3/(2*np.pi)
	dxomega = system_1.B_field.dxomega()*1e-6*positions/(2*np.pi)
	V = system_1.trap1.V
	r = system_1.trap1.r


	print('    Variable                  ','ion_1                ', 'ion2')
	print('Equilibrium positions (m)   ', positions[0], '        ', positions[1] )
	print('mass                  (kg)  ', mass_1, '         ', mass_2 )
	print('charge                (C)   ', charge_1, '            ', charge_2 )
	print('nu/2pi                (kHz) ', omegaT[0],'            ', omegaT[1] )
	print('DeltaOmega/2pi        (MHz) ', dxomega )
	print('coupling J12          (Hz)  ', coupling )
	print('Trap Potential        (V)   ', V )
	print('Trap size (r_0)       (m)   ', r )




	return system_1

system_instance = make_sytem( 171*1.67e-27, 171*1.67e-27, 1e-8, 1e-8, 1.0, 1.0, 29.38, -0.2 ,0.001 )





si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()
a = destroy(2)

hbar = codata.value("Planck constant")

sx_list = []
sy_list = []
sz_list = []
a_list = []
adag_list = []

solver = "mc"
N = 2
omega = 1.0 * 2 * np.pi * np.ones(N)
J = system_instance.J().item((0,1)) * np.ones(N)
nu = np.array([1,1])

print("omegaT")
#nu = system_instance.omegaT()

print(J)
gamma = 0.1*np.ones(N)

# intial state, first spin in state |1>, the rest in state |0>
psi_list = []
psi_list.append(basis(2,1))
for n in range(N-1):
	psi_list.append(basis(2,0))
psi0 = tensor(psi_list)

tlist = np.linspace(0, 10, 200)





for n in range(2):
		op_list = []

		for m in range(2):
			op_list.append(si)

		op_list[n] = sx
		sx_list.append(tensor(op_list))

		op_list[n] = sy
		sy_list.append(tensor(op_list))

		op_list[n] = sz
		sz_list.append(tensor(op_list))

		op_list[n] = a
		a_list.append(tensor(op_list))

		op_list[n] = a.dag()
		adag_list.append(tensor(op_list))




#make the transformation matrix U

P = np.array([ 1* system_instance.sphere_1.mass() * system_instance.omegaT()[0] * system_instance.delta_z().item((0,0)) / hbar, 1j* system_instance.sphere_2.mass() * system_instance.omegaT()[1] * system_instance.delta_z().item((1,0)) /hbar])


omegaT_vec = np.transpose( [np.array( system_instance.omegaT()) ])
dxomega_1 = system_instance.B_field.dxomega()
dxomega_vec = np.array([[dxomega_1],[dxomega_1]])
deltaZ = system_instance.delta_z()
dxomega_mat = np.concatenate( (dxomega_vec, dxomega_vec), axis=1)
deltaZ_mat = np.concatenate( (np.transpose(deltaZ), np.transpose(deltaZ)), axis=0)
omegaT_mat = np.concatenate( (np.transpose(omegaT_vec), np.transpose(omegaT_vec)), axis=0)
S_matrix = system_instance.S_matrix()


epsilon = S_matrix*deltaZ_mat*dxomega_mat/omegaT_mat

exponent = 0
for l in range(2):
	for n in range(2):

		exponent += -1j* system_instance.delta_z().item((l,0))*epsilon.item((n,l))*sz_list[n]*np.asmatrix(P).item((0,l))  * ( adag_list[l] - a_list[l] )

U = exponent.expm()

print(U)

#make the hamiltonian H and H\tilde
H = 0

for n in range(N):
	H += -0.5*omega[n]*sx_list[n]

for n in range(N-1):
	H += -0.5*J[n]*sz_list[n]*sz_list[n+1]

H += nu[0] * a_list[0].dag()*a_list[0] + nu[1] * a_list[1].dag()*a_list[1]


H = U *H *U.dag()






c_op_list = []

for n in range(N):
	if gamma[n] > 0.0:
		c_op_list.append(np.sqrt(gamma[n])*sz_list[n])

if solver =="me":
	result = mesolve(H, psi0, tlist, c_op_list, sz_list)

elif solver == "mc":
	ntraj = 350
	result = mcsolve(H, psi0, tlist, c_op_list, sz_list, ntraj)


sz_expt = result.expect





#sz_expt = integrate(U, N, omega, J, nu, psi0, tlist, gamma, solver)

fig, ax = plt.subplots(figsize=(10,6))

for n in range(N):
    ax.plot(tlist, np.real(sz_expt[n]), label=r'$\langle\sigma_z^{(%d)}\rangle$'%n)

ax.legend(loc=0)
ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'\langle\sigma_z\rangle')
ax.set_title(r'Dynamics of a Heisenberg spin chain')

plt.show()
