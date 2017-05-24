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



class Harmonic_Paul_trap(object):
	def __init__(self, V, r ):
		

		self.V = V # volts
		self.r = r # metres

	def field(self, x):
		# assume trap is centred at x = 0
		#*codata.value('atomic unit of length')
		#*codata.value('atomic unit of electric potential')

		field_val = -2*np.power(x,2)*self.V/np.power(self.r,2)
		return np.array(field_val)

	def dxfield(self,x):
		#x_atomic = x #*codata.value('atomic unit of length')
		#trap_params_atomic = self.trap_params*codata.value('atomic unit of electric potential')
		field_val = -x*4*self.V/np.power(self.r,2)
		return np.array(field_val)

	def d2xfield(self,x):
		#x_atomic = x #*codata.value('atomic unit of length')
		#trap_params_atomic = self.trap_params*codata.value('atomic unit of electric potential')
		field_val = -4*self.V/np.power(self.r,2)
		return np.array(field_val)





class Nanosphere(object):
	def __init__(self, sphere_charge, sphere_radius, sphere_mass):
		
		self.sphere_charge = sphere_charge
		self.sphere_radius = sphere_radius
		self.sphere_mass = sphere_mass



	def radius(self):
		#radius in m
		return np.array(self.sphere_radius)



	def charge(self):
		#charge in units of electron charge
		charge_val = codata.value('elementary charge')*self.sphere_charge
		return np.array(charge_val)
	


	def mass(self):
		#Mass in kG
		return np.array(self.sphere_mass)



	def field(self,x,x_sphere):
		#use SI unit of vacuum permitivity
		#x input in metres
		vac_perm = codata.value('electric constant')
		field_val = self.charge()/(4.0*np.pi*vac_perm)/np.absolute(x-x_sphere)
		
		return np.array(field_val)
	


	def dxfield(self,x1,x2):
		#use SI unit of vacuum permitivity
		#x input in metres
		vac_perm = codata.value('electric constant')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.power(x1-x2,2)

		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)



	def d2xfield(self,x1,x2):
		#use SI unit of vacuum permitivity
		#x input in metres
		vac_perm = codata.value('electric constant')
		field_val = 2 * self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1-x2,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)

	def dydxfield(self,x1,x2):
		

		#use SI unit of vacuum permitivity
		#x input in metres
		vac_perm = codata.value('electric constant')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1-x2,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)




class B_field(object):

	#this code is specific to a linar field gradient

	def __init__(self,field_grad):

		self.field_grad = field_grad
		#field gradient in T/m

	def omega(self, x, n=1):

		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		omega = ge*muB*self.field_grad*x[[n]]
		return omega
	
	def dxomega(self):


		hbar = codata.value("Planck constant")
		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		omega = ge*muB*self.field_grad/hbar
		return omega



class System(object):
	def __init__(self, trap1, sphere_1, sphere_2, B_field):
		
		self.trap1 = trap1
		self.sphere_1 = sphere_1
		self.sphere_2 = sphere_2
		self.B_field = B_field


	def omegaT(self):

		#calculate the energy frequncy spacing of harmonic oscillator
		wT1 = 2 * self.sphere_1.charge() * np.array(self.trap1.V) / ( self.sphere_1.mass() * np.array( self.trap1.r ) * np.array( self.trap1.r ))
		wT2 = 2 * self.sphere_2.charge() * np.array(self.trap1.V) / ( self.sphere_2.mass() * np.array( self.trap1.r ) * np.array( self.trap1.r ))
		return np.sqrt( np.absolute( np.array([wT1,wT2]) ) )


	def delta_z(self):
		hbar = codata.value('Planck constant')
		dZ1 = np.sqrt(  np.absolute(  hbar/(  2* self.sphere_1.sphere_mass* self.omegaT()[[0]]  )   )   )
		dZ2 = np.sqrt(  np.absolute(  hbar/(  2* self.sphere_2.sphere_mass* self.omegaT()[[1]]  )   )   )
		return np.array([dZ1,dZ2])
		

	def potential(self,x):        
		
		potential_val_trap = self.sphere_1.charge()*self.trap1.field(x[[0]]) + self.sphere_2.charge()*self.trap1.field(x[[1]])
		potential_val_ions = self.sphere_1.charge()*self.sphere_2.field(x[[0]],x[[1]]) + self.sphere_2.charge()*self.sphere_1.field(x[[0]],x[[1]])
		potential_val = potential_val_trap + potential_val_ions
		return np.array(potential_val)
	

	def equilibrium(self):

		x0 = np.array([-1,1])

		res = minimize(self.potential, x0, method='nelder-mead',options={'xtol': 1e-10, 'disp': True})
		return res.x


	def hessian(self, x):

		A = np.zeros([2,2])
		A[0,0] = self.sphere_1.charge()*self.trap1.d2xfield(x[[0]]) + self.sphere_1.charge()*self.sphere_2.d2xfield(x[[0]],x[[1]]) + self.sphere_2.charge()*self.sphere_1.d2xfield(x[[0]],x[[1]])
		A[1,1] = self.sphere_1.charge()*self.trap1.d2xfield(x[[1]]) + self.sphere_1.charge()*self.sphere_2.d2xfield(x[[1]],x[[0]]) + self.sphere_2.charge()*self.sphere_1.d2xfield(x[[1]],x[[0]])
		A[0,1] = self.sphere_1.charge()*self.sphere_2.dydxfield(x[[0]],x[[1]]) + self.sphere_2.charge()*self.sphere_1.dydxfield(x[[0]],x[[1]])
		A[1,0] = self.sphere_1.charge()*self.sphere_2.dydxfield(x[[1]],x[[0]]) + self.sphere_2.charge()*self.sphere_1.dydxfield(x[[1]],x[[0]])
		#change outputs to matrices??
		return A


	def S_matrix(self):

		#calculate 
		A = self.hessian( self.equilibrium() )
		eig_vals = np.linalg.eig(A)[0]
		S = np.linalg.eig(A)[1] 
		return S


	def J(self):

		#calculate couplings 
		omegaT_vec = np.transpose( [np.array( self.omegaT()) ])
		dxomega_1 = self.B_field.dxomega()
		dxomega_vec = np.array([[dxomega_1],[dxomega_1]])
		deltaZ = self.delta_z()
		dxomega_mat = np.concatenate( (dxomega_vec, dxomega_vec), axis=1)
		deltaZ_mat = np.concatenate( (np.transpose(deltaZ), np.transpose(deltaZ)), axis=0)
		omegaT_mat = np.concatenate( (np.transpose(omegaT_vec), np.transpose(omegaT_vec)), axis=0)
		S_matrix = self.S_matrix()
		epsilon = S_matrix*deltaZ_mat*dxomega_mat/omegaT_mat
		couplings = np.asmatrix( np.transpose(epsilon) ) * np.asmatrix( epsilon*np.transpose(omegaT_mat))
		return couplings


# function to make a system
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


system_1 = make_sytem( 171*1.67e-27, 171*1.67e-27, 1e-8, 1e-8, 1.0, 1.0, 29.38, -0.2 ,0.001 )


#plotting over single variables
#range_x = np.linspace(0,10,100)
#y = np.zeros(100)
#r_calc_1 = np.zeros(100)
#r_calc_2 = np.zeros(100)


#counter = 0
#for value in range_x:

	#system_1 = make_sytem( 171*1.6e-27, 171*1.67e-27, 1e-8, 1e-8, 1.0, 1.0, value, -0.5 ,0.001 )
	#y[counter] = system_1.J().item((0,1))
	#r_calc_1[counter] = system_1.equilibrium()[[0]]
	#r_calc_2[counter] = system_1.equilibrium()[[1]]

	#counter = counter +1
#print(y)

#plt.plot(range_x,y)
#plt.plot(range_x, r_calc_1)
#plt.plot(range_x, r_calc_2)
plt.show()
