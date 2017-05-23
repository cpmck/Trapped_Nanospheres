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


class Harmonic_Paul_trap(object):
	def __init__(self, V, r ):
		

		self.V = V # volts
		self.r = r # metres

	def r(self):
		return np.array(self.r)

	def V(self):
		return np.array(self.V)
		
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

		
	def charge(self):
		
		charge_val = codata.value('elementary charge')*self.sphere_charge
		return np.array(charge_val)
	
	def mass(self):
		
		#mass_val = codata.value('atomic mass constant')*self.sphere_mass
		return np.array(self.sphere_mass)


	def field(self,x,x_sphere):
		
		#x_atomic = x*codata.value('atomic unit of length')
		#x_sphere_atomic = x_sphere*codata.value('atomic unit of length')
		vac_perm = codata.value('electric constant')
		field_val = self.charge()/(4.0*np.pi*vac_perm)/np.absolute(x-x_sphere)
		
		return np.array(field_val)
	
	def dxfield(self,x1,x2):
		
		#x1_units = x1*codata.value('atomic unit of length')
		#x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('electric constant')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.power(x1-x2,2)

		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)

	def d2xfield(self,x1,x2):
		
		#x1_units = x1*codata.value('atomic unit of length')
		#x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('electric constant')
		field_val = 2 * self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1-x2,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)

	def dydxfield(self,x1,x2):
		

		#x1_units = x1*codata.value('atomic unit of length')
		#x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('electric constant')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1-x2,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)



class Potential(object):
	def __init__(self, trap1, sphere_1, sphere_2):
		
		self.trap1 = trap1
		self.sphere_1 = sphere_1
		self.sphere_2 = sphere_2


	def omegaT(self):

		#wT1 = 2 * self.sphere_1.sphere_charge * self.trap1.V / ( self.sphere_1.sphere_mass * self.trap1.r * self.trap1.r )
		#wT2 = 2 * self.sphere_2.sphere_charge * self.trap1.V / ( self.sphere_2.sphere_mass * self.trap1.r * self.trap1.r )

		wT1 = 2 * self.sphere_1.charge() * self.trap1.V / ( self.sphere_1.mass() * self.trap1.r * self.trap1.r )
		wT2 = 2 * self.sphere_2.charge() * self.trap1.V / ( self.sphere_2.mass() * self.trap1.r * self.trap1.r )

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
		res = minimize(self.potential, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
		
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

		A = self.hessian( self.equilibrium() )
		eig_vals = np.linalg.eig(A)[0]
		S = np.linalg.eig(A)[1] 

		return S



class B_field(object):

	def __init__(self,field_grad):

		self.field_grad = field_grad
		#field gradient in T/m

	def omega(self, x, n=1):

		#field_grad_units = self.field_grad * codata.value('atomic unit of electric field gradient')
		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		omega = ge*muB*self.field_grad*x[[n]]

		return omega
	
	def dxomega(self, x, n=1):

		hbar = codata.value("Planck constant")
		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		
		omega = ge*muB/hbar*self.field_grad

		return omega



sphere_1 = Nanosphere(1.0, 1e-8, 281*1.67e-27)
sphere_2 = Nanosphere(1.0, 1e-8, 171*1.67e-27)
trap_1 = Harmonic_Paul_trap(-1000.0, 0.1)
potential_1 = Potential(trap_1,sphere_1,sphere_2)
B_field = B_field(10)


omegaT_vec = np.transpose( [np.array( potential_1.omegaT()) ])
dxomega = B_field.dxomega(1)
dxomega_vec = np.array([[dxomega],[dxomega]])
deltaZ = potential_1.delta_z()

dxomega_mat = np.concatenate( (dxomega_vec, dxomega_vec), axis=1)
deltaZ_mat = np.concatenate( (np.transpose(deltaZ), np.transpose(deltaZ)), axis=0)
omegaT_mat = np.concatenate( (np.transpose(omegaT_vec), np.transpose(omegaT_vec)), axis=0)
S_matrix = potential_1.S_matrix()


print("Smatrix")
print(S_matrix)

print("deltaZ")
print(  deltaZ )

print("deltaZ Matrix")
print(  deltaZ_mat )

print("dxomega_vec")
print(dxomega_vec)

print("dxomega_Matrix")
print(dxomega_mat)

print("omegaT_vec")
print(omegaT_vec )

print("omegaT_mat")
print(omegaT_mat )

epsilon = S_matrix*deltaZ_mat*dxomega_mat/omegaT_mat

couplings = np.asmatrix( np.transpose(epsilon) ) * np.asmatrix( epsilon*np.transpose(omegaT_mat))

print("epsilon")
print(epsilon)


print("couplings")
print(couplings)

x = np.linspace(1,100,100)