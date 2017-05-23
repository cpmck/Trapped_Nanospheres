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
	def __init__(self, trap_params):
		
		self.trap_params = trap_params
		
	def field(self, x):
		# assume trap is centred at x = 0
		x_atomic = x #*codata.value('atomic unit of length')
		trap_params_atomic = self.trap_params*codata.value('atomic unit of electric potential')
		field_val = trap_params_atomic*np.power(x_atomic,2)
		return np.array(field_val)

	def dxfield(self,x):
		x_atomic = x #*codata.value('atomic unit of length')
		trap_params_atomic = self.trap_params*codata.value('atomic unit of electric potential')
		field_val = trap_params_atomic*x_atomic*2
		return np.array(field_val)

	def d2xfield(self,x):
		x_atomic = x #*codata.value('atomic unit of length')
		trap_params_atomic = self.trap_params*codata.value('atomic unit of electric potential')
		field_val = trap_params_atomic*2
		return np.array(field_val)






class Nanosphere(object):
	def __init__(self, sphere_charge, sphere_radius, sphere_mass):
		
		self.sphere_charge = sphere_charge
		self.sphere_radius = sphere_radius
		self.sphere_mass = sphere_mass
		
	def charge(self):
		
		charge_val = codata.value('atomic unit of charge')*self.sphere_charge
		return np.array(charge_val)
	
	def mass(self):
		
		mass_val = codata.value('atomic mass constant')*self.sphere_mass
		return np.array(mass_val)
	
	def field(self,x,x_sphere):
		
		x_atomic = x*codata.value('atomic unit of length')
		x_sphere_atomic = x_sphere*codata.value('atomic unit of length')
		vac_perm = codata.value('atomic unit of permittivity')
		field_val = self.charge()/(4.0*np.pi*vac_perm)/np.absolute(x_atomic-x_sphere_atomic)
		
		return np.array(field_val)
	
	def dxfield(self,x1,x2):
		
		x1_units = x1*codata.value('atomic unit of length')
		x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('atomic unit of permittivity')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.power(x1_units-x2_units,2)

		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)

	def d2xfield(self,x1,x2):
		
		x1_units = x1*codata.value('atomic unit of length')
		x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('atomic unit of permittivity')
		field_val = 2 * self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1_units-x2_units,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)

	def dydxfield(self,x1,x2):
		

		x1_units = x1*codata.value('atomic unit of length')
		x2_units = x2*codata.value('atomic unit of length')
		vac_perm = codata.value('atomic unit of permittivity')
		field_val = -self.charge()/(4.0*np.pi*vac_perm)/np.absolute(np.power(x1_units-x2_units,3))
		
		if x1 >= x2:
			return np.array(field_val)
		elif x2 > x1:
			return np.array(-field_val)









class Potential(object):
	def __init__(self, trap, sphere_1, sphere_2):
		
		self.trap = trap
		self.sphere_1 = sphere_1
		self.sphere_2 = sphere_2
		
	def potential(self,x):        
		
		potential_val_trap = self.sphere_1.charge()*self.trap.field(x[[0]]) + sphere_2.charge()*self.trap.field(x[[1]])
		potential_val_ions = self.sphere_1.charge()*self.sphere_2.field(x[[0]],x[[1]]) + self.sphere_2.charge()*self.sphere_1.field(x[[0]],x[[1]])
		potential_val = potential_val_trap + potential_val_ions
		
		return np.array(potential_val)
	
	def equilibrium(self):
		x0 = np.array([-1,1])
		res = minimize(self.potential, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
		
		return res.x

	def hessian(self, x):

		A = np.zeros([2,2])

		A[0,0] = self.sphere_1.charge()*self.trap.d2xfield(x[[0]]) + self.sphere_1.charge()*self.sphere_2.d2xfield(x[[0]],x[[1]]) + self.sphere_2.charge()*self.sphere_1.d2xfield(x[[0]],x[[1]])
		A[1,1] = self.sphere_1.charge()*self.trap.d2xfield(x[[1]]) + self.sphere_1.charge()*self.sphere_2.d2xfield(x[[1]],x[[0]]) + self.sphere_2.charge()*self.sphere_1.d2xfield(x[[1]],x[[0]])
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

	def omega(self, x, n=1):

		field_grad_units = self.field_grad * codata.value('atomic unit of electric field gradient')
		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		omega = ge*muB*field_grad_units*x[[n]]

		return omega
	
	def dxomega(self, x, n=1):

		field_grad_units = self.feild_grad * codata.value('atomic unit of electric field gradient')
		ge = codata.value('electron g factor')
		muB = codata.value('Bohr magneton')
		omega = ge*muB*field_grad_units

		return omega



sphere_1 = Nanosphere(2,1,1)
sphere_2 = Nanosphere(2,1,1)
trap_1 = Harmonic_Paul_trap(3)
potential_1 = Potential(trap_1,sphere_1,sphere_2)

potential_1.S_matrix()

bfield = B_field(1)


x = np.linspace(-10,10,1000)

y = bfield.omega(x)