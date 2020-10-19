import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
cimport cython

# kinetics
@cython.cdivision(True)
cdef double efun(double z):
	if z < -5e2:
		return -5e2
	else:
		return z

@cython.cdivision(True)
cdef double gate_inf(double x, double a, double b):
		return 1/(1+exp(-a*x+b))

@cython.cdivision(True)
cdef double tau_gate2(double x,double c,double d,double e,double f,double g,double h):
		cdef double y = x - c
		return d/(exp(-(e*y+f*y**2)) + exp(g*y+h*y**2))

@cython.cdivision(True)
cdef double tau_gate3(double x,double c,double d,double e,double f,double g,double h,double k,double l):
		cdef double y = x - c
		return d/(exp(-(e*y+f*y**2+k*y**3)) + exp(g*y+h*y**2+l*y**3))

def seed(n):
	np.random.seed(n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatexpe2(np.ndarray[double,ndim=2] V,np.ndarray[double,ndim=2] act_var,int i,double tstep, int len_act_var,double a_act,double b_act,double c_act,double d_act,double e_act,double f_act,double g_act,double h_act, double tadj):

	cdef double act_var_inf1

	for j in range(len_act_var):
		act_var_inf1 = gate_inf(V[j,i-1],a_act,b_act)
		act_var[j,i] = act_var_inf1+(act_var[j,i-1]-act_var_inf1)*exp(efun(-tstep*tadj/tau_gate2(V[j,i-1],c_act,d_act,e_act,f_act,g_act,h_act)))

# Forward Euler
def expeuler2(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=2] V,np.ndarray[double,ndim=2] act_var,np.ndarray[double,ndim=2] I_channel,double tstep,double p_act,double a_act,double b_act,double c_act,double d_act,double e_act,double f_act,double g_act,double h_act,double E_channel,double fact_inward):

	cdef double celsius = 37  # original temperature
	cdef double temp = 23  # reference temperature
	cdef double q10 = 2.3  # temperature sensitivity
	cdef double tadj = q10**((celsius - temp)/10)

	#cdef double g_L = 3.334e-2  # mS/cm2
	#cdef double E_L = -80  # not sure which one is used in Podlaski et al. 2017, mV

	cdef double gbar_channel = 1.0  # mS/cm2

	cdef int len_act_var = len(V[:,0])

	for j in range(len_act_var):
		act_var[j,0] = gate_inf(V[j,0],a_act,b_act)

	for i in range(1, t.shape[0]):
		updatexpe2(V,act_var,i,tstep,len_act_var,a_act,b_act,c_act,d_act,e_act,f_act,g_act,h_act,tadj)


	I_channel = fact_inward*tadj*gbar_channel*(act_var**p_act)*(V-E_channel)
	I_channel /= np.max(I_channel)
	# I_L = tadj*g_L*(V-E_L)
	# I_tot = (I_channel + I_L)/np.max(I_channel + I_L)

	return I_channel

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatexpe3(np.ndarray[double,ndim=2] V,np.ndarray[double,ndim=2] act_var,int i,double tstep, int len_act_var,double a_act,double b_act,double c_act,double d_act,double e_act,double f_act,double g_act,double h_act, double k_act,double l_act,double tadj):

	cdef double act_var_inf1

	for j in range(len_act_var):
		act_var_inf1 = gate_inf(V[j,i-1],a_act,b_act)
		act_var[j,i] = act_var_inf1+(act_var[j,i-1]-act_var_inf1)*exp(efun(-tstep*tadj/tau_gate3(V[j,i-1],c_act,d_act,e_act,f_act,g_act,h_act,k_act,l_act)))


def expeuler3(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=2] V,np.ndarray[double,ndim=2] act_var,np.ndarray[double,ndim=2] I_channel,double tstep,double p_act,double a_act,double b_act,double c_act,double d_act,double e_act,double f_act,double g_act,double h_act,double k_act,double l_act,double E_channel,double fact_inward):

	cdef double celsius = 37  # original temperature
	cdef double temp = 23  # reference temperature
	cdef double q10 = 2.3  # temperature sensitivity
	cdef double tadj = q10**((celsius - temp)/10)

	#cdef double g_L = 3.334e-2  # mS/cm2
	#cdef double E_L = -80  # not sure which one is used in Podlaski et al. 2017, mV

	cdef double gbar_channel = 1.0  # mS/cm2

	cdef int len_act_var = len(V[:,0])

	for j in range(len_act_var):
		act_var[j,0] = gate_inf(V[j,0],a_act,b_act)

	for i in range(1, t.shape[0]):
		updatexpe3(V,act_var,i,tstep,len_act_var,a_act,b_act,c_act,d_act,e_act,f_act,g_act,h_act,k_act,l_act,tadj)

	I_channel = fact_inward*tadj*gbar_channel*(act_var**p_act)*(V-E_channel)
	I_channel /= np.max(I_channel)
	# I_L = tadj*g_L*(V-E_L)
	# I_tot = (I_channel + I_L)/np.max(I_channel + I_L)

	return I_channel
