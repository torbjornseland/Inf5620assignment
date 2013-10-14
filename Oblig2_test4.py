#!/usr/bin/env python
import time
from numpy import *


def solver(b, q, I, V, f, Lx, Ly, Nx, Ny, dt, T,user_action=None, version='scalar',plug='no',exact='no'):
		if version == 'vectorized':
			advance = advance_vectorized
		elif version == 'scalar':
			advance = advance_scalar

		x = linspace(0, Lx, Nx+1) # mesh points in x dir
		y = linspace(0, Ly, Ny+1) # mesh points in y dir
		dx = x[1] - x[0]
		dy = y[1] - y[0]
		print dx

		xv = x[:,newaxis] # for vectorized function evaluations
		yv = y[newaxis,:]
		
		from collections import Iterable
		TEST=isinstance(q(x,y), Iterable)

		if TEST==True:
			c=sqrt( abs( max(q(x,y))))
		else:
			c=sqrt(q(x,y))

		
		#Stability limit
		stability_limit = (1/float(c))*(1/sqrt(1/dx**2 + 1/dy**2))
		if plug=='ok':
			dt = dx/q(0,0)
		elif dt <= 0: # max time step?
			if dt <= 0:
				safety_factor = -dt # use negative dt as safety factor
				dt = safety_factor*stability_limit
				print "stab",stability_limit
			elif dt > stability_limit:
				print 'error: dt=%g exceeds the stability limit %g' % (dt, stability_limit)


		Nt = int(round(T/float(dt)))
		t = linspace(0, Nt*dt, Nt+1) # mesh points in time

		Cx2 = (dt/dx)**2; Cy2 = (dt/dy)**2 # help variables

		#Allow f and V to be None or 0
		if f is None or f == 0:
			f = (lambda x, y, t: 0) if version == 'scalar' else lambda x, y, t: zeros((x.shape[0], y.shape[1]))

		if V is None or V == 0:
			V = (lambda x, y: 0) if version == 'scalar' else lambda x, y: zeros((x.shape[0], y.shape[1]))

		if q is None or q==0:
			q=(lambda x, y, t: 0) if version == 'scalar' else lambda x, y, t: zeros((x.shape[0], y.shape[1]))

		order ='C'
		u = zeros((Nx+1,Ny+1), order=order) # solution array
		u_1 = zeros((Nx+1,Ny+1), order=order) # solution at t-dt
		u_2 = zeros((Nx+1,Ny+1), order=order) # solution at t-2*dt
		f_a = zeros((Nx+1,Ny+1), order=order) # for compiled loops
		
		#Exact solution
		max_E = 0
		if exact=='ok':
			A = 1
			mx = 3
			my = 3
			k_x = (mx*pi)/Lx
			k_y = (my*pi)/Ly	
			w = sqrt(q(0,0)*(k_x**2+k_y**2))
			u_e = zeros((Nx+1,Ny+1), order='C') # solution array
			max_E = None
	

		Ix = range(0, u.shape[0])
		Iy = range(0, u.shape[1])
		It = range(0, t.shape[0])

		import time; t0 = time.clock() # for measuring CPU time

		# Load initial condition into u_1
		if version == 'scalar':# or plug == 'ok':
			for i in Ix:
				for j in Iy:
					u_1[i,j] = I(x[i], y[j])
		else: # use vectorized version
			u_1[:,:] = I(xv, yv)

		if user_action is not None:
			user_action(u_1, x, xv, y, yv, t, 0)

		# Special formula for first time step
		n = 0
		# Can use advance function with adjusted parameters (note: u_2=0)
		if version == 'scalar':
			u = advance(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n,Cx2, Cy2, dt, V, step1=True)

		else: # use vectorized version
			f_a[:,:] = f(xv, yv, t[n]) # precompute, size as u
			u = advance(x,y,b,q,u, u_1, u_2, f_a, Cx2, Cy2, dt, V, step1=True)

		if user_action is not None:
			user_action(u, x, xv, y, yv, t, 1)

		u_2[:,:] = u_1; u_1[:,:] = u

		if(exact=='ok'):
			for n in It[1:-1]:
				u_e[:,:] = A*cos(k_x*xv)*cos(k_y*yv)*cos(w*t[n])
				if version == 'scalar':
					# use f(x,y,t) function
					u = advance(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt,V)

				else:
					f_a[:,:] = f(xv, yv, t[n]) # precompute, size as u
					u = advance(x,y,b,q,u, u_1, u_2, f_a, Cx2, Cy2, dt,V)
					
					
					if user_action is not None:
						if user_action(u, x, xv, y, yv, t, n+1):
							break
					

					u_2[:,:], u_1[:,:] = u_1, u
				if max_E==None or max_E < abs(u-u_e).max():
						max_E = abs(u-u_e).max()
		else:			
			for n in It[1:-1]:
				if version == 'scalar':
					# use f(x,y,t) function
					u = advance(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt,V)
				else:
					f_a[:,:] = f(xv, yv, t[n]) # precompute, size as u
					u = advance(x,y,b,q,u, u_1, u_2, f_a, Cx2, Cy2, dt,V)

					if user_action is not None:
						if user_action(u, x, xv, y, yv, t, n+1):
							break

					u_2[:,:], u_1[:,:] = u_1, u

		t1 = time.clock()
		# dt might be computed in this function so return the value
		return u, x, dt, t1 - t0,max_E

def advance_scalar(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt,V, step1=False):
		Ix = range(0, u.shape[0]); Iy = range(0, u.shape[1])
		dt2 = dt**2
		R1=(1+(b*dt/2.0)); R2=((b*dt/2.0) -1)
		if step1:
			A = (1-(R2/R1)); B = 0.0 ; C = 2.0
		else:
			A=1.0; B=1.0; C=0.0
		for i in Ix[1:-1]:
			for j in Iy[1:-1]:
				u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
						-0.5*( (q(x[i-1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i-1,j]) )) 

				u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
						-0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

				u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
												u_yqu_y + dt2*f(x[i],y[j], t[n]) ))

		# Boundary condition du/dn=0 newmann condition

		# South Bondary
		j = Iy[0]
		for i in range(1,Nx):	
			u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
					-0.5*( (q(x[i-1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i-1,j]) )) 

			u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j+1]-u_1[i,j]) )

			u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))


		# North Boundary 
		j = Iy[-1]
		for i in range(1,Nx):
			u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
					-0.5*( (q(x[i-1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i-1,j]) ))

			u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j-1]-u_1[i,j]))

			u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))		


		# West Boundary
		i = Ix[0]
		for j in range(1,Ny):
				u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i+1,j]-u_1[i,j]))

				u_yqu_y =Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
								-0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

				u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
										u_yqu_y + dt2*f(x[i],y[j], t[n]) ))


		# East Boundary
		i = Ix[-1]
		for j in range(0,Ny):
			u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i-1,j]-u_1[i,j]))

			u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
					-0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

			u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
					u_yqu_y + dt2*f(x[i],y[j], t[n]) ))



		# South-west boundary
		j = Iy[0]
		i = Ix[0]	
		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
					-0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i+1,j]) ))

		u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j+1]-u_1[i,j]) )

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))


		# Corner boundaries

		# North East Boundary
		i = Ix[-1]
		j = Iy[-1]
		u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i-1,j]-u_1[i,j]))

		u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j-1]-u_1[i,j]) ) \
						-0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))

		# North West Boundary
		j = Iy[-1]
		i = Ix[0]

		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
						-0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i+1,j]) ))

		u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j-1]-u_1[i,j]))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))

		# South East Boundary
		i = Ix[-1]
		j = Iy[0]
		u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i-1,j]-u_1[i,j]))

		u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
						-0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j+1]) ))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f(x[i],y[j], t[n]) ))

		return u

def advance_vectorized(x, y, b, q, u, u_1, u_2, f_a, Cx2, Cy2, dt,V, step1=False):
		Ix = range(0, u.shape[0]); Iy = range(0, u.shape[1])
		dt2 = dt**2
		R1=(1+(b*dt/2.0)); R2=((b*dt/2.0) -1)
		if step1:
			A = (1-(R2/R1)); B = 0.0 ; C = 2.0

		else:
			A=1.0; B=1.0; C=0.0

		u_xqu_x  = Cx2*( 0.5*( (q(x[2:],y[1:-1])+q(x[1:-1],y[1:-1]) )*(u_1[2:,1:-1]-u_1[1:-1,1:-1])) \
				-0.5*( (q(x[:-2],y[1:-1])+q(x[1:-1],y[1:-1]) )*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ))

		u_yqu_y  = Cy2*( 0.5*( (q(x[1:-1],y[2:])+ q(x[1:-1],y[1:-1]) ) * (u_1[1:-1,2:]-u_1[1:-1,1:-1]) ) \
				-0.5*( (q(x[1:-1],y[:-2])+ q(x[1:-1],y[1:-1]) ) * (u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ))

		u[1:-1,1:-1] = (1/A)*( (2/R1)*u_1[1:-1,1:-1] +B*(R2/R1)*u_2[1:-1,1:-1] - C*(R2/R1)*dt*V(x[1:-1],y[1:-1])  + (1/R1)*(u_xqu_x +\
				u_yqu_y + dt2*f_a[1:-1,1:-1]) )

		# Boundary condition du/dn=0 neumann condition

		# South Bondary
		j = Iy[0]

		u_xqu_x = Cx2*( 0.5*( (q(x[2:],y[j])+q(x[1:-1],y[j]) )*(u_1[2:,j]-u_1[1:-1,j])) \
			-0.5*( (q(x[:-2],y[j])+q(x[1:-1],y[j]) )*(u_1[1:-1,j]-u_1[:-2,j]) )) 

		u_yqu_y = Cy2*( 2*q(x[1:-1],y[j])*(u_1[1:-1,j+1]-u_1[1:-1,j]) )

		u[1:-1,j] = (1/A) *( (2/R1)*u_1[1:-1,j] +B*(R2/R1)*u_2[1:-1,j]  - C*(R2/R1)*dt*V(x[1:-1], y[j]) +(1/R1)*(u_xqu_x +\
			u_yqu_y + dt2*f_a[1:-1,j] ))		

		# North Boundary 
		j = Iy[-1]

		u_xqu_x = Cx2*( 0.5*( (q(x[2:],y[j])+q(x[1:-1],y[j]) )*(u_1[2:,j]-u_1[1:-1,j])) \
				-0.5*( (q(x[:-2],y[j])+q(x[1:-1],y[j]) )*(u_1[1:-1,j]-u_1[:-2,j]) ))

		u_yqu_y = Cy2*( 2*q(x[1:-1],y[j])*(u_1[1:-1,j-1]-u_1[1:-1,j]))

		u[1:-1,j] = (1/A) *( (2/R1)*u_1[1:-1,j] +B*(R2/R1)*u_2[1:-1,j]  - C*(R2/R1)*dt*V(x[1:-1], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f_a[1:-1,j] ))		

		# West Boundary
		i = Ix[0]

		u_xqu_x = Cx2*( 2*q(x[i],y[1:-1])*(u_1[i+1,1:-1]-u_1[i,1:-1]))

		u_yqu_y =Cy2*( 0.5*( (q(x[i],y[2:])+ q(x[i],y[1:-1]) ) * (u_1[i,2:]-u_1[i,1:-1]) ) \
				-0.5*( (q(x[i],y[:-2])+ q(x[i],y[1:-1]) ) * (u_1[i,1:-1]-u_1[i,:-2]) ))

		u[i,1:-1] = (1/A) *( (2/R1)*u_1[i,1:-1] +B*(R2/R1)*u_2[i,1:-1]  - C*(R2/R1)*dt*V(x[i], y[1:-1]) +(1/R1)*(u_xqu_x +\
						u_yqu_y + dt2*f_a[i,1:-1] ))

		# East Boundary
		i = Ix[-1]

		u_xqu_x = Cx2*( 2*q(x[i],y[1:-1])*(u_1[i-1,1:-1]-u_1[i,1:-1]))

		u_yqu_y = Cy2*( 0.5*( (q(x[i],y[2:])+ q(x[i],y[1:-1]) ) * (u_1[i,2:]-u_1[i,1:-1]) ) \
				-0.5*( (q(x[i],y[:-2])+ q(x[i],y[1:-1]) ) * (u_1[i,1:-1]-u_1[i,:-2]) ))

		u[i,1:-1] = (1/A) *( (2/R1)*u_1[i,1:-1] +B*(R2/R1)*u_2[i,1:-1]  - C*(R2/R1)*dt*V(x[i], y[1:-1]) +(1/R1)*(u_xqu_x +\
						u_yqu_y + dt2*f_a[i,1:-1] ))


		# Corner Boundary
		"""
		j = 0
		u[:,j] = 0
		j = u.shape[1]-1
		u[:,j] = 0
		i = 0
		u[i,:] = 0
		i = u.shape[0]-1
		u[i,:] = 0
		"""
		# South-west boundary
		j = Iy[0]
		i = Ix[0]	
		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i+1,j]) ))

		u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j+1]-u_1[i,j]) )

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				u_yqu_y + dt2*f_a[i,j] ))

		# North East Boundary
		i = Ix[-1]
		j = Iy[-1]
		u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i-1,j]-u_1[i,j]))

		u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j-1]-u_1[i,j]) ) \
				-0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				u_yqu_y + dt2*f_a[i,j] ))

		# South East Boundary
		i = Ix[-1]
		j = Iy[0]
		u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i-1,j]-u_1[i,j]))

		u_yqu_y = Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
						-0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j+1]) ))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
								u_yqu_y + dt2*f_a[i,j] ))

		# North West Boundary
		j = Iy[-1]
		i = Ix[0]

		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i+1,j]) ))

		u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j-1]-u_1[i,j]))

		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				u_yqu_y + dt2*f_a[i,j] ))

		return u


def plot_u(u, x, xv, y, yv, t, n,plot_method=2,save_plot=True):
		if t[n] == 0:
			time.sleep(2)

		if plot_method == 1:
			mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],	caxis=[-1,1])
		elif plot_method == 2:
			surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 1],colorbar=True, colormap=hot(), caxis=[-1,1],shading='flat')

		#if plot_method > 0:
		#	time.sleep(0.001) # pause between frames

		if save_plot:
			filename = 'tmp_%04d.png' % n
			#savefig(filename) # time consuming!



from scitools.std import *
def run_Gaussian(plot_method=2, version='vectorized', save_plot=True):
		"""
		Initial Gaussian bell in the middle of the domain.
		plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
		"""

		# version='vectorized'
		# Clean up plot files
		for name in glob('tmp_*.png'):
				os.remove(name)

		Lx = 50
		Ly = 50
		b=0.1
		dt=-1
		c=1

		def h(x,y):
			B0=0.5;Ba=1.0;Bmx=10.0;Bmy=10.0;r=1;Bs=1.0
			return B0+Ba*exp(-0.5*(x-Bmx/Bs)**2 - 0.5*(y-Bmy/r*Bs)**2)
		def q(x, y):
			return 9.81*h(x,y)	
		def I(x, y):
			"""Gaussian peak."""
			Bmx=5.0;Bmy=5.0;b=1;Bs=1.0
			return exp(-0.5*(x-Bmx/Bs)**2 - 0.5*(y-Bmy/b*Bs)**2)
		def V(x,y):
				return 0

		#plot_u(b, q, u, x, xv, y, yv, t, n,plot_method)

		Nx = 50; Ny = 50; T = 50
		u, x, dt, cpu = solver(b, q, I, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=plot_u, version=version)
		return cpu,dt


import nose.tools as nt

def test_constant():
		constant=4.0
		def exact_solution(x, y, t):
				return constant
		def I(x, y):
				return exact_solution(x, y, 0)

		def V(x, y):
				return 0.0

		def f(x, y, t):
				return 0.0

		def q(x,y):
				return 5.0

		Lx = 3; Ly = 3
		Nx = 4; Ny = 4
		dt = -1 # use longest possible steps
		T = 18; b = 0.1

		def assert_no_error(b, q, u, x, xv, y, yv, t, n):
			u_e = exact_solution(xv, yv, t[n])
			diff = abs(u - u_e).max()

			nt.assert_almost_equal(diff, 0, places=11)

		for version in 'scalar', 'vectorized':
			print 'testing', version
			dt, cpu = solver(b, q, I, V, f, Lx, Ly, Nx, Ny, dt, T,user_action=assert_no_error,version=version)


def test_plug():
		#Check that an initial plug is correct back after one period.
		Lx = 1
		I = lambda x,y: 0 if abs(x-Lx/2.0) > 0.1 else 1

		Ly = 1
		Nx = 200
		Ny = 200
		b = 0
		dt = -1
		c_0 = 1.0
		T = 4

		def V(x,y):
			return 0	
		def c(x,y):
			return c_0

		u_s, x, dt, cpu = solver(b, c, I, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None, version='scalar',plug='ok')

		u_v, x, dt, cpu = solver(b, c, I, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None, version='vectorized',plug='ok')

		diff = abs(u_s - u_v).max()
		print "diff",diff
		nt.assert_almost_equal(diff, 0, places=13)
		u_0 = array([I(x_,0) for x_ in x])
		diff = abs(u_s - u_0).max()
		print "diff2",diff
		nt.assert_almost_equal(diff, 0, places=13)


def pulse2(animate=True,version='vectorized',T=2,loc='center',sigma = 0.5,plot_method=2,save_plot=True):

		for name in glob('tmp_*.png'):
			os.remove(name)

		Lx = 10
		Ly = 10
		Nx = 100
		Ny = 100
		b = 0
		dt = -1
		c_0 = 1.0
		if loc == 'center':
			xc = Lx/2
		elif loc == 'left':
			xc = 0

		def I(x, y): 
			return x #0 if abs(x-xc) > sigma else 1

		def V(x,y):
			return 0	
		def c(x,y):
			return c_0


		#plot_u(b, q, u, x, xv, y, yv, t, n,plot_method)

		u,x, dt, cpu = solver(b, c, I, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=plot_u, version=version,plug='ok')
		return dt,cpu

def exact_solution_undamped():
		
		Lx = 50
		Ly = 50
		Nx = 100
		Ny = 100

		dt = -1
		T = 5


		A = 1
		mx = 3
		my = 3
		w = 0.8
		k_x = (mx*pi)/Lx
		k_y = (my*pi)/Ly
		
		b = 0
		c_0 = 0.1
		print "c_0",c_0
		def I(x,y):
			return A*cos(k_x*x)*cos(k_y*y)*cos(0)
		
		def V(x,y):
			return -w*A*cos(k_x*x)*cos(k_y*y)*sin(0)

		def c(x,y):
			return c_0
		
		H_values = [0.2,0.1,0.05,0.01]#,2000]

		for h in H_values:
			print "----------------"
			Nxy = int(Lx/h)
			dt = h/sqrt(2*c_0)
			u,x, dt, cpu,max_E = solver(b, c, I, V, None, Lx, Ly, Nxy, Nxy, dt, T,user_action=None, version='scalar',exact='ok')
			print "C", max_E/float(h)
			print "dt", dt
			print "max",max_E
		

		return dt



if __name__ == '__main__':
	#test_constant()
	#cpu, dt=run_Gaussian(plot_method=2, version='vectorized', save_plot=True)
	#print dt
	#cpu=run_Gaussian(plot_method=2, version='scalar', save_plot=True)
	#run_efficiency_tests(nrefinements=4)
	#dt, cpu = pulse2()
	#pulse2()
	dt = exact_solution_undamped()
