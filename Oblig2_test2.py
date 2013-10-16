#!/usr/bin/env python
"""
2D wave equation solved by finite differences::

dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,
user_action=None, version='scalar',
dt_safety_factor=1)

Solve the 2D wave equation u_tt = D_x(q(x,y)D_xu) + Dy(q(x,y)D_yu) + f(x,t) on (0,L) with
du/dn=0 on the boundary and initial condition du/dt=0.

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f, q are functions: I(x,y), V(x,y), f(x,y,t), q(x,y). V and f
can be specified as None or 0, resulting in V=0 and f=0.

user_action: function of (b, q, u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""
import time
from numpy import *

def solver(b, q, I, V, f, Lx, Ly, Nx, Ny, dt, T,user_action=None, version='scalar',plug='no',exact='no',makeplot=None):
	if version == 'vectorized':
        	advance = advance_vectorized
    	else: 
        	advance = advance_scalar

    	x = linspace(0, Lx, Nx+1) # mesh points in x dir
    	y = linspace(0, Ly, Ny+1) # mesh points in y dir
	if exact =='ok':
			dx=x[1]-x[0]
			dy=dx
	else:
		dx = x[1] - x[0]
		dy = y[1] - y[0]
	#print "dx, dy", dx,  dy
    	

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
	elif dt <= 0:
		safety_factor = -dt # use negative dt as safety factor
		dt = safety_factor*stability_limit
	elif dt > stability_limit:
		print 'error: dt=%g exceeds the stability limit %g' % (dt, stability_limit)

	
	
    	Nt = int(round(T/float(dt)))
    	t = linspace(0, Nt*dt, Nt+1) # mesh points in time
	
    	Cx2 = (dt/dx)**2; Cy2 = (dt/dy)**2 # help variables
	
    	# Allow f ,V  and q to be None or 0
    	if f is None or f == 0:
        	f = (lambda x, y, t: 0) if version == 'scalar' else lambda x, y, t: zeros((x.shape[0], y.shape[1]))
      
    	if V is None or V == 0:
        	V = (lambda x, y: 0) if version == 'scalar' else lambda x, y: zeros((x.shape[0], y.shape[1]))

	if q is None or q==0:
		q=(lambda x, y, t: 0) if version == 'scalar' else lambda x, y, t: zeros((x.shape[0], y.shape[1]))
	
    	
    	u   = zeros((Nx+1,Ny+1)) # solution array
    	u_1 = zeros((Nx+1,Ny+1)) # solution at t-dt
    	u_2 = zeros((Nx+1,Ny+1)) # solution at t-2*dt
    	u_start = zeros((Nx+1,Ny+1)) # solution at t-2*dt
    	f_a = zeros((Nx+1,Ny+1)) # for compiled loops
	q_a = zeros((Nx+1,Ny+1)) # for compiled loops
	V_a = zeros((Nx+1,Ny+1)) # for compiled loops

    	Ix = range(0, u.shape[0])
    	Iy = range(0, u.shape[1])
    	It = range(0, t.shape[0])
	
    	import time; t0 = time.clock() # for measuring CPU time
	
    	# Load initial condition into u_1
    	if version == 'scalar' or plug=='ok':
        	for i in Ix:
            		for j in Iy:
                		u_1[i,j] = I(x[i], y[j])
    	else: # use vectorized version
        	u_1[:,:] = I(xv, yv)
	
    	if user_action is not None:
        	user_action(b, q, u_1, x, xv, y, yv, t, 0)

	if makeplot is not None:
		makeplot(u_1, x, xv, y, yv, t, 0,plot_method=2,save_plot=None)
		u_start[:,:] = u_1

    	# Special formula for first time step
    	n = 0
    	# Can use advance function with adjusted parameters (note: u_2=0)
    	if version == 'scalar':
        	u = advance(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n,Cx2, Cy2, dt, V, step1=True)

    	else: # use vectorized version
        	f_a[:,:] = f(xv, yv, t[n]) # precompute, size as u
		q_a[:,:] = q(xv, yv)
		V_a[:,:] = V(xv, yv)
        	u = advance(x,y,b,q_a,u, u_1, u_2, f_a, Cx2, Cy2, dt, V_a, step1=True)

    	if user_action is not None:
        	user_action(b, q, u, x, xv, y, yv, t, 1)

	if makeplot is not None:
		makeplot(u, x, xv, y, yv, t, 1,plot_method=2,save_plot=None)

    	u_2[:,:] = u_1; u_1[:,:] = u

    	for n in It[1:-1]:
        	if version == 'scalar':
            		# use f(x,y,t) function
            		u = advance(Nx, Ny, b, q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt,V)
        	else:
            		f_a[:,:] = f(xv, yv, t[n]) # precompute, size as u
			q_a[:,:] = q(xv, yv)
			V_a[:,:] = V(xv, yv)
            		u = advance(x,y,b,q_a,u, u_1, u_2, f_a, Cx2, Cy2, dt,V_a)

        	if user_action is not None:
            		if user_action(b, q, u, x, xv, y, yv, t, n+1):
                		break

		if makeplot is not None:
			if makeplot(u, x, xv, y, yv, t, n,plot_method=2,save_plot=None):
				break

		if abs(u_start-u).max()==0:
			print "T", dt*n

        	u_2[:,:], u_1[:,:] = u_1, u

    	t1 = time.clock()
    	# dt might be computed in this function so return the value
    	return u, x, dt, t1 - t0

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
	dx=dt/sqrt(Cx2)
	dy=dt/sqrt(Cy2)
    	j = Iy[0]
    	for i in range(1,Nx):	
		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q(x[i-1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i-1,j]) )) 
	
		u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j+1]-u_1[i,j]) )
 
		u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				 u_yqu_y + dt2*f(x[i],y[j], t[n])   ))
	
		
	# North Boundary 
    	j = Iy[-1]
	
    	for i in range(1,Nx):
 
		u_xqu_x = Cx2*( 0.5*( (q(x[i+1],y[j])+q(x[i],y[j]) )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q(x[i-1],y[j])+q(x[i],y[j]) )*(u_1[i,j]-u_1[i-1,j]) ))

            	u_yqu_y = Cy2*( 2*q(x[i],y[j])*(u_1[i,j-1]-u_1[i,j]))

            	u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				 u_yqu_y + dt2*f(x[i],y[j], t[n])  ))		
		
 	
	# West Boundary
    	i = Ix[0]
    	for j in range(1,Ny):
		u_xqu_x = Cx2*( 2*q(x[i],y[j])*(u_1[i+1,j]-u_1[i,j]))

            	u_yqu_y =Cy2*( 0.5*( (q(x[i],y[j+1])+ q(x[i],y[j]) ) * (u_1[i,j+1]-u_1[i,j]) ) \
				 -0.5*( (q(x[i],y[j-1])+ q(x[i],y[j]) ) * (u_1[i,j]-u_1[i,j-1]) ))

            	u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V(x[i], y[j]) +(1/R1)*(u_xqu_x +\
				 u_yqu_y + dt2*f(x[i],y[j], t[n])  ))
		
	
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
	
def advance_vectorized(x, y, b, q_a, u, u_1, u_2, f_a, Cx2, Cy2, dt,V_a, step1=False):
	Ix = range(0, u.shape[0]); Iy = range(0, u.shape[1])
    	dt2 = dt**2
    	R1=(1+(b*dt/2.0)); R2=((b*dt/2.0) -1)
    	if step1:
        	A = (1-(R2/R1)); B = 0.0 ; C = 2.0

    	else:
        	A=1.0; B=1.0; C=0.0
	
	
	u_xqu_x  = Cx2*( 0.5*( (q_a[2:,1:-1]+q_a[1:-1,1:-1] )*(u_1[2:,1:-1]-u_1[1:-1,1:-1])) \
				-0.5*( (q_a[:-2,1:-1]+q_a[1:-1,1:-1] )*(u_1[1:-1,1:-1]-u_1[:-2,1:-1]) ))

    	u_yqu_y  = Cy2*( 0.5*( (q_a[1:-1,2:]+ q_a[1:-1,1:-1] ) * (u_1[1:-1,2:]-u_1[1:-1,1:-1]) ) \
				 -0.5*( (q_a[1:-1,:-2]+ q_a[1:-1,1:-1] ) * (u_1[1:-1,1:-1]-u_1[1:-1,:-2]) ))
	
	u[1:-1,1:-1] = (1/A)*( (2/R1)*u_1[1:-1,1:-1] +B*(R2/R1)*u_2[1:-1,1:-1] - C*(R2/R1)*dt*V_a[1:-1,1:-1]  + (1/R1)*(u_xqu_x +\
			 u_yqu_y + dt2*f_a[1:-1,1:-1]) )
	
	
	# Boundary condition du/dn=0 neumann condition
	
	# South Bondary
    	j = Iy[0]
    	
	u_xqu_x = Cx2*( 0.5*( (q_a[2:,j]+q_a[1:-1,j] )*(u_1[2:,j]-u_1[1:-1,j])) \
			-0.5*( (q_a[:-2,j]+q_a[1:-1,j] )*(u_1[1:-1,j]-u_1[:-2,j]) )) 
	
	u_yqu_y = Cy2*( 2*q_a[1:-1,j]*(u_1[1:-1,j+1]-u_1[1:-1,j]) )
 
	u[1:-1,j] = (1/A) *( (2/R1)*u_1[1:-1,j] +B*(R2/R1)*u_2[1:-1,j]  - C*(R2/R1)*dt*V_a[1:-1,j] +(1/R1)*(u_xqu_x +\
			u_yqu_y + dt2*f_a[1:-1,j] ))		

	# North Boundary 
    	j = Iy[-1]
 
	u_xqu_x = Cx2*( 0.5*( (q_a[2:,j]+q_a[1:-1,j] )*(u_1[2:,j]-u_1[1:-1,j])) \
			-0.5*( (q_a[:-2,j]+q_a[1:-1,j] )*(u_1[1:-1,j]-u_1[:-2,j]) ))

      	u_yqu_y = Cy2*( 2*q_a[1:-1,j]*(u_1[1:-1,j-1]-u_1[1:-1,j]))

     	u[1:-1,j] = (1/A) *( (2/R1)*u_1[1:-1,j] +B*(R2/R1)*u_2[1:-1,j]  - C*(R2/R1)*dt*V_a[1:-1, j] +(1/R1)*(u_xqu_x +\
			u_yqu_y + dt2*f_a[1:-1,j] ))		
		
	# West Boundary
    	i = Ix[0]
  
	u_xqu_x = Cx2*( 2*q_a[i,1:-1]*(u_1[i+1,1:-1]-u_1[i,1:-1]))

      	u_yqu_y =Cy2*( 0.5*( (q_a[i,2:]+ q_a[i,1:-1] ) * (u_1[i,2:]-u_1[i,1:-1]) ) \
			-0.5*( (q_a[i,:-2]+ q_a[i,1:-1] ) * (u_1[i,1:-1]-u_1[i,:-2]) ))

      	u[i,1:-1] = (1/A) *( (2/R1)*u_1[i,1:-1] +B*(R2/R1)*u_2[i,1:-1]  - C*(R2/R1)*dt*V_a[i, 1:-1] +(1/R1)*(u_xqu_x +\
			u_yqu_y + dt2*f_a[i,1:-1] ))
	
	# East Boundary
    	i = Ix[-1]
   
	u_xqu_x = Cx2*( 2*q_a[i,1:-1]*(u_1[i-1,1:-1]-u_1[i,1:-1]))

      	u_yqu_y = Cy2*( 0.5*( (q_a[i,2:]+ q_a[i,1:-1] ) * (u_1[i,2:]-u_1[i,1:-1]) ) \
			-0.5*( (q_a[i,:-2]+ q_a[i,1:-1] ) * (u_1[i,1:-1]-u_1[i,:-2]) ))

     	u[i,1:-1] = (1/A) *( (2/R1)*u_1[i,1:-1] +B*(R2/R1)*u_2[i,1:-1]  - C*(R2/R1)*dt*V_a[i, 1:-1] +(1/R1)*(u_xqu_x +\
			u_yqu_y + dt2*f_a[i,1:-1] ))
	
	
	# Corner Boundary

	# South-west boundary
	j = Iy[0]
    	i = Ix[0]	
	u_xqu_x = Cx2*( 0.5*( (q_a[i+1,j]+q_a[i,j] )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q_a[i+1,j]+q_a[i,j] )*(u_1[i,j]-u_1[i+1,j]) ))

	u_yqu_y = Cy2*( 2*q_a[i,j]*(u_1[i,j+1]-u_1[i,j]) )
 
	u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V_a[i, j] +(1/R1)*(u_xqu_x +\
		         u_yqu_y + dt2*f_a[i,j] ))

	# North East Boundary
	i = Ix[-1]
    	j = Iy[-1]
	u_xqu_x = Cx2*( 2*q_a[i,j]*(u_1[i-1,j]-u_1[i,j]))

        u_yqu_y = Cy2*( 0.5*( (q_a[i,j-1]+ q_a[i,j] ) * (u_1[i,j-1]-u_1[i,j]) ) \
				 -0.5*( (q_a[i,j-1]+ q_a[i,j] ) * (u_1[i,j]-u_1[i,j-1]) ))

       	u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V_a[i, j] +(1/R1)*(u_xqu_x +\
		u_yqu_y + dt2*f_a[i,j] ))

	# South East Boundary
	i = Ix[-1]
	j = Iy[0]
	u_xqu_x = Cx2*( 2*q_a[i,j]*(u_1[i-1,j]-u_1[i,j]))

        u_yqu_y = Cy2*( 0.5*( (q_a[i,j+1]+ q_a[i,j] ) * (u_1[i,j+1]-u_1[i,j]) ) \
				 -0.5*( (q_a[i,j+1]+ q_a[i,j] ) * (u_1[i,j]-u_1[i,j+1]) ))

        u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V_a[i, j] +(1/R1)*(u_xqu_x +\
		u_yqu_y + dt2*f_a[i,j] ))
	
	# North West Boundary
	j = Iy[-1]
	i = Ix[0]
 
	u_xqu_x = Cx2*( 0.5*( (q_a[i+1,j]+q_a[i,j] )*(u_1[i+1,j]-u_1[i,j])) \
				-0.5*( (q_a[i+1,j]+q_a[i,j] )*(u_1[i,j]-u_1[i+1,j]) ))

        u_yqu_y = Cy2*( 2*q_a[i,j]*(u_1[i,j-1]-u_1[i,j]))

	u[i,j] = (1/A) *( (2/R1)*u_1[i,j] +B*(R2/R1)*u_2[i,j]  - C*(R2/R1)*dt*V_a[i, j] +(1/R1)*(u_xqu_x +\
		u_yqu_y + dt2*f_a[i,j] ))
	
    	return u

from scitools.std import *
def run_Gaussian(plot_method=2, version='scalar', save_plot=None):
    	"""
	Initial Gaussian bell in the middle of the domain.
	plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
	"""
	
    	# Clean up plot files
    	for name in glob('tmp_*.png'):
        	os.remove(name)

    	Lx = 25
    	Ly = 25
	b=0.3
	dt=-1
	c=1
	def h(x,y):
		B0=1;Ba=6;Bmx=5;Bmy=5;r=1;Bs=1.0
		#return B0+Ba*cos(pi*(x-Bmx)/2*Bs)*cos(pi*(y-Bmy)/2*Bs)
        	return B0+Ba*exp(-(x-Bmx/Bs)**2 -(y-Bmy/r*Bs)**2)
	def q(x, y):
		return 9.81*h(x,y)	
		
    	def I(x, y):
        	"""Gaussian peak."""
		bmx=12;bmy=12;b=1;bs=1.0;B0=5
        	return B0+2*exp(-0.5*(x-bmx/bs)**2 - 0.5*(y-bmy/b*bs)**2)
	def V(x,y):
		return 0
		
    	def plot_u(b, q, u, x, xv, y, yv, t, n):
		
		a= len(x)-1
		b=len(y)-1
		Ix = range(0, u.shape[0])
    		Iy = range(0, u.shape[1])
		h = zeros((a+1,b+1))
		for i in Ix:
            		for j in Iy:
                		h[i,j] =-2+ q(x[i], y[j])/9.81
		
        	if t[n] == 0:
			time.sleep(1)
        	if plot_method == 1:
            		mesh(x, y, u, title='t=%g' % t[n], zlim=[2,7],
                 	caxis=[2,7])
        	elif plot_method == 2:
			#surfc(xv, yv,h, title='t=%g' % t[n], zlim=[2, 7],colorbar=True, colormap=hot(), caxis=[2,7],shading='flat')
			#hold('on
			surfc(xv, yv,u, title='t=%g' % t[n], zlim=[2, 7],colorbar=True, colormap=winter(), caxis=[2,7],shading='flat')
			
            		
        	if plot_method > 0:
            		time.sleep(0.001) # pause between frames
            	if save_plot:
                	filename = 'tmp_%04d.png' % n
                	#savefig(filename) # time consuming!

    	Nx = 50; Ny = 50; T = 15
    	dt, cpu = solver(b, q, I, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=plot_u, version=version)
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
 		print diff
        	nt.assert_almost_equal(diff, 0, places=11)

    	for version in 'scalar', 'vectorized':
        	print 'testing', version
        	u, x,dt, cpu = solver(b, q, I, V, f, Lx, Ly, Nx, Ny, dt, T,user_action=assert_no_error,version=version)

def test_qubic():
	Nx = 10; Ny = 20
    	Lx = 3;  Ly = 3
    	b  = 0.0; T = 8
    	dt = -1 # use longest possible steps
    	
    	def exact_solution(x, y, t):
        	return x**2*y**2*(3*Lx -2*x)*(3*Ly- 2*y)*(1+(1.0/2)*t)/36

    	def I(x, y):
        	return exact_solution(x, y, 0)

    	def V(x, y):
        	return 0.5*exact_solution(x, y, 0)

	def q(x,y):
		return 1.0

	def f(x, y, t):
        	return -0.25*Lx*Ly*t*x**2 - 0.25*Lx*Ly*t*y**2 - 0.5*Lx*Ly*x**2 - 0.5*Lx*Ly*y**2 + 0.5*Lx*t*x**2*y \
				+ 0.166666666666667*Lx*t*y**3 + 1.0*Lx*x**2*y + 0.333333333333333*Lx*y**3 + 0.166666666666667*Ly*t*x**3 \
				+ 0.5*Ly*t*x*y**2 + 0.333333333333333*Ly*x**3 + 1.0*Ly*x*y**2 - 0.333333333333333*t*x**3*y \
				- 0.333333333333333*t*x*y**3 - 0.666666666666667*x**3*y - 0.666666666666667*x*y**3 

    	def assert_no_error(b, q, u, x, xv, y, yv, t, n):
		
    		u_e = exact_solution(xv, yv, t[n])
		
        	diff = abs(u - u_e).max()
		
        	#nt.assert_almost_equal(diff, 0, places=0)

    	version = 'scalar'
       
        dt, cpu = solver(b, q, I, V, f, Lx, Ly, Nx, Ny, dt, T,user_action=assert_no_error,version=version)

def plotter(u, x, xv, y, yv, t, n,plot_method=2,save_plot=None):
		if t[n] == 0:
			time.sleep(1)

		if plot_method == 1:
			mesh(x, y, u, title='t=%g' % t[n], zlim=[2,7],	caxis=[2,7])
		elif plot_method == 2:
			surfc(xv, yv, u, title='t=%g' % t[n], zlim=[0, 2],colorbar=True, colormap=winter(), caxis=[0,2],shading='flat')

		if plot_method > 0:
			time.sleep(0.001) # pause between frames

		if save_plot:
			filename = 'tmp_%04d.png' % n
			#savefig(filename) # time consuming!



def test_plug(loc='center',sigma = 0.5):

	Lx = 10
	Ly = 10
	Nx = 100
	Ny = 100
	b = 0
	dt = -1
	c_0 = 1
	T = 10
	if loc == 'center':
		xc = Lx/2
	elif loc == 'left':
		xc = 0
	def I_x(x, y):
		 return 0 if abs(x-xc) > sigma else 1
	def V(x,y):
		return 0	
	def c(x,y):
		return c_0

	u_v, x, dt, max_E = solver(b, c, I_x, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None,version='vectorized',plug='ok',makeplot=None)
	u_s, x, dt, max_E = solver(b, c, I_x, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None,version='scalar',plug='ok',makeplot=None)
	
	diff = abs(u_s - u_v).max()
	nt.assert_almost_equal(diff, 0, places=13)
	u_0 = array([I_x(x_,0) for x_ in x])
	diff = abs(u_v[:,0] - u_0).max()
	nt.assert_almost_equal(diff, 0, places=13)

	#Test for y-axis
	if loc == 'center':
		yc = Ly/2
	elif loc == 'left':
		yc = 0
	def I_y(x, y):
		 return 0 if abs(y-yc) > sigma else 1

	u_v, x, dt, max_E = solver(b, c, I_y, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None,version='vectorized',plug='ok',makeplot=None)
	u_s, x, dt, max_E = solver(b, c, I_y, V, None, Lx, Ly, Nx, Ny, dt, T,user_action=None,version='scalar',plug='ok',makeplot=None)
	
	diff = abs(u_s - u_v).max()
	nt.assert_almost_equal(diff, 0, places=13)
	u_0 = array([I_y(0,y_) for y_ in x])		#Here I'm using x instead of y since they are similar
	diff = abs(u_v[0,:] - u_0).max()
	nt.assert_almost_equal(diff, 0, places=13)



	

	

	

def test_exact_solution_undamped():
		
	Lx = 2; Ly = 2
	#Nx = 4; Ny = 4
    	#dt = -1 # use longest possible steps
    	T = 1; b = 0.0

	def exact_solution(x, y, t):			 
		return  A*cos(k_x*x)*cos(k_y*y)*cos(w*t)

	def I(x,y):
		return exact_solution(x, y, 0)		
	def V(x,y):
		return -w*A*cos(k_x*x)*cos(k_y*y)*sin(0)
	def q(x,y):
		return 0.1
		
	A = 2.5; mx = 3.0; my = 2.0
	k_x = (mx*pi)/Lx; k_y = (my*pi)/Ly
	w = sqrt(q(0,0)*((k_x)**2 + (k_y)**2))	
			
		

	H_values = [0.5,0.3,0.2, 0.15 ,0.1,0.01]
	err_max = ones(len(H_values))*-1
	for i in range(len(H_values)):
		print "----------------"
		h = H_values[i]
		Nx = int(round(Lx/h))
		Ny = int(round(Lx/h))
		dt = h/2.
		
		def assert_no_error(b, q, u, x, xv, y, yv, t, n):			
			u_e = exact_solution(xv, yv, t[n])
			diff = abs(u - u_e).max()	
			if diff > err_max[i] or err_max[i]==-1:
				err_max[i] = diff 
			
		
		u, x, dt, cpu = solver(b, q, I, V, None, Lx, Ly, Nx, Ny, dt,\
			 T,user_action=assert_no_error, version='vectorized',exact='ok')
		#if i>0:
			#print "C=",err_max[i]/float(h**2)
			#print "diff_err=",(err_max[i]/float(H_values[i]**2))-(err_max[i-1]/float(H_values[i-1]**2))
	diff_err = (err_max[-1]/float(H_values[-1]**2))-(err_max[-2]/float(H_values[-2]**2))
	nt.assert_almost_equal(diff_err, 0, places=2)
	
	
def test_exact_solution_damped():
		
	Lx = 2; Ly = 2
	#Nx = 4; Ny = 4
    	#dt = -1 # use longest possible steps
    	T = 1; b = 0.3

	def exact_solution(x, y, t):			 
		return  (A*cos(w*t)+B*sin(w*t))*exp(-c*t) *cos(k_x*x)*cos(k_y*y)

	def I(x,y):
		return 	exact_solution(x, y, 0)	
	def V(x,y):
		return (-A*c*cos(0) - A*w*sin(0) - B*c*sin(0) + B*w*cos(0))*exp(0)*cos(k_x*x)*cos(k_y*y)
	def q(x,y):
		return 0.1
	c=b/2.0	
	A = 2.5; mx = 3.0; my = 2.0
	
	k_x = (mx*pi)/Lx; k_y = (my*pi)/Ly
	w = sqrt(q(0,0)*((k_x)**2 + (k_y)**2)-c**2)	
	B=A*c/w		
		

	H_values = [0.5,0.3,0.2, 0.15 ,0.1,0.05,0.01,0.005]
	err_max = ones(len(H_values))*-1
	for i in range(len(H_values)):
		#print "----------------"
		h = H_values[i]
		Nx = int(round(Lx/h))
		Ny = int(round(Lx/h))
		dt = h/2.
		
		def assert_no_error(b, q, u, x, xv, y, yv, t, n):			
			u_e = exact_solution(xv, yv, t[n])
			diff = abs(u - u_e).max()	
			if diff > err_max[i] or err_max[i]==-1:
				err_max[i] = diff 
			
		
		u, x, dt, cpu = solver(b, q, I, V, None, Lx, Ly, Nx, Ny, dt,\
			 T,user_action=assert_no_error, version='vectorized',exact='ok')
		#if i>0:
			#print "C=",err_max[i]/float(h**2)
			#print "diff_err=",(err_max[i]/float(H_values[i]**2))-(err_max[i-1]/float(H_values[i-1]**2))
		
	diff_err = (err_max[-1]/float(H_values[-1]**2))-(err_max[-2]/float(H_values[-2]**2))
	nt.assert_almost_equal(diff_err, 0, places=2)



if __name__ == '__main__':
	test_exact_solution_damped()
	#test_exact_solution_undamped()
	#test_plug()
	#test_plug(loc='center',sigma = 0.5)
	#test_constant()
	#cpu,dt=run_Gaussian(plot_method=2, version='vectorized', save_plot=True)
	#test_qubic()
	#cpu,dt=run_Gaussian(plot_method=2, version='scalar', save_plot=True)
