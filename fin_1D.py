import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


b_0 = 0.5
b_1 = 0.5
L = 1.
w = 0.5
k = 21.
h = 35.

theta_b = 50.
theta_l = 5.

x = np.linspace(0,L,20)
perimetro = (2*(b_0-(b_0*x/L)))+(2*(b_1-(b_1*x/L)))+2*w
area = ((b_0-(b_0*x/L))+(b_1-(b_1*x/L)))*w
delta_x2 = (x[1]-x[0])**2

A = np.zeros((18,18))

for i in range(0,np.shape(A)[0]):
	for j in range(0,np.shape(A)[1]):
		if i==j:
			A[i,j] = (-2*area[i+1]/delta_x2)-(h*perimetro[i+1]/k)
		if j==i+1:
			A[i,j] = (area[i+1]/delta_x2)-(area[i]/(4.*delta_x2))+(area[i+2]/(4.*delta_x2))
		if j==i-1:
			A[i,j] = (area[i+1]/delta_x2)+(area[i]/(4.*delta_x2))-(area[i+2]/(4.*delta_x2))

b = np.zeros(18)
b[0] = -theta_b*((area[1]/delta_x2)-(area[2]/(4*delta_x2))+(area[0]/(4*delta_x2)))
b[-1] = -theta_l*((area[-2]/delta_x2)-(area[-3]/(4*delta_x2))+(area[-1]/(4*delta_x2)))

theta = np.linalg.solve(A,b)
theta = np.append(theta_b,theta)
theta = np.append(theta,theta_l)

plt.figure()
plt.plot(x,theta)
plt.xlabel('$x (m)$')
plt.ylabel(r'$\theta (^\circ C)$')
plt.show()