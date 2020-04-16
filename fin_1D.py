import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

global h
global k
global w
global theta_b

h = 0.2
k = 1.
w = 1.
theta_b = 50.

#Funcion para calcular eficiencia
def efficiency(x,forma_sup,forma_inf):
	delta_x2 = (x[1]-x[0])**2
	perimetro = (2*forma_sup)+(2*-forma_inf)+2*w
	area = ((forma_sup)+(-forma_inf))*w

	A = np.zeros((len(x)-1,len(x)-1))

	for i in range(0,np.shape(A)[0]-1):
		for j in range(0,np.shape(A)[1]-1):
			if i==j:
				A[i,j] = (-2*area[i+1]/delta_x2)-(h*perimetro[i+1]/k)
			if j==i+1:
				A[i,j] = (area[i+1]/delta_x2)-(area[i]/(4.*delta_x2))+(area[i+2]/(4.*delta_x2))
			if j==i-1:
				A[i,j] = (area[i+1]/delta_x2)+(area[i]/(4.*delta_x2))-(area[i+2]/(4.*delta_x2))
	A[-1,-1] = -3.
	A[-1,-2] = 4.
	A[-1,-3] = -1.

	b = np.zeros(len(x)-1)
	b[0] = -theta_b*((area[1]/delta_x2)-(area[2]/(4*delta_x2))+(area[0]/(4*delta_x2)))
	theta = np.linalg.solve(A,b)
	theta = np.append(theta_b,theta)

	q_real = np.trapz(perimetro*theta,x)*h
	q_max = np.trapz(perimetro,x)*h*theta_b
	effi = q_real/q_max

	return effi, theta

#Funcion para la optimizacion de la forma
def gradient_descent(x,forma_gd):
	forma_inf = x*0.
	fig = plt.figure(figsize = (9,4))
	camera = Camera(fig)
	for i in range(0,300):
		d_effi = []
		for j in range(0,len(x)):
			forma_try = np.copy(forma_gd)
			delta_B = 0.01
			forma_try[j] += delta_B
			effi1, theta1 = efficiency(x,forma_gd,forma_inf)
			effi2, theta2 = efficiency(x,forma_try,forma_inf)

			d_effi.append((effi2-effi1)/delta_B)

		forma_gd += 0.01*np.array(d_effi)
		
		for k in range(0,len(x)):
			if forma_gd[k]>0.5:
				forma_gd[k] = 0.5
			if forma_gd[k]<0.0:
				forma_gd[k] = 0.0

		effi, theta = efficiency(x,forma_gd,forma_inf)
		plt.subplot(121)
		plt.plot(x,theta, c = 'c')
		plt.xlabel('$x (m)$')
		plt.ylabel(r'$\theta (^\circ C)$')
		plt.title('Dist. de Temperatura')
		plt.subplot(122)
		t = plt.plot(x,forma_gd, c = 'b')
		plt.plot(x,forma_inf, c = 'b')
		plt.xlabel('$x (m)$')
		plt.ylabel('$y (m)$')
		plt.title('Forma de la Aleta')
		plt.legend(t, ['$\eta$ = {:.4g}'.format(effi)])
		plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
		camera.snap()

	animation = camera.animate()
	plt.show()

# Punto 1.
L = 1. 
x = np.linspace(0,L,20)
b_0 = 0.3
b_1 = 0.3
forma_sup = b_0 - b_0*x/L
forma_inf = -b_1 + b_1*x/L

eff_1, theta_1 = efficiency(x,forma_sup,forma_inf)

print('La eficiencia del primer punto es: ' + str('{:.3f}'.format(eff_1)))

# Punto 2.
b_1 = 0.0
forma_sup = np.ones((20))*b_0
forma_inf = np.ones((20))*b_1

eff_2, theta_2 = efficiency(x,forma_sup,forma_inf)

print('La eficiencia del segundo punto es: ' + str('{:.3f}'.format(eff_2)))

# Punto 3.
gradient_descent(x,forma_sup)



