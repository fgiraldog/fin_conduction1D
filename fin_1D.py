#Importacion de paquetes importantes para el programa
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

#Definicion de variables globales para el programa
global h
global k
global w
global theta_b
global L

h = 0.2
k = 1.
w = 1.
theta_b = 1.
L = 1. 

#Funcion para calcular eficiencia
def efficiency(x,forma_sup,forma_inf):
	#Definicion de variables necesarias para resolver el problema
	delta_x2 = (x[1]-x[0])**2
	perimetro = 2*(forma_sup-forma_inf) + 2*w
	area = (forma_sup-forma_inf)*w

	#Matriz asociada al problema
	A = np.zeros((len(x)-1,len(x)-1))

	#Aca se llena la matriz
	for i in range(0,np.shape(A)[0]-1):
		for j in range(0,np.shape(A)[1]-1):
			if i==j: #Diagonal principal
				A[i,j] = (-2*area[i+1]/delta_x2)-(h*perimetro[i+1]/k)
			elif j==i+1: #Diagonal arriba de la principal
				A[i,j] = (area[i+1]/delta_x2)-(area[i]/(4.*delta_x2))+(area[i+2]/(4.*delta_x2))
			elif j==i-1: #Diagonal abajo de la principal
				A[i,j] = (area[i+1]/delta_x2)+(area[i]/(4.*delta_x2))-(area[i+2]/(4.*delta_x2))

	#Condiciones de frontera para el ultimo nodo
	A[-1,-1] = -3.
	A[-1,-2] = 4.
	A[-1,-3] = -1.

	#Solucion al problema Ax = b donde x son las temperaturas
	b = np.zeros(len(x)-1)
	b[0] = -theta_b*((area[1]/delta_x2)-(area[2]/(4.*delta_x2))+(area[0]/(4.*delta_x2)))
	theta = np.linalg.solve(A,b)
	theta = np.append(theta_b,theta)

	#Calculo de la eficiencia
	q_real = np.trapz(perimetro*theta*h,x)
	q_max = np.trapz(perimetro*h*theta_b,x)
	effi = q_real/q_max

	return effi, theta

#Funcion para la optimizacion de la forma (DESCENSO DE GRADIENTE POR NODO)
def gradient_descent(x,forma_gd,forma_inf,file_name):
	#Inicializa la camara para la animacion
	fig = plt.figure(figsize = (9,4))
	camera = Camera(fig)

	#While referente a las iteraciones
	cond = True
	effi_max = 0
	i = 0
	while cond == True:
		d_effi = []
		for j in range(0,len(x)):
			forma_try = np.copy(forma_gd)
			delta_B = np.random.rand()*0.001
			forma_try[j] += delta_B
			effi1, theta1 = efficiency(x,forma_gd,forma_inf)
			effi2, theta2 = efficiency(x,forma_try,forma_inf)

			#Calculo del gradiente para cada B(x_i)
			d_effi.append((effi2-effi1)/delta_B)

		#Paso para acercarse a la eficiencia maxima
		forma_gd += 0.01*np.array(d_effi)
		
		#Condiciones referentes al ancho de la aleta
		for k in range(0,len(x)):
			if forma_gd[k]>0.3:
				forma_gd[k] = 0.3
			if forma_gd[k]<0.0:
				forma_gd[k] = 0.0

		#Calculo de la eficiencia del perfil despues del paso
		effi, theta = efficiency(x,forma_gd,forma_inf)

		#Esta condicion es muy importante, ya que previene que el problema se vuelva puramente matematico.
		#Considerando que no hay generacion de calor, la temperatura en el nodo i tiene que ser menor a la temperatura en e nodo i+1
		for y in range(0,len(theta)-1):
			if theta[y]<theta[y+1] or theta[y] < 0:
				cond = False

		#Toma de la foto para la animacion
		if cond and effi>effi_max and i%5 == 0:
			effi_max = effi
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
			plt.suptitle('Descenso de gradiente (Por Nodo)')
			plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
			camera.snap()

		i += 1
	#Creacion de la animacion
	animation = camera.animate()
	animation.save('{}.gif'.format(file_name))

#Funciones para la optimizacion de la forma (DESCENSO DE GRADIENTE POR AJUSTE POLINOMIAL)

def perfil_sup(x, a):
	#Ajuste polinomial y condiciones referentes al ancho de la aleta
    r=np.zeros(len(x))
    for i in range(len(a)):
        r+=a[i]*x**i 
    for k in range(0,len(r)):
            if r[k]>0.3:
                r[k] = 0.3
            if r[k]<0.0:
                r[k] = 0.0
    return r

def gradient_descentpoly(x,a,forma_inf,file_name):
	#Calculo de la forma polinomial
	forma_gd = perfil_sup(x, a)

	#Inicializa la camara para la animacion
	fig = plt.figure(figsize = (9,4))
	camera = Camera(fig)

	#While referente a las iteraciones
	a_max = np.copy(a)
	effi_max = 0
	cond = True 
	i = 0
	while cond:
		d_effi = []

		for j in range(len(a)):
			a_try = np.copy(a)
			delta_B = -0.1
			a_try[j] += delta_B
			forma_try = perfil_sup(x,a_try)
			effi1, theta1 = efficiency(x,forma_gd,forma_inf)
			effi2, theta2 = efficiency(x,forma_try,forma_inf)
			d_effi.append((effi2-effi1)/delta_B)

		a += 0.005*np.array(d_effi)
		forma_gd = perfil_sup(x, a)

		#Calculo de la eficiencia del perfil despues del paso
		effi, theta = efficiency(x,forma_gd,forma_inf)

        #Esta condicion es muy importante, ya que previene que el problema se vuelva puramente matematico.
		#Considerando que no hay generacion de calor, la temperatura en el nodo i tiene que ser menor a la temperatura en e nodo i+1
		for y in range(len(theta)-1):
			if theta[y] < theta[y+1]:
				cond=False

        #Toma de la foto para la animacion
		if cond == True and effi>effi_max and i%5 == 0:
			effi_max = effi
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
			plt.suptitle('Descenso de gradiente (Ajuste Polinomial)')
			plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
			camera.snap()

		i += 1
	#Creacion de la animacion   
	animation = camera.animate()
	animation.save('{}.gif'.format(file_name))

# Punto 1.
x = np.linspace(0,L,20)
b_0 = 0.3
b_1 = 0.3
forma_supt = b_0 - b_0*x/L
forma_inft = -b_1 + b_1*x/L

eff_1, theta_1 = efficiency(x,forma_supt,forma_inft)

print('La eficiencia del primer punto es: ' + str('{:.3f}'.format(eff_1)))

# Punto 2.
forma_supr = np.ones((20))*b_0
forma_infr = x*0.

eff_2, theta_2 = efficiency(x,forma_supr,forma_infr)

print('La eficiencia del segundo punto es: ' + str('{:.3f}'.format(eff_2)))

# Punto 3. ##IMPORTANTE: Para correr cada uno, se debe descomentar de a uno y comentar el resto (Por la animacion)

#Aleta triangular
a = np.array([b_0,0,0,0,0])
# gradient_descentpoly(x,a,forma_inft,'poly_tria')
# gradient_descent(x,forma_supr,forma_inft,'node_tria')


#Aleta rectangular
# gradient_descentpoly(x,a,forma_infr,'poly_rect')
# gradient_descent(x,forma_supr,forma_infr, 'node_rect')