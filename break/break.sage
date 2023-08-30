from fpylll import IntegerMatrix, GSO, LLL, BKZ
import numpy as np

m=280
l=273
q=47
F=GF(47)
s=13

print("Loading data..")
h = np.loadtxt("h.txt")
A = np.loadtxt("A.txt")

# load BKZ-100 (jump=3) reduced q-ary basis of lattice A*Z^n+q*Z^m
# equivalent to +-BKZ-90
# took about 2-3 hours on 8 threads+1 gpu.
B100 = IntegerMatrix.from_file("B100.txt")
gso = GSO.Mat(B100)
gso.update_gso()

print("Starting babai reduction..")
def conv(x):
	x = x%q
	if x > q/2.:
		return x-q
	return x

f_conv = np.vectorize(conv)

def nrbounded(err):
	return np.sum(np.abs(f_conv(err)) <= s)

found = False
for i in range(50):
	for j in range(50):
		err=h+i*np.array(B100[-1])+j*np.array(B100[-2])-B100.multiply_left( gso.babai(h+i*np.array(B100[-1]) + j*np.array(B100[-2]), 0, 278))
		bounded = nrbounded(err)
		if bounded>=l:
			print("Solution found with minl = ", bounded, " >= l")
			# print(err)
			found = True
			break
	if found:
		break


## compute x
h_vec = vector(F, h)
e_vec = vector(F, err)
A_mat = Matrix(F, A)
x_vec=A_mat.solve_right(h_vec-e_vec)
print("x_vec = ", x_vec)

print("Verify solution..")
e_ver = h_vec - A_mat * x_vec
assert(e_vec == e_ver)
e_ZZ = vector(ZZ, e_ver)
assert(nrbounded(e_ZZ) >= l)
print("Solution verified, e=", vector(ZZ, f_conv(e_ZZ)))