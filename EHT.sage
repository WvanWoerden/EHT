from fpylll.util import gaussian_heuristic
from modelBKZ import construct_BKZ_shape
import numpy as np
import scipy as sc

# parameters level 1
q = 47
m = 460
n = 242
l=451
s=13
b=170

# parameters level 3
# q = 47
# m = 696
# n = 367
# l=684
# s=13
# b=300

# parameters level 5
# q = 47
# m = 940
# n = 495
# l=921
# s=13
# b=400

# we run BKZ with blocksize b to reduce the basis
shape=construct_BKZ_shape(q, m-n, n, b)
nq=shape[0]
logprof=np.array(shape[2])
prof = np.exp(2*logprof) # squared gso norms

# we assume that the target t is random
# we use Babai-lift on the last part [nq+b:m),
# to reduce t to a smaller (in the context [nq+b:m) ) target t'.
# the expected squared length of pi_{nq+b}(t') is:
dim_tail = m - (nq+b)
sqnorm_tail = sum(prof[nq+b:])/12. #if we use an exact CVP solver here: gaussian_heuristic(prof[nq+b:])
coeff_variance_tail = sqnorm_tail/dim_tail
print("||pi_{nq+b}(t')|| = ", sqrt(sqnorm_tail), " <= s")
# print("SD per coefficient: ", sqrt(coeff_variance_tail))
if sqrt(sqnorm_tail) <= s:
	print("||pi_{nq+b}(t')|| <= s, so in particular the infinity norm is (under any coefficient embedding)")

# The middle part of the lattice is sufficiently random to 
# use the gaussian heuristic. 
# Use BDGL sieve in [nq:nq+b] to obtain the (4/3)^(b/2)
# shortest vectors in that context.
# Then rerandomize pi_nq(t') in the context [nq:nq+b) 
# to obtain N=(4/3)^(b/2+o(b)) targets.
# we run a batched randomized slicer using BDGL nearest neighbour
# techniques (like in https://eprint.iacr.org/2020/120.pdf).
# This gives N solutions to sqrt(4/3)-approxCVP in the
# context [nq:nq+b), i.e. we have N target cosets t_1,..,t_N of 
# for which the l2 norm on [nq:nq+b] is at most sqrt(4/3)*gh(nq:nq+b)
# The cost of this is similar to a usual BDGL sieve, i.e. 2^(0.292*b+o(b))
dim_middle = b
sqnorm_middle = 4/3.* gaussian_heuristic(prof[nq:nq+b])
coeff_variance_middle = sqnorm_middle/dim_middle
print("||pi_{[nq:nq+b)}(t_i)|| <= ", sqrt(sqnorm_middle))
print("SD per coefficient: ", sqrt(coeff_variance_middle))

# we assume that the direction of pi_{[nq:nq+b)}(t_i)
# is uniform random, as to determine how likely the 
# infinity norm is bounded by s. 
# Then each squared coefficients follow (up to appropriate scaling) 
# a beta(1/2, (b+1)/2) distribution. (see https://scholarlypublications.universiteitleiden.nl/handle/1887/3564770 Lemma 49)
alpha = s/sqrt(sqnorm_middle)
prob_bounded_middle = 1.- b*(1.-sc.special.betainc(0.5, (b+1)/2, alpha^2))
print("Probability that infinity norm is less than ", s, " is lower bounded by ", prob_bounded_middle)

# Now we find a close vector to each t_1, ..., t_N in the context [0:nq) 
# by simply rounding each coefficient to q*Z^nq.
# note that (1+2*s) of the q possible values are small enough.
# The min_l definition of the paper allows m-l of them to have absolute value
# bigger than s. Assuming each target is uniform over qZ^nq/Z^n we thus obtain
# a binomial distribution with p=(1+2*s)/q, n=nq, and we want to know the probability
# that there are at most m-l mistakes, i.e. at least nq-(m-l) successes.
# See https://en.wikipedia.org/wiki/Binomial_distribution
p = (2*s+1)/float(q)
prob_lift = sc.special.betainc(nq-(m-l), (m-l)+1, p)
targets = prob_bounded_middle * (4/3.)^(b/2.)
print("Probability of single lift having min_l <= s, is ", prob_lift)
print("Expected solutions over all targets: ", targets*prob_lift)