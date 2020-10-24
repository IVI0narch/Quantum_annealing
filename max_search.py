from dwave.system import DWaveSampler, EmbeddingComposite
import random
import pyqubo


sampler = EmbeddingComposite(DWaveSampler())

N = int(input())
m = int(input())

weight = [random.randint(-100, 100) for i in range(N)]
x = pyqubo.Array.create('bin_array', shape=N, vartype='BINARY')

lagrangian = 0
for elem in weight:
    lagrangian += abs(elem)

hamiltonian = - sum(a * j for a, j in zip(weight, x)) + lagrangian * (sum(i for i in x) - m)**2

Q, offset = hamiltonian.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=1000)
print(sampleset)  # doctest: +SKIP
print(' '.join(map(str, weight)))
