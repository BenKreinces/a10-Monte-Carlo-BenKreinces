import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

l = 1
d = 1
N = 100000
x = rng.random(N) * d/2
theta = rng.random(N) * np.pi / 2
mask = x <= (l/2) * np.sin(theta)
P = 2 * l / np.pi / d
Htotal = np.sum(mask)
Pmes = Htotal / N
pimes = 2 * l * N / d / Htotal if Htotal > 0 else np.nan

j = np.arange(1,N+1)
H = np.cumsum(mask)
Phat = H / j
pihat = np.where(H>0, 2 * l * j / d / np.maximum(H, 1), np.nan) # Still replaces 0 vals with np.nan, np.maximum used to not throw runtime warnings

plt.plot(j, pihat)
plt.xscale('log')
plt.axhline(np.pi, color='red', linestyle='--', label='True pi')
plt.xlabel('Number of drops')
plt.ylabel('Estimated pi')
plt.legend()
plt.show()

error = (pimes - np.pi) / np.pi

print(f"Crossings: {Htotal}")
print(f"Theoretical probability: {P}")
print(f"Measured probability: {Pmes}")
print(f"Estimated pi: {pimes}")
print(f"Relative error: {error}")

M = 30


x = rng.random((M, N)) * d/2
theta = rng.random((M, N)) * np.pi / 2
mask = x <= (l/2) * np.sin(theta)
j = np.arange(1,N+1)
H = np.cumsum(mask, axis=1)
pihat = np.where(H>0, 2 * l * j / d / np.maximum(H, 1), np.nan) # Still replaces 0 vals with np.nan, np.maximum used to not throw runtime warnings
for i in range(M):
    plt.plot(j, pihat[i])
plt.xscale('log')
plt.axhline(np.pi, color='red', linestyle='--', label='True pi')
plt.xlabel('Number of drops')
plt.ylabel('Estimated pi displaying 30 trials')
plt.legend()
plt.show()

