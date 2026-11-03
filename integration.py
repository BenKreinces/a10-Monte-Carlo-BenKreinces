import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--log-level", type=str.upper, choices = ["INFO","DEBUG","VERBOSE"], default="INFO", help = "Logging detail(default:INFO)")
parser.add_argument("--seed", type=int, default=22, help="Random seed (default: 22)")
args = parser.parse_args()
levels = {
"INFO": logging.INFO,
"DEBUG": logging.DEBUG,
"VERBOSE": VERBOSE,
}
logging.basicConfig(level = levels[args.log_level], format = "%(levelname)s: %(message)s",
)
logger.info("Starting simulation")
logger.debug("Selected log level: %s",args.log_level)
logger.log(VERBOSE, "Detailed logging enabled")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

rng = np.random.default_rng(seed=args.seed)

#Two offset circles
R = 1
N = 10000
xRange = 3*R / 2
yRange = R
x = rng.random(N) * 2 * xRange - xRange
y = rng.random(N) * 2 * yRange - yRange
leftCirc = ((x + R/2)**2 + y**2 <= R**2) & ((x - R/2)**2 + y**2 > R**2)
rightCirc = ((x - R/2)**2 + y**2 <= R**2) & ((x + R/2)**2 + y**2 > R**2)
bothCirc = ((x - R/2)**2 + y**2 <= R**2) & ((x + R/2)**2 + y**2 <= R**2)
noCirc = ((x - R/2)**2 + y**2 > R**2) & ((x + R/2)**2 + y**2 > R**2)
AB = 6 * R**2
H = np.sum(bothCirc)
AoverlapMes = AB * H / N
Aoverlap = (2 * np.pi / 3 - np.sqrt(3)/2) * R**2

leftCount = np.sum(leftCirc)
rightCount = np.sum(rightCirc)
bothCount = np.sum(bothCirc)
print(f"Points in neither circle: {np.sum(noCirc)}")
print(f"Points in left circle: {leftCount}")
print(f"Points in right circle: {rightCount}")
print(f"Points in both circles: {bothCount}")
print(f"Estimated overlap fraction: {H/N}")
print(f"Estimated area: {AoverlapMes / R**2}")
print(f"Relative error: {np.abs(AoverlapMes - Aoverlap) / Aoverlap * 100:.2f}%")

plotLeft = x[leftCirc], y[leftCirc]
plotRight = x[rightCirc], y[rightCirc]
plotBoth = x[bothCirc], y[bothCirc]
plotNone = x[noCirc], y[noCirc]
plt.plot(plotLeft[0], plotLeft[1], 'b.', label='Left Circle Only', markersize=0.7)
plt.plot(plotRight[0], plotRight[1], 'g.', label='Right Circle Only', markersize=0.7)
plt.plot(plotBoth[0], plotBoth[1], 'r.', label='Both Circles', markersize=0.7)
plt.plot(plotNone[0], plotNone[1], 'k.', label='Neither Circle', markersize=0.7)
plt.xlabel('x')
plt.ylabel('y')
theta = np.linspace(0, 2*np.pi, 1000)
plt.plot(-R/2 + R*np.cos(theta), R*np.sin(theta), 'b-', linewidth=2, label='Left Boundary')
plt.plot(R/2 + R*np.cos(theta), R*np.sin(theta), 'g-', linewidth=2, label='Right Boundary')
plt.plot(-R/2, 0, 'bo', markersize=5)
plt.plot(R/2, 0, 'go', markersize=5)
plt.axis('equal')
plt.legend(markerscale=5)
plt.show()


#Two spheres
R = 1
N = 100000
xRange = R / 2
yRange = R
zRange = R
x = rng.random(N) * 2 * xRange - xRange
y = rng.random(N) * 2 * yRange - yRange
z = rng.random(N) * 2 * zRange - zRange
leftCirc = ((x + R/2)**2 + y**2 + z**2 <= R**2) & ((x - R/2)**2 + y**2 + z**2 > R**2)
rightCirc = ((x - R/2)**2 + y**2 + z**2 <= R**2) & ((x + R/2)**2 + y**2 + z**2 > R**2)
bothCirc = ((x - R/2)**2 + y**2 + z**2 <= R**2) & ((x + R/2)**2 + y**2 + z**2 <= R**2)
noCirc = ((x - R/2)**2 + y**2 + z**2 > R**2) & ((x + R/2)**2 + y**2 + z**2 > R**2)
VB = 4 * R**3
H = np.sum(bothCirc)
Vexact = 5 * np.pi / 12 * R**3
V2 = VB * H / N
leftCount = np.sum(leftCirc)
rightCount = np.sum(rightCirc)
bothCount = np.sum(bothCirc)

print(f"Points in both spheres: {bothCount}")
print(f"Estimated overlap fraction: {H/N}")
print(f"Estimated volume: {V2/R**3}")
print(f"Relative error: {np.abs(V2 - Vexact) / Vexact * 100:.2f}%")
temp = np.cumsum(bothCirc) / R**3
Vrunning = VB * np.cumsum(bothCirc) / np.arange(1, N+1)
plt.plot(np.arange(1, N+1), Vrunning, 'b-', linewidth=1, label='Monte Carlo estimate')
plt.axhline(Vexact, color='red', linestyle='--', label='Exact volume')
plt.xscale('log')
plt.xlabel('Number of random points')
plt.ylabel('Estimated overlap volume')
plt.legend()
plt.show()
plt.show()