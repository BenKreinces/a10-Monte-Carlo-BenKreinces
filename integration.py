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
logger.info(f"Starting two-circle simulation: N={N}, R={R}, seed={args.seed}")
logger.debug(f"Circle sampling bounds: x=[{-3*R/2}, {3*R/2}], y=[{-R}, {R}]")
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
logger.info(f"Circle overlap: hits={H}, fraction={H/N:.6f}, estimated area={AoverlapMes/R**2:.6f} R^2")
logger.debug(f"Exact area={Aoverlap/R**2:.6f} R^2, relative error={abs(AoverlapMes-Aoverlap)/Aoverlap*100:.2f}%")
logger.debug(f"Circle counts: left={leftCount}, right={rightCount}, both={bothCount}, neither={np.sum(noCirc)}")
if logger.isEnabledFor(VERBOSE):
    for i in range(N):
        logger.log(VERBOSE, f"Circle point {i+1}: x={x[i]:.5f}, y={y[i]:.5f}, overlap={bothCirc[i]}")
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
logger.info(f"Two-circle simulation completed")


#Two spheres
R = 1
N = 100000
logger.info(f"Starting two-sphere simulation: N={N}, R={R}")
logger.debug(f"Sphere sampling bounds: x=[{-R/2}, {R/2}], y=[{-R}, {R}], z=[{-R}, {R}]")
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
logger.info(f"Two-sphere overlap: hits={H}, fraction={H/N:.6f}, estimated volume={V2/R**3:.6f} R^3")
logger.debug(f"Two-sphere exact volume={Vexact/R**3:.6f} R^3, relative error={abs(V2-Vexact)/Vexact*100:.2f}%")
logger.debug(f"Sphere counts: left only={leftCount}, right only={rightCount}, both={bothCount}, neither={np.sum(noCirc)}")
if logger.isEnabledFor(VERBOSE):
    for i in range(N):
        logger.log(VERBOSE, f"Sphere point {i+1}: x={x[i]:.5f}, y={y[i]:.5f}, z={z[i]:.5f}, two-sphere hit={bothCirc[i]}")

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
logger.info(f"Two-sphere simulation completed")

# Three spheres
logger.info(f"Starting three-sphere intersection using the same N={N} points")
thirdCirc = x**2 + (y - R/2)**2 + z**2 <= R**2

allThree = bothCirc & thirdCirc

H3 = np.sum(allThree)
V3 = VB * H3 / N
logger.info(f"Three-sphere overlap: hits={H3}, fraction={H3/N:.6f}, estimated volume={V3/R**3:.6f} R^3")
logger.debug(f"Three-sphere vs two-sphere hits: H3={H3}, H2={H}, reduction={H-H3}")
if logger.isEnabledFor(VERBOSE):
    for i in range(N):
        logger.log(VERBOSE, f"Three-sphere point {i+1}: third hit={thirdCirc[i]}, all-three hit={allThree[i]}")

print(f"Points in all three spheres: {H3}")
print(f"Three-sphere acceptance fraction: {H3/N}")
print(f"Estimated three-sphere overlap volume: {V3/R**3}")


# Geometric checks

if np.all(~allThree | bothCirc):
    print("Three-sphere hit and Two-sphere hit: Passed")
    logger.info(f"Three-sphere hit and Two-sphere hit: Passed")
else:
    print("Three-sphere hit and Two-sphere hit: Failed")
    logger.info(f"Three-sphere hit and Two-sphere hit: Failed")


if 0 <= V3 <= V2 <= VB:
    print("Volume inequality check: Passed")
    logger.info(f"Volume inequality check: Passed")
else:
    print("Volume inequality check: Failed")
    logger.info(f"Volume inequality check: Failed")

def checkPoint(x, y, z):
    sphere1 = (x + R/2)**2 + y**2 + z**2 <= R**2
    sphere2 = (x - R/2)**2 + y**2 + z**2 <= R**2
    sphere3 = x**2 + (y - R/2)**2 + z**2 <= R**2

    result = (sphere1 and sphere2, sphere1 and sphere2 and sphere3)
    logger.debug(f"checkPoint(x={x}, y={y}, z={z}) -> {result}")
    return result

# Origin: inside all three
if checkPoint(0, 0, 0) == (True, True):
    print("Origin check: Passed")
    logger.info(f"Origin check: Passed")
else:
    print("Origin check: Failed")
    logger.info(f"Origin check: Failed")

# Inside first two, outside third
if checkPoint(0, -3*R/4, 0) == (True, False):
    print("Two-sphere-only point check: Passed")
    logger.info(f"Two-sphere-only point check: Passed")
else:
    print("Two-sphere-only point check: Failed")
    logger.info(f"Two-sphere-only point check: Failed")

# Outside all three
if checkPoint(0, -R, 0) == (False, False):
    print("Outside point check: Passed")
    logger.info(f"Outside point check: Passed")
else:
    print("Outside point check: Failed")
    logger.info(f"Outside point check: Failed")
logger.info(f"Three-sphere simulation and geometric checks completed")
