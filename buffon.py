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

l = 1
d = 1
N = 100000

logger.info(f"Starting single Buffon experiment: N={N}, seed={args.seed}")
logger.debug(f"Starting single buffon(N={N}, l={l}, d={d})")
x = rng.random(N) * d/2
theta = rng.random(N) * np.pi / 2
mask = x <= (l/2) * np.sin(theta)
P = 2 * l / np.pi / d
Htotal = np.sum(mask)
Pmes = Htotal / N
pimes = 2 * l * N / d / Htotal if Htotal > 0 else np.nan
logger.debug(f"Single experiment: H={Htotal}, Pmes={Pmes}, pimes={pimes}")

j = np.arange(1,N+1)
H = np.cumsum(mask)
Phat = H / j

logger.info("Starting pi estimation")
pihat = np.where(H>0, 2 * l * j / d / np.maximum(H, 1), np.nan) # Still replaces 0 vals with np.nan, np.maximum used to not throw runtime warnings

logger.debug(f"Running estimate: first={pihat[0]}, last={pihat[-1]}")
logger.info("Plotting pi estimation")

for i in [0, 9, 99, 999, 9999, N-1]:
    logger.log(VERBOSE, f"Drop {j[i]}: H={H[i]}, estimated pi={pihat[i]}")

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

logger.info(f"Single experiment completed: pi={pimes}, relative error={error}")

M = 30
logger.info(f"Starting {M} independent experiments")
logger.debug(f"Starting multi-buffon(M={M}, N={N}, l={l}, d={d})")

x = rng.random((M, N)) * d/2
theta = rng.random((M, N)) * np.pi / 2
mask = x <= (l/2) * np.sin(theta)
j = np.arange(1,N+1)
H = np.cumsum(mask, axis=1)
pihat = np.where(H>0, 2 * l * j / d / np.maximum(H, 1), np.nan) # Still replaces 0 vals with np.nan, np.maximum used to not throw runtime warnings
for i in range(M):
    logger.log(VERBOSE, f"Experiment {i+1}: H={H[i, -1]}, estimated pi={pihat[i, -1]}")
    plt.plot(j, pihat[i])
plt.xscale('log')
plt.axhline(np.pi, color='red', linestyle='--', label='True pi')
plt.xlabel('Number of drops')
plt.ylabel('Estimated pi displaying 30 trials')
plt.legend()
plt.show()

logger.info(f"Completed {M} independent experiments")
logger.debug(f"Final estimates: mean={np.nanmean(pihat[:, -1])}, std={np.nanstd(pihat[:, -1])}")