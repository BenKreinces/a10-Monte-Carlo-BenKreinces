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
rightCirc = ((x - R/2))**2 + y**2 <= R**2 & ((x + R/2)**2 + y**2 > R**2)
bothCirc = ((x - R/2))**2 + y**2 <= R**2 & ((x + R/2)**2 + y**2 <= R**2)
noCirc = ((x - R/2))**2 + y**2 > R**2 & ((x + R/2)**2 + y**2 > R**2)
AB = 6 * R**2
H = np.sum(bothCirc)
AoverlapMes = AB * H / N
Aoverlap = (2 * np.pi / 3 - np.sqrt(3)/2) * R**2
print(Aoverlap)
print(AoverlapMes)