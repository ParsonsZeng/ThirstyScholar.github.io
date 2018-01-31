import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
plt.style.use('seaborn-paper')
fig, ax = plt.subplots(1, 1)

# Param. for Gamma distribution
alpha, beta = 6, 6

# Separate x-axis to conver cum. prob.1% ~ 99%
x = np.linspace(gamma.ppf(.01, alpha, beta),
                gamma.ppf(.99, alpha, beta),
                100)

# Plot
ax.plot(x, gamma.pdf(x, alpha, beta), label='Gam({0}, {1})'.format(alpha, beta))

plt.title('Pdf of Gamma Distrubtion')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

