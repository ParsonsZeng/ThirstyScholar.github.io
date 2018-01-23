import numpy as np


def Ema(data, window):

    alpha = 2 / (window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n+1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev**(n-1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


data = np.random.randint(2, 9, (200))
window = 20

ema = Ema(data, window)
print(ema)

import matplotlib.pyplot as plt
plt.plot(data)
plt.plot(ema)
plt.show()

