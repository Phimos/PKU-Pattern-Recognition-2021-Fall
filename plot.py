# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 5, 0.01)
fx = 1 * np.abs(x - 0) - 1 * np.abs(x-3) + 1 * np.abs(x - 4)
plt.plot(x, fx)
plt.title("$f(x)$")

plt.savefig("plot.png", dpi=300)

# %%

x = np.arange(-1, 5, 0.01)
b1x = -1/24 * np.abs(x - 0) + 1/6 * np.abs(x-3) + 1 / 8 * np.abs(x - 4)
b2x = 1/6 * np.abs(x-0) - 2/3 * np.abs(x-3) + 1 / 2 * np.abs(x - 4)
b3x = 1/8 * np.abs(x-0) + 1/2 * np.abs(x-3) - 3 / 8 * np.abs(x - 4)
plt.plot(x, b1x)
plt.plot(x, b2x)
plt.plot(x, b3x)
plt.title("$f(x)$")

# %%
plt.plot(x, 1*b1x+4*b2x+3*b3x)
# %%
plt.plot(x, 1*b1x, label="$y_1 b_1(x)$")
plt.plot(x, 4*b2x, label="$y_2 b_2(x)$")
plt.plot(x, 3*b3x, label="$y_3 b_3(x)$")
plt.legend()
plt.title("$y_i b_i(x)$")
plt.savefig("plot.png", dpi=300)
# %%
