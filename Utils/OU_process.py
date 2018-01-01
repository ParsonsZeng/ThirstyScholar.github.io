t_0 = 0
t_end = 10
length = 11000
theta = .1
mu = .8
sigma = 1.

t = np.linspace(t_0, t_end, length)
dt = np.mean(np.diff(t))

y = np.zeros(length)
y[0] = 1.

noise = np.random.randn(length) * np.sqrt(dt)

# Solve SDE
for i in range(1, length):
    y[i] = y[i-1] + theta * (mu - y[i-1]) * dt + sigma * noise[i]

# Shift to avoid negative price
print('min:', y.min())
if y.min() < 0: y -= (y.min() - .1)

print(y.shape)
plt.plot(y)
plt.show()
