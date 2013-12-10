import matplotlib.pyplot as plt

import numpy as np
plt.figure(1)
plt.plot([1,2,3,4])
plt.ylabel('some numbers ')

plt.figure(2)
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])

plt.figure (3)
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^', linewidth = 2.0)

plt.figure(4)
lines = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
lines[0].set_lw(5.0)
plt.setp(lines[1])


plt.figure(5)
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(6)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2),'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plot.show()