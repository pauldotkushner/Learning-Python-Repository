import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg

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



def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(6)
plt.subplot(211)
lines = plt.plot(t1, f(t1), 'bo', t2, f(t2),'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.setp(lines[0])

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

#histogram
plt.figure(7)
n, bins, patches = plt.hist(x, 50, normed = 1, facecolor = 'g', alpha = 0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
t= plt.text(60, 0.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.setp(t, fontsize=18)

#using annotate
plt.figure(8)
ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
mpl.rcParams['font.size'] = 14
mpl.rc('lines',linewidth=6)
plt.annotate('local max', xy=(2,1), xytext = (3, 1.5), arrowprops = dict(facecolor='black', shrink = 0.05))
plt.ylim(-2,3)
ax.annotate('local max another command', xy=(3, 1),  xycoords='data',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='blue', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )

fig = plt.figure(9)
fig.suptitle('bold figure suptitle', fontsize = 14, fontweight = 'bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top = 0.85)
ax.set_title('axes title')
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')
ax.text(3, 8, 'boxed italics text in data coords', style = 'italic', bbox = {'facecolor':'red', 'alpha':0.5, 'pad':10})
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize = 15)
ax.text(3, 2, unicode('unicode: Institut f\374r Festk\366rperphysik', 'latin-1'))

ax.text(0.95, 0.01, 'colored text in axes coords', verticalalignment ='bottom', horizontalalignment = 'right',
        transform = ax.transAxes, color = 'green', fontsize = 15)

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy= (2,1) , xytext = (3,4), arrowprops = dict(facecolor='black', shrink = 0.05))

ax.axis([0, 10, 0, 10])


fig = plt.figure(10)
ax = fig.add_subplot(111, polar = True)
r = np.arange(0,1,0.001)
theta = 2*2*np.pi*r
line, = ax.plot(theta, r, color='#ee8d18', lw = 3)

ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'bo')
ax.annotate('a polar annotation', xy = (thistheta, thisr), xytext = (0.05, 0.05), textcoords='figure fraction', arrowprops=dict(facecolor='black', shrink = 0.05), horizontalalignment = 'left', verticalalignment = 'bottom')

import matplotlib.image as mpimg
plt.figure(11)
img=mpimg.imread('stinkbug.png')
imgplot = plt.imshow(img)
plt.figure(12)
imgplot = plt.imshow(img[:,:,0])
plt.figure(13)
imgplot = plt.imshow(img[:,:,0])
imgplot.set_cmap('hot')
plt.colorbar()
plt.figure(14)
lum_img = img[:,:,0]
plt.subplot(311)
plt.hist(lum_img.flatten(), 256, range = (0,1.0), fc='k', ec='k')
plt.subplot(312)
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('spectral')
plt.subplot(313)
imgplot2 = plt.imshow(lum_img)
imgplot2.set_cmap('spectral')
imgplot2.set_clim(0.0, 0.7)


