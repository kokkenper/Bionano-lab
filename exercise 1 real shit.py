from __future__ import division, unicode_literals, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
mpl.rc('figure', figsize=(10,5))
mpl.rc('image', cmap='gray')
import pims as ps
import trackpy as tp
Min_Mass=1.7*10**2
def main():
    @ps.pipeline
    def gray(image):
        return image[ :, :, 1]  # Take just the green channel
    frames = gray(ps.open('/Users/jorge/OneDrive/Desktop/Skola/5. semester/Bionano/A1 real/*.tif'))
    print(frames[0])
    frames[0]
    plt.imshow(frames[0])
    f=tp.locate(frames[0], 11,invert=True)
    f.head()
    tp.annotate(f, frames[0])
    f=tp.locate(frames[0], 11, invert=True, minmass=Min_Mass)
    tp.annotate(f, frames[0])
    ax=plt.subplot()
    ax.hist(f['mass'], bins=20)
    ax.set(xlabel='mass', ylabel='count')
    tp.subpx_bias(f)
    tp.subpx_bias(tp.locate(frames[0], 7,invert=True,minmass=Min_Mass) )
    f=tp.batch(frames, 11, minmass=Min_Mass, invert=True)
    t=tp.link(f, 5, memory=3)
    t.head()
    t1=tp.filter_stubs(t, 2)
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())
    plt.figure()
    tp.mass_size(t1.groupby('particle').mean())
    t2=t1[((t1['mass'] < 600) & (t1['size'] <3.2) & (t1['ecc'] < 0.3))]
    plt.figure()
    tp.annotate(t2[t2['frame'] == 0], frames[0])
    plt.figure()
    tp.plot_traj(t2)
    d=tp.compute_drift(t2)
    d.plot()
    plt.show()
    tm=tp.subtract_drift(t2.copy(),d)
    ax=tp.plot_traj(tm)
    plt.show()
    im=tp.imsd(tm, 0.228, 7.67) #micrometer/pixel funent ved hjelp av gimp. 200/877
    fig, ax=plt.subplots()
    ax.plot(im.index, im, 'k-', alpha=0.1)
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    em=tp.emsd(tm, 0.228, 7.67) #micrometer/pixel funnet ved hjelp av gimp. 200/877
    fig, ax=plt.subplots()
    ax.plot(em.index, em, 'o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(ylabel=r'$\langle \Delta r^2 \ rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    ax.set(ylim=(1e-2, 10))
    plt.figure()
    plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    plt.xlabel('lag time $t$')
    print(tp.utils.fit_powerlaw(em))
    
    #fordeling av D
    nim=im.values.tolist()
    new_im=nim[np.logical_not(np.isnan(nim))] #denna funker ikke
    print(new_im)
    
if __name__ == "__main__":
    main()