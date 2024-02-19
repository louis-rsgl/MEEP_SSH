from meep.materials import SiO2
import numpy as np
import matplotlib.pyplot as plt

wvl_min = 0.4 # units of μm
wvl_max = 1 # units of μm
nwvls = 21
wvls = np.linspace(wvl_min, wvl_max, nwvls)

SiO2_epsilon = np.array([SiO2.epsilon(1/w)[0][0] for w in wvls])

plt.subplot(1,2,1)
plt.plot(wvls,np.real(SiO2_epsilon),'bo-')
plt.xlabel('wavelength (μm)')
plt.ylabel('real(ε)')

plt.subplot(1,2,2)
plt.plot(wvls,np.imag(SiO2_epsilon),'ro-')
plt.xlabel('wavelength (μm)')
plt.ylabel('imag(ε)')

plt.suptitle('SiO$_2$ from Meep materials library')
plt.subplots_adjust(wspace=0.4)
plt.show()