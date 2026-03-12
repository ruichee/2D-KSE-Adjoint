kx = 2*np.pi * (X / Lx)
ks = 2*np.pi * (S / Ls)
u0 = 1.5 * np.cos(ks) * np.sin(kx) + \
    0.8 * np.sin(2 * ks + 2 * kx + 0.5)
stages = (1e4,)*2