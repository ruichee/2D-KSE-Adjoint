'''from io import StringIO

    # 1. Read the raw text and strip the parentheses
    with open(r'2D_KS_adj_fixedpoints\u_k_save1.dat', 'r') as file:
        raw_text = file.read()
        clean_text = raw_text.replace('(', '').replace(')', '')

    # 2. Trick NumPy into reading the cleaned string as if it were a file
    u1 = np.loadtxt(StringIO(clean_text), dtype=complex)

    # 1. Read the raw text and strip the parentheses
    with open(r'2D_KS_adj_fixedpoints\u_k_save.dat', 'r') as file:
        raw_text = file.read()
        clean_text = raw_text.replace('(', '').replace(')', '')

    # 2. Trick NumPy into reading the cleaned string as if it were a file
    u2 = np.loadtxt(StringIO(clean_text), dtype=complex)

    print(np.round(np.fft.fft2(u1) - np.fft.fft2(u2)[::-1], 2))

    print([i for i in np.round(np.fft.fft2(u1), 2).flatten() if i !=0])
    print([i for i in np.round(np.fft.fft2(u2)[::-1], 2).flatten() if i !=0])

    # check fourier values
    func = lambda x,y: np.round(np.abs(np.fft.fft2(u1)[-x,y]), 2)
    print("\nFourier Coefficients")
    print("\t", func(1, 0), func(1, 1), func(0, 1), "\n")
    print(f"\t e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"\t {func(0, 2)} {func(1, 2)} {func(0, 3)} {func(1, 3)} {func(2, 0)} {func(2, 1)} {func(2, 2)} \n")
    print(f"\t e(0,3): {func(1, -1)}, e(1,3): {func(2, 4)}")
    print()

    func = lambda x,y: np.round(np.abs(np.fft.fft2(u2)[x,y]), 2)
    print("\nFourier Coefficients")
    print("\t", func(1, 0), func(1, 1), func(0, 1), "\n")
    print(f"\t e(2,0) e(2,1) e(3,0) e(3,1) e(0,2) e(1,2) e(2,2)")
    print(f"\t {func(0, 2)} {func(1, 2)} {func(0, 3)} {func(1, 3)} {func(2, 0)} {func(2, 1)} {func(2, 2)} \n")
    print(f"\t e(0,3): {func(1, -1)}, e(1,3): {func(2, 4)}") '''