import numpy as np
from scipy.integrate import solve_ivp
from get_G import get_G
from input_vars import nx, ns, f

def adj_descent(L, rtol: float, atol: float, t_end: int, dt: float) -> tuple[list, list]:

    global nx, ns, f

    # Set up the time interval
    nt = int(t_end / dt) + 1  
    tspan = np.linspace(0, t_end, nt)

    # Integration: use solve_ivp with method='BDF' (stiff system solver)
    solution = solve_ivp(
        fun=lambda t, L: get_G(t, L, print_res=True),
        t_span=(0, t_end),                                      # (start_time, end_time)
        y0=L,                                                   # Initial condition
        method='BDF',                                           # 'BDF' or 'Radau' - implicit + adaptive time stepping
        t_eval=tspan,                                           # The specific time steps returned
        rtol=rtol,                                              # Relative tolerance
        atol=atol                                               # Absolute tolerance
    )

    # Extract the output list of iteration values
    u_lst = np.array(solution.y.T)
    t_lst = solution.t.T
    
    return u_lst, t_lst
