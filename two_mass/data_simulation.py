import scipy.signal as sig
import numpy as np


def simulate_two_mass(initial_conditions,  end_time, steps, mass1, mass2, stiffness_secondary, damping_secondary):

    stiffness_primary = 1e6 * 4  # Stiffness of primary suspension
    damping_primary = 1e4 * 4    # Damping of primary suspension

    time_vector = np.linspace(0, end_time, steps)

    # Initialize input vectors
    input_zero = np.zeros_like(time_vector)
    input_derivative_zero = np.zeros_like(input_zero)

    # State-space representation matrices
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-stiffness_secondary / mass2, -damping_secondary / mass2, stiffness_secondary / mass2, damping_secondary / mass2],
        [0.0, 0.0, 0.0, 1.0],
        [stiffness_secondary / mass1, damping_secondary / mass1, -(stiffness_secondary + stiffness_primary) / mass1, -(damping_secondary + damping_primary) / mass1]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [stiffness_primary / mass1, damping_primary / mass1]
    ])

    C = np.eye(4)
    D = np.zeros((4, 2))

    # Create state-space system
    two_mass_system = sig.StateSpace(A, B, C, D)
    system_input = np.hstack((input_zero, input_derivative_zero))

    # Simulate the system response
    time_simulation, output_all, state_simulation = sig.lsim(two_mass_system, system_input, np.squeeze(time_vector), X0=initial_conditions)

    # Return results
    parameters = [mass2, mass1, stiffness_primary, damping_primary, stiffness_secondary, damping_secondary]
    return time_simulation, output_all, state_simulation, parameters
    
def get_data(y_orig_all, r_debug, exp_len, time_step):

      y_m2 = np.expand_dims(y_orig_all[:exp_len,0], 1) # careful m2 is the upper mass!
      y_m2_dx = np.expand_dims(y_orig_all[:exp_len, 1], 1)
      y_m1 = np.expand_dims(y_orig_all[:exp_len, 2], 1)
      y_m1_dx = np.expand_dims(y_orig_all[:exp_len, 3], 1)

      u = np.zeros_like(y_m2)[:exp_len]
      up = np.zeros_like(u)[:exp_len]
      if r_debug:
          y_m2_dx2 = np.expand_dims(np.gradient(y_orig_all[:exp_len, 1]) / time_step,1)
          y_m1_dx2 = np.expand_dims(np.gradient(y_orig_all[:exp_len, 3]) / time_step,1)

          return y_m2, y_m2_dx, y_m2_dx2, y_m1, y_m1_dx, y_m1_dx2, u, up
      else:
          return y_m2, None, None, y_m1, None, None, u, up

def get_simulated_data_two_mass(start_vector, end_time=20, steps=4001, exp_len=400,m1=15000, m2=15000, css=0.5e6 * 2,dss=1.5e4 * 2, debug_data=True):
    tsim_nom_orig, y_orig_all, xsim_nom_orig, simul_const = simulate_two_mass(start_vector, end_time, steps,m1, m2, css, dss)
    y_m2_out, y_m2_dx_out, y_m2_dx2_out, y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out = get_data(y_orig_all, debug_data, exp_len=exp_len, time_step=end_time/steps)
    return [y_m2_out, y_m2_dx_out, y_m2_dx2_out, y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out, np.expand_dims(tsim_nom_orig[:exp_len], 1)], simul_const








