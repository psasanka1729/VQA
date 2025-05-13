import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, kron, identity
from scipy.sparse.linalg import expm
from functools import reduce

class NoisyInfiniteRanged:
     def __init__(self, epsilon, gamma_e, gamma_d, Delta, J, L, angle_bounds):
          self.gamma_e = gamma_e
          self.gamma_d = gamma_d
          self.Delta = Delta
          self.J = J
          self.L = L
          self.angle_bounds = angle_bounds

          if self.angle_bounds == None:
               self.angle_bounds = {
                    'theta_int': (-np.inf, np.inf),
                    'theta_f'  : (-np.inf, np.inf),
                    'theta_rel': (-np.inf, np.inf),
                    'theta_dep': (-np.inf, np.inf)
               }
          else:
               self.angle_bounds = angle_bounds
               
          self.I2 = csc_matrix(np.eye(2))
          self.sigma_x = csc_matrix(np.array([[0, 1], [1, 0]]))
          self.sigma_y = csc_matrix(np.array([[0, -1j], [1j, 0]]))
          self.sigma_z = csc_matrix(np.array([[1, 0], [0, -1]]))
          self.sigma_plus = (self.sigma_x + 1j * self.sigma_y).toarray() / 2
          self.sigma_minus = (self.sigma_x - 1j * self.sigma_y).toarray() / 2

          self.gamma_depol = epsilon
          self.depolarizing_noise = expm(self.L_depol_super().toarray())

     def vectorize_rho(self, rho):
          if hasattr(rho, "toarray"):
               rho = rho.toarray()
          rho_vec = rho.flatten(order="F")
          return csc_matrix(rho_vec).T

     def unvectorize_rho(self, rho_vec):
          if hasattr(rho_vec, "toarray"):
               rho = rho_vec.toarray().reshape((2, 2), order="F")
          else:
               rho = rho_vec.reshape((2, 2), order="F")
          return csc_matrix(rho)

     def L_depol_super(self):
          X, Y, Z = self.sigma_x, self.sigma_y, self.sigma_z
          I = self.I2
          return self.gamma_depol * ((1/3) * (kron(X.conj(), X) + kron(Y.conj(), Y) + kron(Z.conj(), Z)) - kron(I, I))

     def L_int(self, x):
          return -1j * self.J * self.L * x * (kron(self.I2, self.sigma_x) - kron(self.sigma_x, self.I2))

     def L_f(self):
          return -1j * (kron(self.I2, -self.Delta * self.sigma_z) - kron(-self.Delta * self.sigma_z, self.I2))

     def L_rel(self):
          return self.gamma_e * (kron(np.conjugate(self.sigma_minus), self.sigma_minus)
               - 0.5 * kron(self.I2, np.conjugate(self.sigma_minus).T @ self.sigma_minus)
               - 0.5 * kron((np.conjugate(self.sigma_minus).T @ self.sigma_minus).T, self.I2))

     def L_dep(self):
          return self.gamma_d * (kron(np.conjugate((self.I2 + self.sigma_z) / 2), (self.I2 + self.sigma_z) / 2)
               - 0.5 * kron(self.I2, np.conjugate((self.I2 + self.sigma_z) / 2).T @ (self.I2 + self.sigma_z) / 2)
               - 0.5 * kron((np.conjugate((self.I2 + self.sigma_z) / 2).T @ (self.I2 + self.sigma_z) / 2).T, self.I2))

     def transform_angle(self, unbounded_theta, angle_type):
          r"""
          Transform unbounded parameter to bounded angle using tanh
          Parameters
          ----------
          unbounded_theta : float
               Unbounded parameter to be transformed.
          angle_type : str
               Type of angle to be transformed ('theta_int', 'theta_f', 'theta_rel', 'theta_dep').
          Returns
          -------
          float
               Transformed angle within the specified bounds.
          """
          lower, upper = self.angle_bounds[angle_type]
          if lower == -np.inf and upper == np.inf:
               return unbounded_theta
          range_size = upper - lower
          midpoint = (upper + lower) / 2
          return midpoint + range_size/2 * np.tanh(unbounded_theta)

     def get_gradient_factor(self, unbounded_theta, angle_type):
          r"""
          Calculate the gradient factor for backpropagation with constrained parameters
          Parameters
          ----------
          unbounded_theta : float
               Unbounded parameter to be transformed.
          angle_type : str
               Type of angle to be transformed ('theta_int', 'theta_f', 'theta_rel', 'theta_dep').
          Returns
          -------
          float
               Gradient factor for the specified angle type.
          """
          lower, upper = self.angle_bounds[angle_type]
          if lower == -np.inf and upper == np.inf:
               return 1.0
          range_size = upper - lower
          return range_size/2 * (1 - np.tanh(unbounded_theta)**2)

     def transform_angles_list(self, unbounded_angles_lst):
          transformed_angles = []
          for layer in unbounded_angles_lst:
               theta_int = self.transform_angle(layer[0], 'theta_int')
               theta_f = self.transform_angle(layer[1], 'theta_f')
               theta_rel = self.transform_angle(layer[2], 'theta_rel')
               theta_dep = self.transform_angle(layer[3], 'theta_dep')
               transformed_angles.append([theta_int, theta_f, theta_rel, theta_dep])
          return transformed_angles

     def build_mean_field_hamiltonian(self, x_mean):
          return self.J * self.L * x_mean * self.sigma_x - self.Delta * self.sigma_z

     def jump_operators(self):
          L1 = np.sqrt(self.gamma_e) * (self.sigma_x - 1j * self.sigma_y) / 2
          L2 = np.sqrt(self.gamma_d) * (self.I2 + self.sigma_z) / 2
          return L1, L2

     def lindbladian(self, rho, x_mean):
          H = self.build_mean_field_hamiltonian(x_mean)
          L1, L2 = self.jump_operators()
          comm = -1j * (H @ rho - rho @ H)
          dis1 = L1 @ rho @ L1.conj().T - 0.5 * (L1.conj().T @ L1 @ rho + rho @ L1.conj().T @ L1)
          dis2 = L2 @ rho @ L2.conj().T - 0.5 * (L2.conj().T @ L2 @ rho + rho @ L2.conj().T @ L2)
          return comm + dis1 + dis2

     def cost_function(self, rho, x_mean):
          rho = csr_matrix(rho)
          L_rho = self.lindbladian(rho, x_mean)
          return np.real(L_rho.getH().dot(L_rho).diagonal().sum())
          
     # Reparameterization functions
     def transform_angle(self, unbounded_theta, angle_type):
          """Transform unbounded parameter to bounded angle using tanh"""
          lower, upper = self.angle_bounds[angle_type]
          
          # If no bounds or effectively infinite bounds, return the angle directly
          if lower == -np.inf and upper == np.inf:
               return unbounded_theta
          
          # Apply tanh transformation to map from (-inf, inf) to (lower, upper)
          range_size = upper - lower
          midpoint = (upper + lower) / 2
          return midpoint + range_size/2 * np.tanh(unbounded_theta)
          
     def transform_angles_list(self, unbounded_angles_lst):
          """Transform a list of unbounded parameters to bounded angles"""
          transformed_angles = []
          
          for layer in unbounded_angles_lst:
               theta_int = self.transform_angle(layer[0], 'theta_int')
               theta_f = self.transform_angle(layer[1], 'theta_f')
               theta_rel = self.transform_angle(layer[2], 'theta_rel')
               theta_dep = self.transform_angle(layer[3], 'theta_dep')
               
               transformed_angles.append([theta_int, theta_f, theta_rel, theta_dep])
               
          return transformed_angles
     
     def get_gradient_factor(self, unbounded_theta, angle_type):
          """Calculate the gradient factor for backpropagation with constrained parameters"""
          lower, upper = self.angle_bounds[angle_type]
          
          # If no bounds, gradient factor is 1 (no adjustment)
          if lower == -np.inf and upper == np.inf:
               return 1.0
          
          # Calculate derivative of tanh transformation
          range_size = upper - lower
          return range_size/2 * (1 - np.tanh(unbounded_theta)**2)

     # Variational ansatz for the system with depolarizing noise acting before and after each gate
     def variational_ansatz(self, number_of_layers, unbounded_angles_lst, rho_initial_vec):
          """Apply variational ansatz with transformed (bounded) angles"""
          # Transform unbounded parameters to angles within bounds
          angles_lst = self.transform_angles_list(unbounded_angles_lst)

          rho_vec = rho_initial_vec

          # x is calculated from the initial density matrix
          x = np.real((self.sigma_x @ self.unvectorize_rho(rho_vec)).diagonal().sum())

          # List to store the E_l matrices for each layer
          E_l_lst = []

          for i in range(number_of_layers):
               theta_int, theta_f, theta_rel, theta_dep = angles_lst[i]

               E_l = (expm(theta_dep * self.L_dep().toarray()) @ self.depolarizing_noise @
               expm(theta_rel * self.L_rel().toarray()) @ self.depolarizing_noise @               
               expm(theta_f * self.L_f().toarray()) @ self.depolarizing_noise @ 
               expm(theta_int * self.L_int(x).toarray()) @ self.depolarizing_noise)
               E_l_lst.append(E_l)

               rho_vec = ( E_l @ rho_vec)

               # x is updated after each layer
               x = np.real((self.sigma_x @ self.unvectorize_rho(rho_vec)).diagonal().sum())

          # E_l_lst = [E^1, E^(2), ..., E^L]
          return self.vectorize_rho(rho_vec), E_l_lst

     def solution_set(self):
          r""" 
          Returns the exact solution set of the system.
          """
          sqrt_term = np.sqrt(self.gamma_e * (self.gamma_e + self.gamma_d) / (8 * self.J * self.L * self.Delta))
          correction_term = 1 - (16 * self.Delta**2 + (self.gamma_e + self.gamma_d)**2) / (16 * self.J * self.L * self.Delta)

          if correction_term < 0:
               x_pos = 0
               x_neg = 0
               y_pos = 0
               y_neg = 0
               z = -1
          else:
               x_pos = (4 * self.Delta / (self.gamma_e + self.gamma_d)) * sqrt_term * np.sqrt(correction_term)
               x_neg = -x_pos
               y_pos = sqrt_term * np.sqrt(correction_term)
               y_neg = -y_pos
               z = (-16 * self.Delta**2 - (self.gamma_e + self.gamma_d)**2) / (16 * self.J * self.L * self.Delta)
          return [(x_pos, y_pos, z), (x_neg, y_neg, z)]

     def gradient_descent_optimizer(self, rho_initial_vec, number_of_layers, unbounded_angles_lst, learning_rate, max_iterations, tolerance=1e-5):
          r"""
          Parameters
          ----------
          rho_initial : array-like
               Initial density matrix vectorized.
          number_of_layers : int
               Number of layers in the variational ansatz.
          unbounded_angles_lst : list of lists
               List of unbounded parameters for each layer, to be transformed into angles.
          learning_rate : float
               Learning rate for the gradient descent optimization.
          max_iterations : int
               Maximum number of iterations for the optimization.
          tolerance : float, optional
               Tolerance for convergence, default is 1e-5.
          Returns
          -------
          best_angles_lst : the best angles found during optimization (in bounded form)
          best_unbounded_angles_lst : the best unbounded parameters found during optimization
          best_cost : the best cost function value found
          x_values_lst : list of x values at each iteration
          rho_vec_lst : list of density matrix vectors at each iteration
          cost_function_lst : list of cost function values at each iteration
          -------
          This function performs gradient descent optimization to minimize the cost function of the system.
          """

          # Store the initial density matrix vector
          rho_vec_lst = [rho_initial_vec]

          # Calculate initial x, y, z values from the initial density matrix
          self.x_initial = np.real((self.sigma_x @ self.unvectorize_rho(rho_initial_vec)).diagonal().sum())
          self.y_initial = np.real((self.sigma_y @ self.unvectorize_rho(rho_initial_vec)).diagonal().sum())
          self.z_initial = np.real((self.sigma_z @ self.unvectorize_rho(rho_initial_vec)).diagonal().sum())

          # Get bounded angles for cost function evaluation and other calculations
          angles_lst = self.transform_angles_list(unbounded_angles_lst)

          cost_function_value = self.cost_function(self.unvectorize_rho(rho_initial_vec), self.x_initial)
          cost_function_lst = [cost_function_value]
          x_values_lst = [self.x_initial]
          best_unbounded_angles_lst = [list(a) for a in unbounded_angles_lst]
          best_cost = float('inf')

          print(f"Initial cost function value={cost_function_value:.10f}")
          print(f"Initial unbounded parameters: {unbounded_angles_lst}")
          print(f"Initial bounded angles: {angles_lst}")
          print(f"Angle bounds: {self.angle_bounds}")

          # Store the gradient values in a multidimensional array
          gradients_lst = []

          for iteration in range(max_iterations):
               current_learning_rate = learning_rate

               if cost_function_value < best_cost:
                    best_cost = cost_function_value
                    best_unbounded_angles_lst = [list(a) for a in unbounded_angles_lst]

               if iteration > 0 and abs(cost_function_lst[-1]) < tolerance:
                    break

               # Calculate gradients for all layers but only store the last layer's gradients
               gradients = [[0.0 for _ in range(4)] for _ in range(number_of_layers)]  # Need full array for optimization
               final_layer_gradients = [[0.0 for _ in range(4)]]  # This is what we'll store

               # Calculation of rho_final and E_l_lst after applying the variational ansatz
               # This is fixed for a given iteration
               rho_final_vec, E_l_lst = self.variational_ansatz(number_of_layers, unbounded_angles_lst, rho_initial_vec)

               # Save the density matrix at this step
               rho_vec_lst.append(rho_final_vec)

               # Calculate the value of x at this iteration
               x_val = np.real((self.sigma_x @ self.unvectorize_rho(rho_final_vec)).diagonal().sum())
               x_values_lst.append(x_val)                                  

               # Calculate the cost function value for the current iteration
               cost_function_value = np.real(self.cost_function(self.unvectorize_rho(rho_final_vec), x_val))
               cost_function_lst.append(cost_function_value)

               # Get current bounded angles for display
               angles_lst = self.transform_angles_list(unbounded_angles_lst)

               print(f"gamma = {self.gamma_e}, epsilon = {self.gamma_depol}, Delta = {self.Delta}")
               print(f"Iteration {iteration}:")
               print(f"  Cost function value = {cost_function_value:.10f}")
               print(f"  Current unbounded parameters:")
               for layer_idx, params in enumerate(unbounded_angles_lst):
                    print(f"    Layer {layer_idx + 1}: {params}")
               print(f"  Current bounded angles:")
               for layer_idx, angles in enumerate(angles_lst):
                    print(f"    Layer {layer_idx + 1}: {angles}")

               r""" 
                    The final density matrix after applying the variational ansatz is given by:

                    rho^L_f = E^L  @ E^(L-1) @ ... @ E^1 @ rho_0

                    where rho_0 is the initial density matrix and 

                    E^k = exp(theta_4 * L_dep) @
                          exp(theta_3 * L_rel) @
                          exp(theta_2 * L_f) @
                          exp(theta_1 * L_H)
               
               """

               # Calculate gradients for all layers
               for layer in range(number_of_layers):
                    # Get the transformed bounded angles for each layer
                    theta_int = self.transform_angle(unbounded_angles_lst[layer][0], 'theta_int')
                    theta_f   = self.transform_angle(unbounded_angles_lst[layer][1], 'theta_f')
                    theta_rel = self.transform_angle(unbounded_angles_lst[layer][2], 'theta_rel')
                    theta_dep = self.transform_angle(unbounded_angles_lst[layer][3], 'theta_dep')

                    # For the first layer the rho_initial_vec is the initial density matrix
                    if layer == 0:
                         rho_after_layer_l = rho_initial_vec
                    else:
                         # Multiply E^{(layer)} @ E^{(layer-1)} @ … @ E^{(1)} @ rho_initial_vec
                         # E_l_lst = [E^1, ..... , E^(L-1), E^L]
                         E_l_lst_till_layer_l = E_l_lst[:layer]
                         # Initializing the matrix for the chain as identity matrix
                         E_l_lst_till_layer_l_matrix = kron(self.I2, self.I2)
                         for E_l in E_l_lst_till_layer_l:
                              E_l_lst_till_layer_l_matrix = E_l @ E_l_lst_till_layer_l_matrix

                         rho_after_layer_l = E_l_lst_till_layer_l_matrix @ rho_initial_vec
                         
                    # Calculating x_l, y_l, z_l
                    x_l = np.real((self.sigma_x @ self.unvectorize_rho(rho_after_layer_l)).diagonal().sum())
                    y_l = np.real((self.sigma_y @ self.unvectorize_rho(rho_after_layer_l)).diagonal().sum())
                    z_l = np.real((self.sigma_z @ self.unvectorize_rho(rho_after_layer_l)).diagonal().sum())

                    # Derivative of E^L with respect to theta^l_1
                    dE_L_f_dtheta_1 = self.J * self.L * x_l * ( y_l * (kron(self.sigma_z, self.I2) - 
                    kron(self.I2, self.sigma_z)) + z_l * (kron(self.sigma_x, self.I2) - kron(self.I2, self.sigma_x)))
                    dE_L_f_dtheta_1 = self.depolarizing_noise @ dE_L_f_dtheta_1

                    # Rest of the layers are unchanged
                    dE_L_f_dtheta_1 = self.depolarizing_noise @ expm((theta_f * self.L_f()).toarray())   @ dE_L_f_dtheta_1
                    dE_L_f_dtheta_1 = self.depolarizing_noise @ expm((theta_rel * self.L_rel()).toarray()) @ dE_L_f_dtheta_1
                    dE_L_f_dtheta_1 = self.depolarizing_noise @ expm((theta_dep * self.L_dep()).toarray()) @ dE_L_f_dtheta_1


                    r""" Constructing the chain E^L @ ..... @ E^(l+1) @ (dE^l/dtheta^l_1) @ E^(l-1) @ ..... @ E^1 """
                    E_l_lst_copy = E_l_lst.copy()
                    E_l_lst_copy[layer] = dE_L_f_dtheta_1
                    # Calculating the product of all the matrices in the chain
                    # to calculate drho^L_f/dtheta^l_1
                    dE_L_f_dtheta_1 = reduce(lambda x, y: x @ y, E_l_lst_copy)

                    # Calculation of the derivative of rho^L_f with respect to theta^l_1
                    # First term inside the Trace
                    Lrhof = self.lindbladian(self.unvectorize_rho(rho_final_vec), x_l)
                    # Second term inside the Trace
                    Ld_rhof_1 = self.lindbladian(self.unvectorize_rho(dE_L_f_dtheta_1 @ rho_initial_vec), x_l)
                    
                    # Apply chain rule for gradient with respect to unbounded parameter
                    gradient_factor = self.get_gradient_factor(unbounded_angles_lst[layer][0], 'theta_int')
                    gradients[layer][0] = 2 * np.real((Lrhof.getH().dot(Ld_rhof_1)).diagonal().sum()) * gradient_factor

                    r"""
                         Evaluate the gradients with respect to theta_l_f, theta_l_r, and theta_l_d
                    """
                    # Calculate the derivative with respect to theta_l_f
                    dE_L_dtheta_f = (self.depolarizing_noise @ expm((theta_dep * self.L_dep()).toarray())
                                   @ self.depolarizing_noise @ expm((theta_rel * self.L_rel()).toarray())
                                   @ self.depolarizing_noise @ expm((theta_f * self.L_f()).toarray())
                                   @ (self.L_f().toarray()) @ self.depolarizing_noise @
                                   expm(theta_int * self.L_int(x_l).toarray()))

                    # Calculation of derivative of rho^L_f with respect to theta^l_f
                    E_l_lst_copy = E_l_lst.copy()
                    E_l_lst_copy[layer] = dE_L_dtheta_f
                    # Calculating the product of all the E^k matrices
                    dE_L_dtheta_f = reduce(lambda x, y: x @ y, E_l_lst_copy)

                    # Calculate the derivative with respect to theta_l_r
                    dE_L_dtheta_r = (self.depolarizing_noise @ expm((theta_dep * self.L_dep()).toarray())
                                   @ self.depolarizing_noise @ expm((theta_rel * self.L_rel()).toarray())
                                   @ (self.L_rel().toarray()) @ self.depolarizing_noise
                                   @ expm((theta_f * self.L_f()).toarray()) @ self.depolarizing_noise @
                                    expm(theta_int * self.L_int(x_l).toarray()))

                    # Calculation of derivative of rho^L_f with respect to theta^l_r
                    E_l_lst_copy = E_l_lst.copy()
                    E_l_lst_copy[layer] = dE_L_dtheta_r
                    # Calculating the product of all the E^k matrices
                    dE_L_dtheta_r = reduce(lambda x, y: x @ y, E_l_lst_copy)

                    # Calculate the derivative with respect to theta_l_d
                    dE_L_dtheta_d = (self.depolarizing_noise @ expm((theta_dep * self.L_dep()).toarray())
                                   @ (self.L_dep().toarray()) @ self.depolarizing_noise
                                   @ expm((theta_rel * self.L_rel()).toarray()) @ self.depolarizing_noise
                                   @ expm((theta_f * self.L_f()).toarray()) @ self.depolarizing_noise @
                                   expm(theta_int * self.L_int(x_l).toarray()))
                    
                    # Calculation of derivative of rho^L_f with respect to theta^l_d
                    E_l_lst_copy = E_l_lst.copy()
                    E_l_lst_copy[layer] = dE_L_dtheta_d
                    # Calculating the product of all the E^k matrices
                    dE_L_dtheta_d = reduce(lambda x, y: x @ y, E_l_lst_copy)

                    # Second term inside the Trace
                    Ld_rhof_f = self.lindbladian(self.unvectorize_rho(dE_L_dtheta_f @ rho_initial_vec), x_l)
                    Ld_rhof_r = self.lindbladian(self.unvectorize_rho(dE_L_dtheta_r @ rho_initial_vec), x_l)
                    Ld_rhof_d = self.lindbladian(self.unvectorize_rho(dE_L_dtheta_d @ rho_initial_vec), x_l)

                    # Calculate gradients for theta_l_f, theta_l_r, and theta_l_d with chain rule factors
                    gradient_factor_f = self.get_gradient_factor(unbounded_angles_lst[layer][1], 'theta_f')
                    gradient_factor_r = self.get_gradient_factor(unbounded_angles_lst[layer][2], 'theta_rel')
                    gradient_factor_d = self.get_gradient_factor(unbounded_angles_lst[layer][3], 'theta_dep')
                    
                    gradients[layer][1] = 2 * np.real((Lrhof.getH().dot(Ld_rhof_f)).diagonal().sum()) * gradient_factor_f
                    gradients[layer][2] = 2 * np.real((Lrhof.getH().dot(Ld_rhof_r)).diagonal().sum()) * gradient_factor_r
                    gradients[layer][3] = 2 * np.real((Lrhof.getH().dot(Ld_rhof_d)).diagonal().sum()) * gradient_factor_d

                    # Update all parameters using their respective gradients
                    for p in range(4):
                         unbounded_angles_lst[layer][p] -= current_learning_rate * gradients[layer][p]
                         # If this is the last layer, store its gradients for return
                         if layer == number_of_layers - 1:
                              final_layer_gradients[0][p] = gradients[layer][p]

               # Only append the gradients from the last layer to gradients_lst
               gradients_lst.append(final_layer_gradients)

          # Return the bounded angles for the best solution found
          best_angles_lst = self.transform_angles_list(best_unbounded_angles_lst)
          
          return best_angles_lst, best_unbounded_angles_lst, best_cost, x_values_lst, rho_vec_lst, cost_function_lst, gradients_lst