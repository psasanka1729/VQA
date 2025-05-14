import numpy as np
from scipy.sparse import csc_matrix, identity, kron, csr_matrix
from scipy.sparse.linalg import expm
from functools import reduce

class TFIM_HVA_Ansatz:
     def __init__(self, L, initial_density_matrix, number_of_layers, g, epsilon, gamma_e, gamma_d):
          self.L = L
          self.number_of_layers = number_of_layers
          self.g = g
          self.gamma_e = gamma_e
          self.gamma_d = gamma_d
          self.epsilon = epsilon
          
          self.I2 = csc_matrix(np.eye(2, dtype=np.complex128))
          self.sigma_x = csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
          self.sigma_y = csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
          self.sigma_z = csc_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
          
          self.sigma_plus_arr = (self.sigma_x.toarray() + 1j * self.sigma_y.toarray()) / 2.0
          self.sigma_minus_arr = (self.sigma_x.toarray() - 1j * self.sigma_y.toarray()) / 2.0
          
          self.rho_initial_vec = self.vectorize_rho(initial_density_matrix)
          
          # HIGHLIGHT: Correct initialization of depolarizing_noise_super
          l_depol_generator = self.L_depol_super()
          # Ensure the generator is not empty and expm gets a dense array
          if l_depol_generator.shape[0] > 0 and l_depol_generator.shape[1] > 0:
            self.depolarizing_noise_super = expm(l_depol_generator.toarray())
          else: 
            # Default to identity if L=0 or generator is somehow empty
            dim_super_sq = (2**self.L)**2
            if dim_super_sq == 0 : dim_super_sq = 1 # for L=0, super_dim is 1
            self.depolarizing_noise_super = identity(dim_super_sq, format="csc", dtype=np.complex128)


     def I(self, n):
          return identity(n, format="csc", dtype=np.complex128)

     def vectorize_rho(self, rho_matrix):
          if hasattr(rho_matrix, "toarray"):
              temp_arr = rho_matrix.toarray()
              rho_arr = np.atleast_2d(temp_arr) if temp_arr.ndim < 2 else temp_arr
          elif isinstance(rho_matrix, np.ndarray):
              rho_arr = np.atleast_2d(rho_matrix) if rho_matrix.ndim < 2 else rho_matrix
          else:
              rho_arr = np.asarray(rho_matrix, dtype=np.complex128)
              if rho_arr.ndim < 2: rho_arr = np.atleast_2d(rho_arr)
          
          rho_vec = rho_arr.flatten(order="F") 
          
          if rho_vec.size == 0: 
              return csc_matrix((0,1), dtype=rho_vec.dtype) 
          
          rho_vec_column = rho_vec.reshape(-1, 1) 
          return csc_matrix(rho_vec_column)

     def unvectorize_rho(self, rho_vec):
          dim_super_flat = rho_vec.shape[0]
          if dim_super_flat == 0: 
              return csc_matrix((0,0), dtype=rho_vec.dtype)
          dim_hilbert = int(np.sqrt(dim_super_flat))
          
          if hasattr(rho_vec, "toarray"):
               rho_arr = rho_vec.toarray().reshape((dim_hilbert, dim_hilbert), order="F") 
          else:
               rho_arr = rho_vec.reshape((dim_hilbert, dim_hilbert), order="F")
          return csc_matrix(rho_arr)
     
     def op_on_qubit(self, qubit_idx, single_qubit_operator):
          if self.L == 0: 
              if single_qubit_operator.shape == (1,1): return single_qubit_operator 
              return csc_matrix(np.array([[1.0]]), dtype=np.complex128)

          op_list = [self.I2] * self.L 
          op_list[qubit_idx] = single_qubit_operator
          
          if not op_list: 
              return self.I(2**self.L)

          if len(op_list) == 1: 
              return op_list[0]
          return reduce(lambda a, b: kron(a, b, format="csc"), op_list)

     def L_depol_super(self):
          dim_super = (2**self.L)**2 
          depol_super_total = csc_matrix((dim_super, dim_super), dtype=np.complex128)
          if self.L == 0: return depol_super_total # Return empty sparse for L=0

          I_full_system = self.I(2**self.L) 

          for q in range(self.L):
               X_q = self.op_on_qubit(q, self.sigma_x)
               Y_q = self.op_on_qubit(q, self.sigma_y)
               Z_q = self.op_on_qubit(q, self.sigma_z)
               
               term_pauli_sum_q = (kron(X_q.conj(), X_q, format="csc") + 
                                   kron(Y_q.conj(), Y_q, format="csc") + 
                                   kron(Z_q.conj(), Z_q, format="csc")) / 3.0
               
               term_identity_q = kron(I_full_system, I_full_system, format="csc")
               
               depol_super_total += self.epsilon * (term_pauli_sum_q - term_identity_q) 
               
          return depol_super_total

     def H_ZZ(self):
          H_ZZ_op = csc_matrix((2**self.L, 2**self.L), dtype=np.complex128)
          if self.L < 2: return H_ZZ_op # No ZZ for L=0 or L=1

          for i in range(self.L - 1): 
               ops = [self.I2] * self.L; ops[i] = self.sigma_z; ops[i+1] = self.sigma_z
               H_ZZ_op += reduce(kron, ops)
          if self.L > 1: 
            ops_pb = [self.I2] * self.L; ops_pb[self.L-1] = self.sigma_z; ops_pb[0] = self.sigma_z
            H_ZZ_op += reduce(kron, ops_pb)
          return -H_ZZ_op

     def H_X(self):
          H_field_op = csc_matrix((2**self.L, 2**self.L), dtype=np.complex128)
          if self.L == 0: return H_field_op
          for i in range(self.L):
               op_list = [self.I2] * self.L; op_list[i] = self.sigma_x
               H_field_op += reduce(kron, op_list) if self.L > 1 else op_list[0]
          return -self.g * H_field_op

     def TFIM_hamiltonian(self):
          return self.H_ZZ() + self.H_X()

     def L_unitary_super(self, H_operator):
          dim_hilbert = 2**self.L
          I_full_hilbert = self.I(dim_hilbert)
          return -1j * (
               kron(I_full_hilbert, H_operator, format="csc") - 
               kron(H_operator.T, I_full_hilbert, format="csc") 
          )

     def L_ZZ_super(self): return self.L_unitary_super(self.H_ZZ())
     def L_X_super(self): return self.L_unitary_super(self.H_X())

     def L_jump_super(self, jump_operator_L):
          dim_hilbert = 2**self.L
          I_hilbert = self.I(dim_hilbert)
          L_dag_L = jump_operator_L.conj().T @ jump_operator_L
          return (kron(jump_operator_L.conj(), jump_operator_L, format="csc") -
                  0.5 * kron(I_hilbert, L_dag_L, format="csc") -
                  0.5 * kron(L_dag_L.T, I_hilbert, format="csc"))

     def L_rel(self, qubit_idx):
          if self.gamma_e == 0: return csc_matrix(((2**self.L), (2**self.L)), dtype=np.complex128)
          return np.sqrt(self.gamma_e) * self.op_on_qubit(qubit_idx, csc_matrix(self.sigma_minus_arr))

     def L_dep(self, qubit_idx):
          if self.gamma_d == 0: return csc_matrix(((2**self.L), (2**self.L)), dtype=np.complex128)
          op_q = (self.I2 + self.sigma_z) / 2.0
          return np.sqrt(self.gamma_d) * self.op_on_qubit(qubit_idx, op_q)

     def L_rel_total_super(self):
          total_L_s = csc_matrix(((2**self.L)**2, (2**self.L)**2), dtype=np.complex128)
          if self.L == 0 or self.gamma_e == 0: return total_L_s
          for i in range(self.L):
               total_L_s += self.L_jump_super(self.L_rel(i))
          return total_L_s

     def L_dep_total_super(self):
          total_L_s = csc_matrix(((2**self.L)**2, (2**self.L)**2), dtype=np.complex128)
          if self.L == 0 or self.gamma_d == 0: return total_L_s
          for i in range(self.L):
               total_L_s += self.L_jump_super(self.L_dep(i))
          return total_L_s
     
     def U_0_super(self): return self.L_X_super()      
     def U_1_super(self): return self.L_ZZ_super()     
     def U_2_super(self): return self.L_rel_total_super() 
     def U_3_super(self): return self.L_dep_total_super() 
     
     def _get_layer_superoperator(self, beta, gamma, theta, phi):
          # HIGHLIGHT: Using self.depolarizing_noise_super (exponentiated channel)
          D_channel = self.depolarizing_noise_super 
          U0s, U1s, U2s, U3s = self.U_0_super(), self.U_1_super(), self.U_2_super(), self.U_3_super()

          # Ensure expm gets dense arrays
          return (D_channel @
                  expm((theta * U3s).toarray()) @ D_channel @
                  expm((phi   * U2s).toarray()) @ D_channel @
                  expm((beta  * U1s).toarray()) @ D_channel @
                  expm((gamma * U0s).toarray()) @ D_channel)

     def W_l_super(self, angles_lst, l_target_exclusive):
          # Evolution operator from layer 0 up to (l_target_exclusive - 1)
          # W_s = E_{l_target_exclusive-1} @ ... @ E_0
          W_s = self.I( (2**self.L)**2 )
          if self.L == 0: return W_s
          for l_idx in range(l_target_exclusive):
               beta_l, gamma_l, theta_l, phi_l = angles_lst[l_idx]
               E_l_idx = self._get_layer_superoperator(beta_l, gamma_l, theta_l, phi_l)
               W_s = E_l_idx @ W_s 
          return W_s
     
     def rho_l_vec(self, angles_lst, l_target_exclusive):
          # rho_{l_target_exclusive-1} = W_l_super @ rho_initial
          W_current_s = self.W_l_super(angles_lst, l_target_exclusive)
          return W_current_s @ self.rho_initial_vec
     
     def V_l_super(self, angles_lst, l_start_inclusive):
          # Evolution operator from layer l_start_inclusive up to N_layers-1
          # V_s = E_{N_layers-1} @ ... @ E_{l_start_inclusive}
          V_s = self.I( (2**self.L)**2 )
          if self.L == 0: return V_s
          # Build from right to left: E_{N-1} @ ( ... @ (E_{l_start} @ I) )
          for l_idx in range(self.number_of_layers - 1, l_start_inclusive - 1, -1):
              beta_l, gamma_l, theta_l, phi_l = angles_lst[l_idx]
              E_l_idx = self._get_layer_superoperator(beta_l, gamma_l, theta_l, phi_l)
              V_s = E_l_idx @ V_s
          return V_s
     
     def variational_ansatz(self, angles_lst):
          # Total evolution: U_total = E_{N-1} @ ... @ E_0
          total_evolution_op = self.I((2**self.L)**2)
          if self.L == 0: return self.rho_initial_vec

          for l_idx in range(self.number_of_layers): 
              beta_l, gamma_l, theta_l, phi_l = angles_lst[l_idx]
              E_l_idx = self._get_layer_superoperator(beta_l, gamma_l, theta_l, phi_l)
              total_evolution_op = E_l_idx @ total_evolution_op 
          
          return total_evolution_op @ self.rho_initial_vec

     def O(self): 
          if self.L < 2:
              return self.I(2**self.L)
          Z_0 = self.op_on_qubit(0, self.sigma_z)
          Z_1 = self.op_on_qubit(1, self.sigma_z)
          return Z_0 @ Z_1

     def cost_function(self, rho_final_state_vec):     
          obs_matrix = self.O() 
          rho_final_unvec = self.unvectorize_rho(rho_final_state_vec)
          
          if obs_matrix.shape[1] != rho_final_unvec.shape[0]:
              raise ValueError(f"Dim mismatch O@rho: O:{obs_matrix.shape}, rho:{rho_final_unvec.shape}")
          cost = (obs_matrix @ rho_final_unvec).trace()
          # HIGHLIGHT: Taking real part of the cost
          return np.real(cost)

     def optimize(self, learning_rate, max_iter, initial_angles):
          angles_lst = [list(angs) for angs in initial_angles] 

          cost_history = []
          initial_cost = self.cost_function(self.rho_initial_vec)
          cost_history.append(initial_cost)
          # print(f"Initial Cost: {initial_cost}")

          if self.L == 0:
            # print("L=0, no optimization possible/meaningful.")
            return {'final_cost': initial_cost, 'cost_history': cost_history, 'final_angles': angles_lst}

          for i_iter in range(max_iter):
               rho_L_final_vec = self.variational_ansatz(angles_lst)
               current_cost = self.cost_function(rho_L_final_vec)
               cost_history.append(current_cost)
               print(f"Iter {i_iter}, Cost: {current_cost:.8f}")
               
               if i_iter > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-9:
                   # print(f"Converged: small cost change at iter {i_iter}.")
                   break
               if abs(current_cost) < 1e-9 and self.epsilon > 1e-9: # Only check cost target if noise/dissipation is on
                   # print(f"Converged: cost target reached at iter {i_iter}.")
                   break

               new_angles_lst = [list(layer_angs) for layer_angs in angles_lst]

               for l_idx in range(self.number_of_layers):
                    # HIGHLIGHT: Gradient calculation logic and angle update order corrected
                    beta_orig, gamma_orig, theta_orig, phi_orig = angles_lst[l_idx]
                    
                    rho_evolved_before_l_vec = self.rho_l_vec(angles_lst, l_idx) 
                    # V_evol_after_l_super is E_{N-1} ... E_{l_idx+1}
                    V_evol_after_l_super = self.V_l_super(angles_lst, l_idx + 1)

                    D_channel = self.depolarizing_noise_super
                    U0s, U1s, U2s, U3s = self.U_0_super(), self.U_1_super(), self.U_2_super(), self.U_3_super()
                    
                    # Ensure expm gets dense arrays
                    exp_gU0 = expm((gamma_orig * U0s).toarray())
                    exp_bU1 = expm((beta_orig  * U1s).toarray())
                    exp_phU2= expm((phi_orig   * U2s).toarray())
                    exp_thU3= expm((theta_orig * U3s).toarray())

                    # Grad for gamma_l (U0) - parameter angles_lst[l_idx][1]
                    term_deriv_gamma = (D_channel @ exp_thU3 @ D_channel @ exp_phU2 @ D_channel @ 
                                        exp_bU1 @ D_channel @ (U0s @ exp_gU0) @ D_channel)
                    d_rho_dgamma_vec = V_evol_after_l_super @ term_deriv_gamma @ rho_evolved_before_l_vec
                    grad_gamma_l = self.cost_function(d_rho_dgamma_vec)
                    
                    # Grad for beta_l (U1) - parameter angles_lst[l_idx][0]
                    term_deriv_beta = (D_channel @ exp_thU3 @ D_channel @ exp_phU2 @ D_channel @ 
                                       (U1s @ exp_bU1) @ D_channel @ exp_gU0 @ D_channel)
                    d_rho_dbeta_vec = V_evol_after_l_super @ term_deriv_beta @ rho_evolved_before_l_vec
                    grad_beta_l = self.cost_function(d_rho_dbeta_vec)
                    
                    # Grad for phi_l (U2) - parameter angles_lst[l_idx][3]
                    term_deriv_phi = (D_channel @ exp_thU3 @ D_channel @ 
                                      (U2s @ exp_phU2) @ D_channel @ 
                                      exp_bU1 @ D_channel @ exp_gU0 @ D_channel)
                    d_rho_dphi_vec = V_evol_after_l_super @ term_deriv_phi @ rho_evolved_before_l_vec
                    grad_phi_l = self.cost_function(d_rho_dphi_vec)

                    # Grad for theta_l (U3) - parameter angles_lst[l_idx][2]
                    term_deriv_theta = (D_channel @ 
                                        (U3s @ exp_thU3) @ D_channel @ 
                                        exp_phU2 @ D_channel @ 
                                        exp_bU1 @ D_channel @ exp_gU0 @ D_channel)
                    d_rho_dtheta_vec = V_evol_after_l_super @ term_deriv_theta @ rho_evolved_before_l_vec
                    grad_theta_l = self.cost_function(d_rho_dtheta_vec)
                    
                    # Update angles: angles_lst[l_idx] = [beta, gamma, theta, phi]
                    new_angles_lst[l_idx][0] -= learning_rate * grad_beta_l   # beta
                    new_angles_lst[l_idx][1] -= learning_rate * grad_gamma_l  # gamma
                    new_angles_lst[l_idx][2] -= learning_rate * grad_theta_l  # theta
                    new_angles_lst[l_idx][3] -= learning_rate * grad_phi_l    # phi
               
               angles_lst = new_angles_lst 
          
          final_rho_L_f_vec = self.variational_ansatz(angles_lst)
          final_cost = self.cost_function(final_rho_L_f_vec)
          # print(f"Optimization finished. Final Cost: {final_cost:.8f}")
          
          return {
               'final_state_vec': final_rho_L_f_vec,
               'cost_history': cost_history,
               'final_angles': angles_lst,
               'final_cost': final_cost
          }