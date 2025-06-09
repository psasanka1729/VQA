import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, kron, identity
from scipy.sparse.linalg import expm, expm_multiply

class VQE_Noisy_Optimizer:
    # Encapsulates the VQE simulation with a Lindblad-based cost function.

    # Define Pauli matrices as class attributes.
    I2 = csc_matrix(np.eye(2, dtype=complex))
    sigma_x = csc_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_y = csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
    sigma_z = csc_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    sigma_plus = csc_matrix((sigma_x + 1j * sigma_y) / 2)
    sigma_minus = csc_matrix((sigma_x - 1j * sigma_y) / 2)

    def __init__(self, J=1.0, Delta=0.5, L=1.0, gamma_e=0.05, gamma_d=0.05, gamma_depol=0.0):
        # Initializes the VQE optimizer with system and noise parameters.
        self.J = J
        self.Delta = Delta
        self.L = L
        self.gamma_e = gamma_e
        self.gamma_d = gamma_d
        self.gamma_depol = gamma_depol

        # Pre-calculate the depolarizing noise superoperator.
        self.D_mat = self._get_depolarizing_superop()

        # To store optimization results.
        self.best_angles_ = None
        self.best_cost_ = None
        self.cost_history_ = []
        self.gradients_history_ = []

    # --- Utility Methods ---
    @staticmethod
    def _vectorize_rho(rho):
        # Converts a 2x2 density matrix to a 4x1 column vector.
        if hasattr(rho, "toarray"):
            rho = rho.toarray()
        rho_vec = rho.flatten(order="F")[:, None] # Ensure shape is (4,1) not (1,4).
        return csc_matrix(rho_vec)

    @staticmethod
    def _unvectorize_rho(rho_vec):
        # Converts a 4x1 density vector back to a 2x2 matrix.
        rho_dense_vec = rho_vec.toarray() if hasattr(rho_vec, "toarray") else rho_vec
        rho_dense_mat = rho_dense_vec.reshape((2, 2), order="F")
        return csc_matrix(rho_dense_mat)

    # --- Generator Superoperators (Liouvillians) ---
    def _L_depol_super(self):
        # Generator for depolarizing noise.
        term_x = kron(self.sigma_x.conj(), self.sigma_x, format='csc')
        term_y = kron(self.sigma_y.conj(), self.sigma_y, format='csc')
        term_z = kron(self.sigma_z.conj(), self.sigma_z, format='csc')
        return self.gamma_depol * ((1/3.0) * (term_x + term_y + term_z) - kron(self.I2, self.I2))

    def _get_depolarizing_superop(self):
        # Matrix exponential of the depolarizing generator.
        if abs(self.gamma_depol) < 1e-12:
            return np.eye(4, dtype=complex)
        return expm(self._L_depol_super().toarray())

    def _L_int(self, x_mean):
        # Generator for mean-field interaction Hamiltonian.
        H = self.J * self.L * x_mean * self.sigma_x
        return -1j * (kron(self.I2, H) - kron(H.T, self.I2))

    def _L_f(self):
        # Generator for free Hamiltonian.
        H = -self.Delta * self.sigma_z
        return -1j * (kron(self.I2, H) - kron(H.T, self.I2))

    def _L_rel(self):
        # Generator for relaxation.
        L = np.sqrt(self.gamma_e) * self.sigma_minus
        LdL = L.conj().T @ L
        return (kron(L.conj(), L) - 0.5 * kron(self.I2, LdL) - 0.5 * kron(LdL.T, self.I2))

    def _L_dep(self):
        # Generator for dephasing.
        L = np.sqrt(self.gamma_d) * (self.I2 + self.sigma_z) / 2.0
        LdL = L.conj().T @ L
        return (kron(L.conj(), L) - 0.5 * kron(self.I2, LdL) - 0.5 * kron(LdL.T, self.I2))

    # --- Lindbladian, Cost, and Gradient for L^2 cost function ---
    def _lindbladian_mat(self, rho_mat, x_mean):
        # Calculates L(rho) and returns the result as a 2x2 matrix.
        H = self.J * self.L * x_mean * self.sigma_x - self.Delta * self.sigma_z
        L1 = np.sqrt(self.gamma_e) * self.sigma_minus
        L2 = np.sqrt(self.gamma_d) * (self.I2 + self.sigma_z) / 2
        
        comm = -1j * (H @ rho_mat - rho_mat @ H)
        dis1 = L1 @ rho_mat @ L1.conj().T - 0.5 * (L1.conj().T @ L1 @ rho_mat + rho_mat @ L1.conj().T @ L1)
        dis2 = L2 @ rho_mat @ L2.conj().T - 0.5 * (L2.conj().T @ L2 @ rho_mat + rho_mat @ L2.conj().T @ L2)
        
        return comm + dis1 + dis2

    def _cost_function_lindblad(self, rho_final_mat, x_final):
        # Calculates the Lindblad cost: Tr[ (L*rho)^dagger * (L*rho) ].
        L_rho = self._lindbladian_mat(rho_final_mat, x_final)
        product_mat = L_rho.conj().T @ L_rho
        # Convert to dense array before trace to avoid sparse diag error.
        return np.real(product_mat.toarray().trace())

    def _calculate_gradients_lindblad(self, n_layers, E_lists, M_list, rho_list, x_list):
        # Calculates the exact gradient for the Lindblad cost function.
        E_int_list, E_f_list, E_rel_list, E_dep_list = E_lists
        # Initialize the gradients for each angle
        grads = [[0.0] * 4 for _ in range(n_layers)]

        rho_final_mat = self._unvectorize_rho(rho_list[-1])
        # Calculate x_final from the true final density matrix
        x_final = np.real(np.trace((self.sigma_x @ rho_final_mat).toarray()))
        L_rho_final_mat = self._lindbladian_mat(rho_final_mat, x_final)

        for l in range(n_layers):
            # Product of layers after layer l
            propagator_after_layer = np.eye(4, dtype=complex)
            for i in range(l + 1, n_layers):
                propagator_after_layer = M_list[i] @ propagator_after_layer
            
            # Density matrix before layer l
            rho_before_layer = rho_list[l]

            # --- Gradient for theta_int (k=0) ---
            propagator = self.D_mat @ E_dep_list[l] @ self.D_mat @ E_rel_list[l] @ self.D_mat @ E_f_list[l] @ self.D_mat
            # Using \mathcal{L}_int = -i [H_int, .] to calculate the commutator
            drho_init = self._L_int(x_list[l]) @ (self.D_mat @ rho_before_layer)
            drho_final_mat = self._unvectorize_rho(propagator_after_layer @ propagator @ drho_init)
            L_d_rho = self._lindbladian_mat(drho_final_mat, x_final)
            trace_term = (L_rho_final_mat.conj().T @ L_d_rho).toarray().trace()
            grads[l][0] = 2 * np.real(trace_term)

            # --- Gradient for theta_f (k=1) ---
            propagator = self.D_mat @ E_dep_list[l] @ self.D_mat @ E_rel_list[l] @ self.D_mat
            rho_before_op = E_int_list[l] @ (self.D_mat @ rho_before_layer)
            drho_init = self._L_f() @ (self.D_mat @ rho_before_op)
            drho_final_mat = self._unvectorize_rho(propagator_after_layer @ propagator @ drho_init)
            L_d_rho = self._lindbladian_mat(drho_final_mat, x_final)
            trace_term = (L_rho_final_mat.conj().T @ L_d_rho).toarray().trace()
            grads[l][1] = 2 * np.real(trace_term)

            # --- Gradient for theta_rel (k=2) ---
            propagator = self.D_mat @ E_dep_list[l] @ self.D_mat
            rho_before_op = E_f_list[l] @ (self.D_mat @ E_int_list[l] @ (self.D_mat @ rho_before_layer))
            drho_init = self._L_rel() @ (self.D_mat @ rho_before_op)
            drho_final_mat = self._unvectorize_rho(propagator_after_layer @ propagator @ drho_init)
            L_d_rho = self._lindbladian_mat(drho_final_mat, x_final)
            trace_term = (L_rho_final_mat.conj().T @ L_d_rho).toarray().trace()
            grads[l][2] = 2 * np.real(trace_term)

            # --- Gradient for theta_dep (k=3) ---
            propagator = self.D_mat
            rho_before_op = E_rel_list[l] @ (self.D_mat @ E_f_list[l] @ (self.D_mat @ E_int_list[l] @ (self.D_mat @ rho_before_layer)))
            drho_init = self._L_dep() @ (self.D_mat @ rho_before_op)
            drho_final_mat = self._unvectorize_rho(propagator_after_layer @ propagator @ drho_init)
            L_d_rho = self._lindbladian_mat(drho_final_mat, x_final)
            trace_term = (L_rho_final_mat.conj().T @ L_d_rho).toarray().trace()
            grads[l][3] = 2 * np.real(trace_term)
            
        return grads

    # --- Ansatz and Optimizer ---
    def _variational_ansatz(self, number_of_layers, angles_lst, rho_initial_vec):
        # Applies the ansatz and returns intermediate values needed for the gradient.
        rho_vec = rho_initial_vec.copy()
        
        rho_list = [rho_vec]
        x_list = []
        E_lists = ([], [], [], [])
        M_list = []

        x_cur = np.real(np.trace((self.sigma_x @ self._unvectorize_rho(rho_vec)).toarray()))

        for i in range(number_of_layers):
            theta_int, theta_f, theta_rel, theta_dep = angles_lst[i]
            x_list.append(x_cur)

            E_int = expm((theta_int * self._L_int(x_cur)).toarray())
            E_f   = expm((theta_f   * self._L_f()).toarray())
            E_rel = expm((theta_rel * self._L_rel()).toarray())
            E_dep = expm((theta_dep * self._L_dep()).toarray())

            M = self.D_mat @ E_dep @ self.D_mat @ E_rel @ self.D_mat @ E_f @ self.D_mat @ E_int @ self.D_mat
            rho_vec = M @ rho_vec
            
            x_cur = np.real(np.trace((self.sigma_x @ self._unvectorize_rho(rho_vec)).toarray()))

            rho_list.append(rho_vec)
            E_lists[0].append(E_int)
            E_lists[1].append(E_f)
            E_lists[2].append(E_rel)
            E_lists[3].append(E_dep)
            M_list.append(M)

        return rho_vec, E_lists, M_list, rho_list, x_list

    def run_optimizer(self, n_layers, initial_angles, learning_rate, max_iterations, tolerance=1e-8, clip_threshold=1.0):
        # Performs gradient descent to minimize the Lindblad cost function.
        psi_initial = np.random.rand(2) + 1j * np.random.rand(2)
        psi_initial /= np.linalg.norm(psi_initial)
        rho_initial_vec = self._vectorize_rho(np.outer(psi_initial, psi_initial.conj()))
        angles_lst = [list(params) for params in initial_angles]
        best_angles = [list(a) for a in angles_lst]
        
        self.cost_history_ = []
        self.grads_history_ = []
        best_cost = np.inf
        rho_lst = []

        for iteration in range(max_iterations):
            rho_final_vec, E_lists, M_list, rho_list, x_list = \
                self._variational_ansatz(n_layers, angles_lst, rho_initial_vec)

            rho_final_mat = self._unvectorize_rho(rho_final_vec)
            rho_lst.append(rho_final_mat)
            
            # --- FIX: Calculate x_final from the true final density matrix ---
            # The previous logic used x_list[-1], which was the value *before* the last layer evolution.
            x_final = np.real(np.trace((self.sigma_x @ rho_final_mat).toarray()))
            # --- END FIX ---
            
            current_cost = self._cost_function_lindblad(rho_final_mat, x_final)
            self.cost_history_.append(current_cost)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_angles = [list(a) for a in angles_lst]
            
            print(f"Iteration {iteration}: Cost = {current_cost:.5e}, Best Cost = {best_cost:.5e}")

            if current_cost < tolerance:
                print(f"Convergence reached at iteration {iteration}.")
                break
            
            gradients = self._calculate_gradients_lindblad(n_layers, E_lists, M_list, rho_list, x_list)
            
            grad_flat = np.array(gradients).flatten()
            grad_norm = np.linalg.norm(grad_flat)
            if grad_norm > clip_threshold:
                gradients = (np.array(gradients) * clip_threshold / grad_norm).tolist()
            
            self.grads_history_.append(gradients)

            for l in range(n_layers):
                for k in range(4):
                    angles_lst[l][k] -= learning_rate * gradients[l][k]
                    
        self.best_angles_ = best_angles
        self.best_cost_ = best_cost
        
        # --- FIX: Recalculate final state and x value with best angles to ensure consistency ---
        final_rho_vec, _, _, _, _ = self._variational_ansatz(n_layers, self.best_angles_, rho_initial_vec)
        final_rho_mat = self._unvectorize_rho(final_rho_vec)
        final_x = np.real(np.trace((self.sigma_x @ final_rho_mat).toarray()))
        # --- END FIX ---
        
        print(f"\nOptimization finished.")
        print(f"Best Cost: {self.best_cost_:.8e}")
        print(f"Best Angles: {self.best_angles_}")
        print(f"Final x value: {final_x}") # --- FIX: Print the corrected final x value ---
        
        return self.best_angles_, self.best_cost_, self.cost_history_, self.grads_history_, rho_lst

    # --- Plotting Methods ---
    def plot_cost_history(self):
        # Plots the cost function vs. iterations.
        if not self.cost_history_:
            print("No cost history to plot. Run the optimizer first.")
            return
        plt.figure(figsize=(10, 7))
        plt.plot(np.real(self.cost_history_), 'o-')
        plt.xlabel("Iterations")
        plt.ylabel("Cost = Tr[(Lρ)²]")
        plt.title("Cost vs Iterations")
        plt.yscale('log')  # Log scale for better visibility
        plt.show()

    def plot_gradient_history(self):
        # Plots the gradients vs. iterations.
        if not self.grads_history_:
            print("No gradient history to plot. Run the optimizer first.")
            return
            
        param_names = ["θ_int", "θ_f", "θ_rel", "θ_dep"]
        fig, ax = plt.subplots(figsize=(10, 7))
        
        gradients_by_param = [[] for _ in range(4)]
        for grad_step in self.grads_history_:
            # This plotting simplification assumes n_layers = 1.
            for l_idx, layer_grads in enumerate(grad_step):
                if l_idx == 0:
                    for p_idx in range(4):
                        gradients_by_param[p_idx].append(layer_grads[p_idx])

        for param_idx, param_name in enumerate(param_names):
            ax.plot(gradients_by_param[param_idx], label=param_name)
            
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Gradient")
        ax.set_title("Gradients vs Iterations (Layer 0)")
        ax.legend()
        plt.show()