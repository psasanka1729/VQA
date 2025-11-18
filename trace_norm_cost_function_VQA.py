import numpy as np
from qiskit import*
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os, qiskit_aer, pickle, itertools
from qiskit_algorithms import TimeEvolutionProblem
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.sparse import csc_matrix, kron, identity

# Importing the custom trotter class that supports open quantum systems.
import trotter_for_open_quantum_systems as trotter

class VQA:

     # Define the Pauli matrices.
     I2 = csc_matrix(np.eye(2, dtype=complex))
     I4 = identity(4, dtype=complex, format='csc')    
     I4_sparse = identity(4, dtype=complex, format='csc')
     sigma_x = csc_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
     sigma_y = csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
     sigma_z = csc_matrix(np.array([[1, 0], [0, -1]], dtype=complex))     

     # Single qubit jump operators.
     L_in_op = csc_matrix((sigma_x + 1j * sigma_y) / 2)
     L_out_op = csc_matrix((sigma_x - 1j * sigma_y) / 2)   

     def __init__(self, hamiltonian_of_molecule, cutoff):

          # Hamiltonian_of_molecule is a FermionicOp object
          self.hamiltonian_of_molecule = hamiltonian_of_molecule
          # Use Jordan-Wigner mapping as default.
          self.mapper = JordanWignerMapper()

          # Jordan-Wigner transformation of the fermionic Hamiltonian.
          self.qubit_hamiltonian = self.mapper.map(hamiltonian_of_molecule)
          # Number of qubits.
          self.L = len(self.qubit_hamiltonian.paulis.to_labels()[0])

          # Set the cutoff for the number of terms in the Hamiltonian.
          self.cutoff = cutoff
          # Truncate small terms in the qubit Hamiltonian.
          self.qubit_hamiltonian_truncated = self.qubit_hamiltonian.chop(cutoff).simplify()
          print(f"Number of terms in original qubit Hamiltonian: {len(self.qubit_hamiltonian.paulis.to_labels())}")
          print(f"Number of terms in truncated qubit Hamiltonian: {len(self.qubit_hamiltonian_truncated.paulis.to_labels())}")

          self.L_in_op = VQA.L_in_op
          self.L_out_op = VQA.L_out_op
          self.I2 = VQA.I2
          self.I4 = VQA.I4
          self.I4_sparse = VQA.I4_sparse
          self.sigma_x = VQA.sigma_x
          self.sigma_y = VQA.sigma_y
          self.sigma_z = VQA.sigma_z

          # # Option to remove the identity term from the qubit Hamiltonian.
          # self.remove_identity = remove_identity
          # if self.remove_identity == True:
          #      self.qubit_hamiltonian_truncated = self.qubit_hamiltonian_truncated - SparsePauliOp(["I"* (self.L)], [self.qubit_hamiltonian_truncated.coeffs[0]])
          #      print("The identity term has been removed from the qubit Hamiltonian.")  

     """ 
     We now determine the index of the qubit that corresponds to the LUMO orbital.
     """
     # This function generates all possible half-filled states and calculates their energies to find the lowest energy states.
     def lowest_half_filled_states(self):
          """
          This function returns the lowest energy half-filled states of the fermionic_hamiltonian_matrix. 
          """
          # Generate all possible half-filled states.
          zeros = [0] * (self.L//2)
          ones = [1] * (self.L//2)
          binary_list = zeros + ones
          unique_permutations = set(itertools.permutations(binary_list))
          half_filled_states_lst = ["".join(map(str, perm)) for perm in unique_permutations]

          energy_half_filled_states = []
          # Calculate the energy of each half-filled state.
          for half_filled_state in half_filled_states_lst:
               #half_filled_state = half_filled_state[::-1]
               energy_half_filled_states.append((half_filled_state, np.real(Statevector.from_label(half_filled_state).expectation_value(self.qubit_hamiltonian_truncated))))
          sorted_energy_half_filled_states = sorted(energy_half_filled_states, key = lambda x: x[1])
          return sorted_energy_half_filled_states  
     
     def LUMO_index(self):

          all_states = self.lowest_half_filled_states()
          # Ground state.
          gs = all_states[0]
          # First excited state.
          es = all_states[1]

          # Check if the ground state is degenerate. This loop will run until we find a non-degenerate excited state.
          es_idx = 2
          while True:
               if gs != es:
                    break
               else:
                    es = all_states[es_idx]
                    es_idx += 1

          # LUMO index.
          # Since Qiskit counts qubits from right to left, we reverse the bitstrings.  
          gs = gs[0][::-1]
          es = es[0][::-1]
          # LUMO is the first index where the ground state and first excited state differ.
          for idx in range(self.L):
               if gs[idx] != es[idx]:
                    LUMO_idx =  idx
                    # print(f"The LUMO is at qubit index {LUMO_idx}")                    
                    return LUMO_idx                                    


     # Initial state is the lowest energy half-filled state.
     def initial_state(self):
          initial_state_of_the_system = self.lowest_half_filled_states()[0][0]          
          return initial_state_of_the_system

     # This function creates a noise model based on T1 and T2 times.
     def noise_model(self, T1, T2):
     
          # Import from Qiskit Aer noise module
          from qiskit_aer.noise import (
          NoiseModel,
          QuantumError,
          ReadoutError,
          depolarizing_error,
          pauli_error,
          thermal_relaxation_error,
          )

          # T1 and T2 values for qubits 0-L
          T1s = np.random.normal(T1, 10e3, self.L + 1)
          T2s = np.random.normal(T2, 10e3, self.L + 1)
          
          # Truncate random T2s <= T1s
          T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(self.L + 1)])
          
          # Instruction times (in nanoseconds)
          time_u1 = 0  # virtual gate
          time_u2 = 50  # (single X90 pulse)
          time_u3 = 100  # (two X90 pulses)
          time_cx = 300
          time_reset = 1000  # 1 microsecond
          time_measure = 1000  # 1 microsecond
          
          # QuantumError objects
          errors_reset = [
          thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)
          ]
          errors_measure = [
          thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)
          ]
          errors_u1 = [
          thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)
          ]
          errors_u2 = [
          thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)
          ]
          errors_u3 = [
          thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)
          ]
          errors_cx = [
          [
               thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx)
               )
               for t1a, t2a in zip(T1s, T2s)
          ]
          for t1b, t2b in zip(T1s, T2s)
          ]
          
          # Add errors to noise model
          noise_thermal = NoiseModel()
          for j in range(self.L + 1):
               noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
               noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
               noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
               noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
               noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
               for k in range(self.L + 1):
                    noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
     
          return noise_thermal
     
     """
          The number operator is given by N = \sum_{i} (I - Z_{i})/2 = (n/2) * I - 0.5 * \sum_{i} Z_{i}
          where n is the total number of qubits.
     
     """
     def number_operator(self):
          n = self.L
          # Qiskit: rightmost char is qubit 0
          z_terms = ["I"*(n-1-j) + "Z" + "I"*j for j in range(n)]
          op = SparsePauliOp(z_terms, -0.5*np.ones(n))
          op += SparsePauliOp(["I"*n], [0.5*n])  # (n/2) * I

          # Add one "I" at the end of each pauli string for the ancilla qubit.
          new_pauli_labels = [pauli + "I" for pauli in op.paulis.to_labels()]
          op = SparsePauliOp(new_pauli_labels, op.coeffs)

          return op.simplify()

     # This function constructs the multi-qubit jump operators acting on the LUMO orbital.
     def jump_operators(self):
          """Build jump operators on LUMO qubit"""
          LUMO_idx = self.LUMO_index()  # From right: 0 is rightmost
          LUMO_idx_from_left = self.L - LUMO_idx - 1  # Convert to left indexing
          
          # Build from left to right
          L_in_full = None
          L_out_full = None
          
          for qubit in range(self.L):  # 0, 1, 2, ..., L-1 (left to right)
               if qubit == LUMO_idx_from_left:
                    factor_in = self.L_in_op
                    factor_out = self.L_out_op
               else:
                    factor_in = self.I2
                    factor_out = self.I2
               
               if L_in_full is None:
                    L_in_full = factor_in
                    L_out_full = factor_out
               else:
                    L_in_full = kron(L_in_full, factor_in, format='csc')
                    L_out_full = kron(L_out_full, factor_out, format='csc')
          
          return L_in_full, L_out_full


     """
          Cost function C[\rho] = Tr[L(rho)^\dagger L(rho)].
     """
     def cost_function(self, rho, gamma_in, gamma_out):
          # Hamiltonian part.
          H = self.qubit_hamiltonian_truncated.to_matrix(sparse = True)
          # Commutator part.
          L_rho = -1j * (H @ rho - rho @ H)
          # Dissipative part.
          L_1, L_2 = self.jump_operators()
          L_rho += gamma_in * (L_1 @ rho @ L_1.getH() - 0.5 * (L_1.conj().T @ L_1 @ rho + rho @ L_1.conj().T @ L_1))
          L_rho += gamma_out * (L_2 @ rho @ L_2.getH() - 0.5 * (L_2.conj().T @ L_2 @ rho + rho @ L_2.conj().T @ L_2))

          cost = np.trace(L_rho.conj().T @ L_rho)
          if np.imag(cost) > 1.e-10:
               print("Warning: Cost function has an imaginary part:", np.imag(cost))
          return np.real(cost)
          
     # This function calculates the charge current as a function of time using the Trotter circuits.     
     def charge_current_vs_time(self,
                              gamma_in,
                              gamma_out,
                              dt,
                              number_of_data_points,
                              simulation_type):
          
          """
          Parameters
          ----------
          gamma_in : float
               The rate of electron tunneling into the LUMO orbital from the reservoir.
          gamma_out : float
               The rate of electron tunneling out of the LUMO orbital to the reservoir.
          dt : float
               The time step for the Trotter evolution.
          number_of_data_points : int
               The number of time steps to simulate.
          simulation_type : dict
               A dictionary specifying whether to use a noisy or noiseless simulator.
               Example: {"Noisy": True, "T_1": 50000, "T_2": 30000} for noisy simulation with T1=50us and T2=30us.
               Example: {"Noisy": False} for noiseless simulation.    
          Returns
          -------
          current_expectation_values_mean_lst : list
               A list of the mean values of the charge current at each time step.
          current_expectation_values_std_lst : list
               A list of the standard deviations of the charge current at each time step.
          circuit_lst : list
               A list of the Trotter circuits used for the time evolution.
          
          """
          
          # Initial state for the time evolution.
          initial_state = Statevector.from_label(self.initial_state())
          LUMO_qubit_index = self.LUMO_index()              

          # Expectation value of the number operator will be measured. The derivative of this gives the charge current.
          number_operator_observable = self.number_operator()

          expectation_values_mean_lst = []
          expectation_values_std_lst = []

          circuit_lst = []

          if simulation_type["Noisy"] == True:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator
               noise_thermal = self.noise_model(T1 = simulation_type["T_1"], T2 = simulation_type["T_2"])
               noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_thermal)))

          elif simulation_type["Noisy"] == False:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator    
               noiseless_estimator = Estimator()                       

          for ii in range(1, number_of_data_points):
               
               if gamma_in * dt >= 1.0 or gamma_out * dt >= 1.0:
                    print("Either gamma_in, gamma_out or dt is too large. Make gamma_in * dt and gamma_out * dt < 1.")
                    print("This is necessary for the Trotter approximation to be valid.")
                    break

               t_final = np.around(ii*dt, 5)
               print("Time = ", t_final)
          
               # This section sets up the time evolution problem and the trotter circuit using the custom trotter class.
               problem = TimeEvolutionProblem(self.qubit_hamiltonian_truncated, initial_state = initial_state, time = t_final)

               if simulation_type["Noisy"] == True:
                    trotterop = trotter.TrotterQRTE(num_timesteps = ii, LUMO_qubit_index = LUMO_qubit_index, gamma_out = gamma_out, gamma_in = gamma_in, estimator = noisy_estimator)
                    passmanager = generate_preset_pass_manager(2, AerSimulator()) # Noisy simulator.                    
               elif simulation_type["Noisy"] == False:              
                    trotterop = trotter.TrotterQRTE(num_timesteps = ii, LUMO_qubit_index = LUMO_qubit_index, gamma_out = gamma_out, gamma_in = gamma_in, estimator = noiseless_estimator)
                    passmanager = generate_preset_pass_manager(2, AerSimulator())

               # These are standard ways to calculate the time evolution using Qiskit.
               result = trotterop.evolve(problem)
               circuit_lst.append(result)
               trotter_circuit = result.evolved_state.decompose()

               isa_psi = passmanager.run(trotter_circuit)
               
               print("Final depth = ", isa_psi.depth())  
               isa_observables = number_operator_observable.apply_layout(isa_psi.layout)

               if simulation_type["Noisy"] == True:
                    job = noisy_estimator.run([(isa_psi, isa_observables)])
               elif simulation_type["Noisy"] == False:
                    job = noiseless_estimator.run([(isa_psi, isa_observables)])

               pub_result = job.result()[0]
               expectation_values_mean_lst.append(pub_result.data.evs)
               expectation_values_std_lst.append(pub_result.data.stds)       

          # Calculate the charge current as the derivative of the expectation value of the number operator.
          current_expectation_values_mean_lst = np.diff(expectation_values_mean_lst) / dt
          current_expectation_values_std_lst = np.diff(expectation_values_std_lst) / dt
               
          return current_expectation_values_mean_lst, current_expectation_values_std_lst, circuit_lst        


     # This function is almost the same as the above function except it returns the density matrix at the final time step.
     def time_evolved_of_density_matrix(self,
                         gamma_in,
                         gamma_out,
                         dt,
                         number_of_data_points,
                         simulation_type):
          
          """
          Parameters
          ----------
          gamma_in : float
               The rate of electron tunneling into the LUMO orbital from the reservoir.
          gamma_out : float
               The rate of electron tunneling out of the LUMO orbital to the reservoir.
          dt : float
               The time step for the Trotter evolution.
          number_of_data_points : int
               The number of time steps to simulate.
          simulation_type : dict
               A dictionary specifying whether to use a noisy or noiseless simulator.
               Example: {"Noisy": True, "T_1": 50000, T_2 = 30000} for noisy simulation with T1=50us and T2=30us.
               Example: {"Noisy": False} for noiseless simulation.    
          Returns
          -------
          rho_system : ndarray
               The reduced density matrix of the system (tracing out the ancilla qubit) at the final time step.    
          current_expectation_values_std_lst : list
               A list of the standard deviations of the charge current at each time step.
          circuit_lst : list
               A list of the Trotter circuits used for the time evolution.
          """

          initial_state = Statevector.from_label(self.initial_state())
          LUMO_qubit_index = self.LUMO_index()              

          rho_lst = []

          if simulation_type["Noisy"] == True:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator
               noise_thermal = self.noise_model(T1 = simulation_type["T_1"], T2 = simulation_type["T_2"])
               sim_thermal = AerSimulator(noise_model = noise_thermal)
               noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_thermal)))

          elif simulation_type["Noisy"] == False:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator    
               noiseless_estimator = Estimator()      
               sim_noiseless = AerSimulator()                 


          for ii in range(1, number_of_data_points):
               
               if gamma_in * dt >= 1.0 or gamma_out * dt >= 1.0:
                    print("Either gamma_in, gamma_out or dt is too large. Make gamma_in * dt and gamma_out * dt < 1.")
                    print("This is necessary for the Trotter approximation to be valid.")
                    break

               t_final = np.around(ii*dt,2)
               print("Time = ", t_final)
          
               # This section sets up the time evolution problem and the trotter circuit using the custom trotter class.
               problem = TimeEvolutionProblem(self.qubit_hamiltonian_truncated, initial_state = initial_state, time = t_final)

               if simulation_type["Noisy"] == True:
                    trotterop = trotter.TrotterQRTE(num_timesteps = ii, LUMO_qubit_index = LUMO_qubit_index, gamma_out = gamma_out, gamma_in = gamma_in, estimator = noisy_estimator)
                    # passmanager = generate_preset_pass_manager(3, AerSimulator()) # Noisy simulator.                    
               elif simulation_type["Noisy"] == False:              
                    trotterop = trotter.TrotterQRTE(num_timesteps = ii, LUMO_qubit_index = LUMO_qubit_index, gamma_out = gamma_out, gamma_in = gamma_in, estimator = noiseless_estimator)
                    # passmanager = generate_preset_pass_manager(3, AerSimulator())

               result = trotterop.evolve(problem)
               trotter_circuit = result.evolved_state.decompose()
               # Transpile the quantum circuit. This step is necessary for density matrix simulation.
               trotter_circuit = transpile(trotter_circuit, basis_gates = ['sx', 'cx', 'rz'], optimization_level = 2)
               trotter_circuit.save_density_matrix()     

               print(f"Trotter circuit depth at time {t_final} is {trotter_circuit.depth()}.")
               if simulation_type["Noisy"] == True:   
                    job = sim_thermal.run([trotter_circuit])
                    result = job.result().data()
                    rho = result.get('density_matrix')                        
               elif simulation_type["Noisy"] == False:
                    job = sim_noiseless.run([trotter_circuit])
                    result = job.result().data()
                    rho = result.get('density_matrix')

               # Trace over the ancilla qubit.
               rho_system = partial_trace(rho.data, [0])
               rho_lst.append(rho_system.data)

          return rho_lst
   
     
     def variational_circuit(self, gamma_in, gamma_out, angles_lst, simulation_type):

          """
          Parameters
          ----------
          gamma_in : float
               The rate of electron tunneling into the LUMO orbital from the reservoir.
          gamma_out : float
               The rate of electron tunneling out of the LUMO orbital to the reservoir.
          angles_lst : list
               A list of tuples containing the angles for each layer of the variational circuit.
               Example: [(theta_1, theta_gamma_in_1, theta_gamma_out_1), (theta_2, theta_gamma_in_2, theta_gamma_out_2), ...]
          simulation_type : dict
               A dictionary specifying whether to use a noisy or noiseless simulator.
               Example: {"Noisy": True, "T_1": 305, T_2 = 350}.
               Example: {"Noisy": False} for noiseless simulation.
          Returns
          -------
          cost : float
               The value of the cost function C[\rho] = Tr[L(rho)^\dagger L(rho)] for the final density matrix rho.
          rho_system : ndarray
               The reduced density matrix of the system (tracing out the ancilla qubit) after applying the variational circuit.
          """

          initial_state = Statevector.from_label(self.initial_state())
               
          LUMO_qubit_index = self.LUMO_index()

          if simulation_type["Noisy"] == True:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator
               noise_thermal = self.noise_model(T1 = simulation_type["T_1"], T2 = simulation_type["T_2"])
               sim_thermal = AerSimulator(noise_model = noise_thermal)
               noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_thermal)))

          elif simulation_type["Noisy"] == False:
               from qiskit_aer import AerSimulator
               from qiskit_aer.primitives import EstimatorV2 as Estimator    
               noiseless_estimator = Estimator()      
               sim_noiseless = AerSimulator()             

          trotter_circuit_total = QuantumCircuit(self.L + 1) # +1 for the ancilla qubit.

          for layer in range(len(angles_lst)):
               # Unpacking the angles for unitary and the ancilla rotations.
               theta_unitary, theta_gamma_in, theta_gamma_out = angles_lst[layer]

               problem = TimeEvolutionProblem(self.qubit_hamiltonian_truncated, initial_state = initial_state, time = theta_unitary)

               if simulation_type["Noisy"] == True:
                    trotterop = trotter.TrotterQRTE(num_timesteps = 1, LUMO_qubit_index = LUMO_qubit_index, gamma_out = theta_gamma_out, gamma_in = theta_gamma_in, estimator = noisy_estimator)
                    # passmanager = generate_preset_pass_manager(3, AerSimulator()) # Noisy simulator.                    
               elif simulation_type["Noisy"] == False:              
                    trotterop = trotter.TrotterQRTE(num_timesteps = 1, LUMO_qubit_index = LUMO_qubit_index, gamma_out = theta_gamma_out, gamma_in = theta_gamma_in, estimator = noiseless_estimator)

               result = trotterop.evolve(problem)
               trotter_circuit = result.evolved_state.decompose()
               trotter_circuit_total = trotter_circuit_total.compose(trotter_circuit)

          # Transpile the quantum circuit.
          trotter_circuit_transpiled = transpile(trotter_circuit_total, basis_gates = ['sx', 'cx', 'rz'], optimization_level = 2)
          trotter_circuit_transpiled.save_density_matrix()     
          print(f"Variational circuit depth is {trotter_circuit_transpiled.depth()}.")

          # isa_psi = passmanager.run(trotter_circuit)
          if simulation_type["Noisy"] == True:   
               job = sim_thermal.run([trotter_circuit_transpiled])
               result = job.result().data()
               rho = result.get('density_matrix')                        
          elif simulation_type["Noisy"] == False:
               job = sim_noiseless.run([trotter_circuit_transpiled])
               result = job.result().data()
               rho = result.get('density_matrix')

          rho_system = partial_trace(rho.data, [0]) # Partial trace over the ancilla qubit.        

          cost = self.cost_function(rho_system.data, gamma_in, gamma_out)

          return rho_system, cost       
     
     def parameter_shift_gradient(self, gamma_in, gamma_out, angles_lst, simulation_type, shift = np.pi/2):
          """
          This function calculates the gradient of the energy expectation value with respect to the parameters using the parameter-shift rule.
          """
          gradients = []

          for layer_idx, (theta_unitary, theta_gamma_in, theta_gamma_out) in enumerate(angles_lst):
               # Gradient for theta_unitary
               angles_plus = angles_lst.copy()
               angles_minus = angles_lst.copy()
               
               angles_plus[layer_idx] = (theta_unitary + shift, theta_gamma_in, theta_gamma_out)
               angles_minus[layer_idx] = (theta_unitary - shift, theta_gamma_in, theta_gamma_out)
               
               _, cost_plus = self.variational_circuit(gamma_in, gamma_out, angles_plus, simulation_type)
               _, cost_minus = self.variational_circuit(gamma_in, gamma_out, angles_minus, simulation_type)
               
               grad_theta_unitary = (cost_plus - cost_minus) / (2 * np.sin(shift))
               
               # Gradient for theta_gamma_in
               angles_plus = angles_lst.copy()
               angles_minus = angles_lst.copy()
               
               angles_plus[layer_idx] = (theta_unitary, theta_gamma_in + shift, theta_gamma_out)
               angles_minus[layer_idx] = (theta_unitary, theta_gamma_in - shift, theta_gamma_out)
               
               _, cost_plus = self.variational_circuit(gamma_in, gamma_out, angles_plus, simulation_type)
               _, cost_minus = self.variational_circuit(gamma_in, gamma_out, angles_minus, simulation_type)
               
               grad_theta_gamma_in = (cost_plus - cost_minus) / (2 * np.sin(shift))

               # Gradient for theta_gamma_out
               angles_plus = angles_lst.copy()
               angles_minus = angles_lst.copy()

               angles_plus[layer_idx] = (theta_unitary, theta_gamma_in, theta_gamma_out + shift)
               angles_minus[layer_idx] = (theta_unitary, theta_gamma_in, theta_gamma_out - shift)

               _, cost_plus = self.variational_circuit(gamma_in, gamma_out, angles_plus, simulation_type)
               _, cost_minus = self.variational_circuit(gamma_in, gamma_out, angles_minus, simulation_type)
               grad_theta_gamma_out = (cost_plus - cost_minus) / (2 * np.sin(shift))

               gradients.append((grad_theta_unitary, grad_theta_gamma_in, grad_theta_gamma_out))

          return gradients
     
     # This function performs the optimization using gradient descent.
     def optimize(self, gamma_in, gamma_out, initial_angles, learning_rate, num_iterations, simulation_type):
          # Print a statement about the depth of the ansatz circuit.
          print(f"Running a depth {len(initial_angles)} ansatz circuit.")
          if len(initial_angles) > 10:
               print(f"Depth is quite large. This may take a long time to run.")

          # Ensure angles is a mutable list of tuples of floats
          angles = [(float(a[0]), float(a[1]), float(a[2])) for a in initial_angles]
          angles_history = [angles.copy()]
          rho_history = []
          cost_history = []
          # Calculate initial cost and density matrix
          initial_rho, initial_cost = self.variational_circuit(gamma_in, gamma_out, initial_angles, simulation_type)
          cost_history = [initial_cost]
          rho_history = [initial_rho]

          for iteration in range(num_iterations):
               print(f"Iteration {iteration + 1}/{num_iterations}")
               print(f"Calculating gradients.")
               gradients = self.parameter_shift_gradient(gamma_in, gamma_out, angles, simulation_type)
               # Update angles using gradient descent (cast grads to float)
               angles = [
                    (float(theta_unitary) - learning_rate * float(grad_unitary),
                     float(theta_gamma_in) - learning_rate * float(grad_theta_gamma_in),
                     float(theta_gamma_out) - learning_rate * float(grad_theta_gamma_out))
                    for (theta_unitary, theta_gamma_in, theta_gamma_out), (grad_unitary, grad_theta_gamma_in, grad_theta_gamma_out)
                    in zip(angles, gradients)
               ]
               angles_history.append(angles.copy())
               rho, cost = self.variational_circuit(gamma_in, gamma_out, angles, simulation_type)
               rho_history.append(rho)
               cost_history.append(cost)
               print(f"Cost : {cost}")

          return angles, angles_history, rho_history, cost_history

     def optimize_last_layer(self, gamma_in, gamma_out, angles, learning_rate, num_iterations, simulation_type):
          # Ensure angles is a mutable list of tuples of floats
          angles = [(float(a[0]), float(a[1]), float(a[2])) for a in angles]
          angles_history = [angles.copy()]
          rho_history = []
          cost_history = []
          # Calculate initial cost and density matrix
          initial_rho, initial_cost = self.variational_circuit(gamma_in, gamma_out, angles, simulation_type)
          cost_history = [initial_cost]
          rho_history = [initial_rho]

          for iteration in range(num_iterations):
               print(f"Iteration {iteration + 1}/{num_iterations}")
               print(f"Calculating gradients.")
               gradients = self.parameter_shift_gradient(gamma_in, gamma_out, angles, simulation_type)
               
               # Get the current values for the last layer
               theta_unitary, theta_gamma_in, theta_gamma_out = angles[-1]
               
               # Get gradients for the last layer
               grad_unitary, grad_theta_gamma_in, grad_theta_gamma_out = gradients[-1]
               
               # Update only the last layer
               angles[-1] = (
                    theta_unitary - learning_rate * float(grad_unitary),
                    theta_gamma_in - learning_rate * float(grad_theta_gamma_in),
                    theta_gamma_out - learning_rate * float(grad_theta_gamma_out)
               )
               
               angles_history.append(angles.copy())
               rho, cost = self.variational_circuit(gamma_in, gamma_out, angles, simulation_type)
               rho_history.append(rho)
               cost_history.append(cost)
               print(f"Cost : {cost}")

          return angles, angles_history, rho_history, cost_history