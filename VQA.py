import numpy as np
from qiskit import*
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os, qiskit_aer, pickle, itertools
from qiskit_algorithms import TimeEvolutionProblem
import trotter_for_open_quantum_systems as trotter

class VQA:
     def __init__(self, hamiltonian_of_molecule, cutoff, remove_identity):

          # Input: hamiltonian_of_molecule is a FermionicOp object
          self.hamiltonian_of_molecule = hamiltonian_of_molecule
          self.mapper = JordanWignerMapper()

          # Jordan-Wigner transformation of the fermionic Hamiltonian.
          self.qubit_hamiltonian = self.mapper.map(hamiltonian_of_molecule)
          self.L = len(self.qubit_hamiltonian.paulis.to_labels()[0]) # Number of qubits.

          # Set the cutoff for the number of terms in the Hamiltonian.
          self.cutoff = cutoff
          self.qubit_hamiltonian_truncated = self.qubit_hamiltonian.chop(cutoff).simplify()
          print(f"Number of terms in original qubit Hamiltonian: {len(self.qubit_hamiltonian.paulis.to_labels())}")
          print(f"Number of terms in truncated qubit Hamiltonian: {len(self.qubit_hamiltonian_truncated.paulis.to_labels())}")


          self.remove_identity = remove_identity
          if self.remove_identity == True:
               self.qubit_hamiltonian_truncated = self.qubit_hamiltonian_truncated - SparsePauliOp(["I"* (self.L)], [self.qubit_hamiltonian_truncated.coeffs[0]])
               print("The identity term has been removed from the qubit Hamiltonian.")

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

          # Check if the ground state is degenerate.
          es_idx = 2
          while True:
               if gs != es:
                    break
               else:
                    es = all_states[es_idx]
                    es_idx += 1

          # LUMO index.
          # Since Qiskit counts qubits from right to left, we reverse the bitstrings.
          # print(f"Ground state {gs}")
          # print(f"Excited state {es}")          
          gs = gs[0][::-1]
          es = es[0][::-1]
          # LUMO is the first index where the ground state and first excited state differ.
          for idx in range(self.L):
               if gs[idx] != es[idx]:
                    LUMO_idx =  idx
                    # print(f"The LUMO is at qubit index {LUMO_idx}")                    
                    return LUMO_idx                                    
                    
     def initial_state(self):
          initial_state_of_the_system = self.lowest_half_filled_states()[0][0]          
          return initial_state_of_the_system

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
     
     def number_operator(self):
          n = self.L  # total qubits
          # Qiskit: rightmost char is qubit 0
          z_terms = ["I"*(n-1-j) + "Z" + "I"*j for j in range(n)]
          op = SparsePauliOp(z_terms, -0.5*np.ones(n))
          op += SparsePauliOp(["I"*n], [0.5*n])  # (n/2) * I

          # Add one "I" at the end of each pauli string for the ancilla qubit.
          new_pauli_labels = [pauli + "I" for pauli in op.paulis.to_labels()]
          op = SparsePauliOp(new_pauli_labels, op.coeffs)

          return op.simplify()

     def energy_expectation_value_estimator(self,
                                   gamma_in,
                                   gamma_out,
                                   dt,
                                   number_of_data_points,
                                   simulation_type):

          initial_state = Statevector.from_label(self.initial_state())
          LUMO_qubit_index = self.LUMO_index()              

          new_pauli_labels = [pauli + "I" for pauli in self.qubit_hamiltonian_truncated.paulis.to_labels()]

          # Energy current.
          qubit_hamiltonian_truncated_obs = SparsePauliOp(new_pauli_labels, self.qubit_hamiltonian_truncated.coeffs)          

          expectation_values_mean_lst = []
          expectation_values_std_lst = []

          statevector_lst = []
          circuit_lst = []

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
                    passmanager = generate_preset_pass_manager(3, AerSimulator()) # Noisy simulator.                    
               elif simulation_type["Noisy"] == False:              
                    trotterop = trotter.TrotterQRTE(num_timesteps = ii, LUMO_qubit_index = LUMO_qubit_index, gamma_out = gamma_out, gamma_in = gamma_in, estimator = noiseless_estimator)
                    passmanager = generate_preset_pass_manager(3, AerSimulator())

               result = trotterop.evolve(problem)
               circuit_lst.append(result)
               trotter_circuit = result.evolved_state.decompose()
               statevector_lst.append(Statevector(result.evolved_state))

               isa_psi = passmanager.run(trotter_circuit)
               
               print("Final depth = ", isa_psi.depth())  
               isa_observables = qubit_hamiltonian_truncated_obs.apply_layout(isa_psi.layout)

               if simulation_type["Noisy"] == True:
                    job = noisy_estimator.run([(isa_psi, isa_observables)])
               elif simulation_type["Noisy"] == False:
                    job = noiseless_estimator.run([(isa_psi, isa_observables)])

               pub_result = job.result()[0]
               expectation_values_mean_lst.append(pub_result.data.evs)
               expectation_values_std_lst.append(pub_result.data.stds)               
               
          return expectation_values_mean_lst, expectation_values_std_lst, circuit_lst, statevector_lst
     
     def variational_circuit(self, angles_lst):
                 
          initial_state = Statevector.from_label(self.initial_state())
          LUMO_qubit_index = self.LUMO_index()

     
          new_pauli_labels = [pauli + "I" for pauli in self.qubit_hamiltonian_truncated.paulis.to_labels()]

          # Energy current.
          qubit_hamiltonian_truncated_obs = SparsePauliOp(new_pauli_labels, self.qubit_hamiltonian_truncated.coeffs)          

          # Always use noiseless estimator for VQA.
          from qiskit_aer import AerSimulator
          from qiskit_aer.primitives import EstimatorV2 as Estimator    
          noiseless_estimator = Estimator()     

          trotter_circuit_total = QuantumCircuit(self.L + 1) # +1 for the ancilla qubit.

          for layer in range(len(angles_lst)):
               # Unpacking the angles for unitary and the ancilla rotations.
               theta_unitary, theta_gamma_in, theta_gamma_out = angles_lst[layer]

               problem = TimeEvolutionProblem(self.qubit_hamiltonian_truncated, initial_state = initial_state, time = theta_unitary)
               trotterop = trotter.TrotterQRTE(num_timesteps = 1, LUMO_qubit_index = LUMO_qubit_index, gamma_out = theta_gamma_out, gamma_in = theta_gamma_in, estimator = noiseless_estimator)
               passmanager = generate_preset_pass_manager(3, AerSimulator())
               result = trotterop.evolve(problem)
               trotter_circuit = result.evolved_state.decompose()
               trotter_circuit_total = trotter_circuit_total.compose(trotter_circuit)

          isa_psi = passmanager.run(trotter_circuit_total)           
          isa_observables = qubit_hamiltonian_truncated_obs.apply_layout(isa_psi.layout)        

          job = noiseless_estimator.run([(isa_psi, isa_observables)])   
          pub_result = job.result()[0]

          energy_expectation_val = pub_result.data.evs          
          return energy_expectation_val
     
     def parameter_shift_gradient(self, angles_lst, shift = np.pi/2):
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
               
               energy_plus = self.variational_circuit(angles_plus)
               energy_minus = self.variational_circuit( angles_minus)
               
               grad_theta_unitary = (energy_plus - energy_minus) / (2 * np.sin(shift))
               
               # Gradient for theta_gamma_in
               angles_plus = angles_lst.copy()
               angles_minus = angles_lst.copy()
               
               angles_plus[layer_idx] = (theta_unitary, theta_gamma_in + shift, theta_gamma_out)
               angles_minus[layer_idx] = (theta_unitary, theta_gamma_in - shift, theta_gamma_out)
               
               energy_plus = self.variational_circuit(angles_plus)
               energy_minus = self.variational_circuit(angles_minus)
               
               grad_theta_gamma_in = (energy_plus - energy_minus) / (2 * np.sin(shift))

               # Gradient for theta_gamma_out
               angles_plus = angles_lst.copy()
               angles_minus = angles_lst.copy()

               angles_plus[layer_idx] = (theta_unitary, theta_gamma_in, theta_gamma_out + shift)
               angles_minus[layer_idx] = (theta_unitary, theta_gamma_in, theta_gamma_out - shift)

               energy_plus = self.variational_circuit(angles_plus)
               energy_minus = self.variational_circuit(angles_minus)
               grad_theta_gamma_out = (energy_plus - energy_minus) / (2 * np.sin(shift))

               gradients.append((grad_theta_unitary, grad_theta_gamma_in, grad_theta_gamma_out))

          return gradients
     
     def optimize(self, initial_angles, learning_rate, num_iterations):
          # Print a statement about the depth of the ansatz circuit.
          print(f"Running a depth {len(initial_angles)} ansatz circuit.")
          if len(initial_angles) > 10:
               print(f"Depth is quite large. This may take a long time to run.")

          # Ensure angles is a mutable list of tuples of floats
          angles = [(float(a[0]), float(a[1]), float(a[2])) for a in initial_angles]
          angles_history = [angles.copy()]
          energy_history = [np.array(self.variational_circuit(angles)).item()]

          for iteration in range(num_iterations):
               print(f"Iteration {iteration + 1}/{num_iterations}")
               print(f"Calculating gradients.")
               gradients = self.parameter_shift_gradient(angles)
               # Update angles using gradient descent (cast grads to float)
               angles = [
                    (float(theta_unitary) - learning_rate * float(grad_unitary),
                     float(theta_gamma_in) - learning_rate * float(grad_theta_gamma_in),
                     float(theta_gamma_out) - learning_rate * float(grad_theta_gamma_out))
                    for (theta_unitary, theta_gamma_in, theta_gamma_out), (grad_unitary, grad_theta_gamma_in, grad_theta_gamma_out)
                    in zip(angles, gradients)
               ]
               angles_history.append(angles.copy())
               energy = np.array(self.variational_circuit(angles)).item()
               energy_history.append(energy)
               print(f"Energy: {energy}")

          return angles, angles_history, energy_history