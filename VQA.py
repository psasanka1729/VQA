# %%
# General imports
import numpy as np
import qiskit
#from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt

# %%
# [Reference]: https://docs.quantum.ibm.com/guides/build-noise-models
# https://docs.quantum.ibm.com/guides/simulate-with-qiskit-aer

from qiskit_aer import AerSimulator

from qiskit_aer.primitives import SamplerV2 as Sampler
#from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit.primitives import StatevectorSampler
# sampler = StatevectorSampler()
sampler = Sampler()


from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import (NoiseModel,QuantumError,ReadoutError,depolarizing_error,pauli_error,thermal_relaxation_error)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# YOUR_API_TOKEN = "a44932650c6324c729fbd156e2808cff2dc96cc78fdaf3058a5e3583edfadad4acf73471140788f0efa7ed71a98f1e13069af52fe115e4e0e2bb868ab0be5d70"
# QiskitRuntimeService.save_account(channel="ibm_quantum", token= YOUR_API_TOKEN, overwrite = True)
# service = QiskitRuntimeService()
# backend = service.backend("ibm_brisbane")
# noise_model = NoiseModel.from_backend(backend)

# %% [markdown]
# # Code for Shadow Tomography of Quantum States

# %%
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, kron
from numpy import trace

# Here we will write the necessary codes for shadow tomography.

# Inverse shadow channel for Clifford gates.
def Minv(N_qubits,X):
     return ((2**N_qubits+1.))*X - np.eye(2**N_qubits)

# Function performs shadow tomography for a given quantum circuit.
def shadow_tomography_clifford(N_qubits,quantum_circuit, reps = 1):

     """
     This function performs shadow tomography for a given quantum circuit.
     Parameters:
     N_qubits: int
          Number of qubits in the quantum circuit.
     quantum_circuit: qiskit.QuantumCircuit

     Output:
     shadows: list
          List of shadow density matrices of the system.
     """

     # Random Clifford gates to apply to the quantum circuit.
     cliffords = [qiskit.quantum_info.random_clifford(N_qubits,) for _ in range(n_Shadows)]

     #rho_actual = qiskit.quantum_info.DensityMatrix(quantum_circuit).data

     results = []
for cliff in cliffords:
     # Compose the quantum circuit with the random Clifford gate.
     # This amounts to randomly rotating the state as called in the original paper.
     qc_c = quantum_circuit.copy()
     qc_c.append(cliff.to_circuit(), quantum_circuit.qubits[1:])

     # If not transpiled gives error.
     qc_c = transpile(qc_c, basis_gates = ["rx", "ry", "rz", "cx", "x", "y", "z"])
     # Measuring the state in computational basis.
     #counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(reps)
     # Simulate the circuit and get the counts.
     qr_meas = qiskit.QuantumRegister(N_qubits+1, "q")
     cr_meas = qiskit.ClassicalRegister(N_qubits, "c")
     qc_meas = qiskit.QuantumCircuit(qr_meas, cr_meas)  

     for instr, qargs, cargs in qc_c:
          qc_meas.append(instr, qargs, cargs)

     for i in range(1, N_qubits+1):
          qc_meas.measure(qr_meas[i], cr_meas[i-1])            

     pm = generate_preset_pass_manager(optimization_level = 3)
     isa_circuit = pm.run(qc_meas)
     #isa_circuit = transpile(isa_circuit)
     result = sampler.run([isa_circuit], shots = 1).result()
     data_pub = result[0].data
     counts = data_pub.c.get_counts()

     results.append(counts)
     
# This section calculates the shadow density matrices using the inverse channel.        
shadows = []
for cliff, res in zip(cliffords, results):
     mat    = cliff.adjoint().to_matrix()
     for bit,count in res.items():
          Ub = mat[:,int(bit,2)] # this is Udag|b>
          shadows.append(Minv(N_qubits,np.outer(Ub,Ub.conj()))*count)

#rho_shadow = np.sum(shadows,axis=0)/(nShadows*reps)

return shadows

def linear_function_prediction(N, K, operator_linear, list_of_shadows):

     """ 
     This function calculates the linear function prediction of the operator_linear given the list of shadows.
     The total number of shadows is N*K. The function returns the median of the K means as described in
     https://arxiv.org/abs/2002.08953
     """

     list_of_means = []
     operator_linear_sparse = csr_matrix(operator_linear)
     
     # Calculating K means.
     for k in range(1,K+1):
          shadows_mean = 0.0
          for j in range(N*(k-1)+1,N*k+1):
               rho_j_sparse = csr_matrix(list_of_shadows[j-1])
               shadows_mean += trace(operator_linear_sparse @ rho_j_sparse).real
               
          list_of_means.append(shadows_mean/N)
          
     # Calculating the median of K means.
     return np.median(list_of_means)

def quadratic_function_prediction(N, K, operator_m, operator_n, list_of_shadows):

     list_of_means = []

     SWAP = csr_matrix(np.matrix([[1,0,0,0],
                                   [0,0,1,0],
                                   [0,1,0,0],
                                   [0,0,0,1]]))

     # While calculating the operator O the order of operator_m and operator_n is irrelevant.
     O_quadratic = SWAP @ kron(csr_matrix(operator_m), csr_matrix(operator_n))

     # Calculating K means
     for k in range(1, K + 1):
          shadows_mean = 0.0        
          for j in range(N * (k - 1) + 1, N * k + 1):
               for l in range(N * (k - 1) + 1, N * k + 1):
                    if j != l:
                         rho_j_sparse = csr_matrix(list_of_shadows[j - 1])
                         rho_l_sparse = csr_matrix(list_of_shadows[l - 1])
                         #shadows_mean += trace(O_quadratic @ kron(rho_j_sparse, rho_l_sparse)).real
                         shadows_mean += (O_quadratic @ kron(rho_j_sparse, rho_l_sparse)).diagonal().sum().real
                         
          list_of_means.append(shadows_mean / (N * (N - 1)))
     
     # Calculating their median
     return np.median(list_of_means)  

# %%
# VQA circuit.
from qiskit import QuantumCircuit, transpile

def vqa_circuit(theta_x, theta_y, theta_z, phi):
     vqa_circuit = QuantumCircuit(2)
     vqa_circuit.rx(theta_x, 0)
     vqa_circuit.ry(theta_y, 0)
     vqa_circuit.rz(theta_z, 0)
     vqa_circuit.cry(phi, 0, 1)
     return vqa_circuit

def anstaz_circuit(angles_lst, number_of_layers):  

     number_of_angles_per_layer = 4

     if len(angles_lst*number_of_angles_per_layer) % number_of_layers != 0:
          raise ValueError("The number of angles should be divisible by the number of layers.")
     else:
          n_qubits = 1
          anstaz_circuit = QuantumCircuit(n_qubits+1)
          for i in range(number_of_layers):
               anstaz_circuit.rx(angles_lst[i][0], 0)
               anstaz_circuit.ry(angles_lst[i][1], 0)
               anstaz_circuit.rz(angles_lst[i][2], 0)
               anstaz_circuit.cry(angles_lst[i][3],0,1)  
               anstaz_circuit.reset(0)
     return anstaz_circuit     

# %%
# Drawing the circuit for visualization.
trans_vqa = transpile(anstaz_circuit([(0.1,0.2,0.3,0.4), (0.1,0.2,0.3,0.4)],2), basis_gates = ["rx", "ry", "rz", "cx"])
#trans_vqa.draw("mpl", scale=1.5)

# %%
# Shadow tomography.

# Determining the number of shadows required.
# number_of_functions_to_predict = 6
# epsilon = 0.1
# delta = 0.8

# N = 2*np.log(2*number_of_functions_to_predict/delta)
# K = 34/epsilon**2
# n_Shadows = int(N*K)

N = 100
K = 100
n_Shadows = N*K
np.save("n_Shadows.npy", n_Shadows)

# %%
#st_instance = shadow_tomography_clifford(1, vqa_circuit(0.1,0.2,0.3,0.4), reps = 1)

# %%
#quadratic_function_prediction(N, K, np.array([[1,0],[0,-1]]), np.array([[1,0],[0,-1]]), st_instance)

# %%
I = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

# %%
def cost_function(b, gamma_1, gamma_2, rho_shadow_lst):

     # Coefficients in the cost function.
     f00 = b**2/2 + 5*gamma_1**2/8 + gamma_1*gamma_2 + 2*gamma_2**2
     f02 = - b*gamma_1
     f03 = -gamma_1**2
     f11 = -b**2/2
     f33 = 3*gamma_1**2/8 - gamma_1*gamma_2 - 2*gamma_2**2
     f23 = b*gamma_1/2 - 2*gamma_2*b

     return (f00*quadratic_function_prediction(N,K,I,I,rho_shadow_lst) + 
     f02*quadratic_function_prediction(N,K,I,sigma_y,rho_shadow_lst) +  
     f03*quadratic_function_prediction(N,K,I,sigma_z,rho_shadow_lst) + 
     f11*quadratic_function_prediction(N,K,sigma_x,sigma_x,rho_shadow_lst) + 
     f33*quadratic_function_prediction(N,K,sigma_z,sigma_z,rho_shadow_lst) + 
     f23*quadratic_function_prediction(N,K,sigma_y,sigma_z,rho_shadow_lst))


# %%
tolerance_for_convergence = 1.e-2
theta_x = np.random.uniform(-np.pi, np.pi)
theta_y = np.random.uniform(-np.pi, np.pi)
theta_z = np.random.uniform(-np.pi, np.pi)
phi = np.random.uniform(-np.pi, np.pi)

#print("Initial set of parameters =", theta_x, theta_y, theta_z, phi)
np.save("initial_params.npy", [theta_x, theta_y, theta_z, phi])

initial_learning_rate = 2.0
number_of_iterations = 0
max_iterations = 100

best_cost = float("inf")
best_theta_x, best_theta_y, best_theta_z, best_phi = theta_x, theta_y, theta_z, phi
best_iteration = 0

b = 0.5
gamma_1 = 2
gamma_2 = 3


while number_of_iterations < max_iterations:

     num_layers = 1
     st_instance = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y,theta_z,phi)],num_layers))
     cost_value = np.real(cost_function(b, gamma_1, gamma_2, st_instance))

     # Update best parameters if a new minimum cost is found
     if cost_value < best_cost:
          best_cost = cost_value
          best_theta_x, best_theta_y, best_theta_z, best_phi = theta_x, theta_y, theta_z, phi
          best_iteration = number_of_iterations  # Update iteration number for best cost    

     if cost_value <= tolerance_for_convergence:
          np.save("converged_cost_function.npy", [theta_x, theta_y, theta_z, phi])
          np.save("converged_best_cost.npy", best_cost)
          break

     # Decaying learning rate
     learning_rate = initial_learning_rate / np.sqrt(number_of_iterations + 1)

     # Parameter shift rule for gradient calculation

     st_instance_p = shadow_tomography_clifford(1,anstaz_circuit([(theta_x + np.pi/2,theta_y,theta_z,phi)],num_layers))
     st_instance_n = shadow_tomography_clifford(1,anstaz_circuit([(theta_x - np.pi/2,theta_y,theta_z,phi)],num_layers))
     gradient_theta_x = 0.5 * (np.real(cost_function(b, gamma_1, gamma_2, st_instance_p)) - np.real(cost_function(b, gamma_1, gamma_2, st_instance_n)))

     st_instance_p = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y + np.pi/2,theta_z,phi)],num_layers))
     st_instance_n = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y - np.pi/2,theta_z,phi)],num_layers))
     gradient_theta_y = 0.5 * (np.real(cost_function(b, gamma_1, gamma_2, st_instance_p)) - np.real(cost_function(b, gamma_1, gamma_2, st_instance_n)))

     st_instance_p = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y,theta_z + np.pi/2,phi)],num_layers))
     st_instance_n = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y,theta_z - np.pi/2,phi)],num_layers))
     gradient_theta_z = 0.5 * (np.real(cost_function(b, gamma_1, gamma_2, st_instance_p)) - np.real(cost_function(b, gamma_1, gamma_2, st_instance_n)))

     st_instance_p = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y,theta_z,phi + np.pi/2)],num_layers))
     st_instance_n = shadow_tomography_clifford(1,anstaz_circuit([(theta_x,theta_y,theta_z,phi - np.pi/2)],num_layers))
     gradient_phi = 0.5 * (np.real(cost_function(b, gamma_1, gamma_2, st_instance_p)) - np.real(cost_function(b, gamma_1, gamma_2, st_instance_n)))

     # Parameter update
     theta_x -= learning_rate * gradient_theta_x
     theta_y -= learning_rate * gradient_theta_y
     theta_z -= learning_rate * gradient_theta_z
     phi -= learning_rate * gradient_phi

     number_of_iterations += 1
     #print("Cost function =", cost_value)
     #print("Number of iterations =", number_of_iterations)
     #print("Learning rate =", learning_rate)

if number_of_iterations == max_iterations:
     np.save("max_iter_cost_function.npy", [theta_x, theta_y, theta_z, phi])
     np.save("max_iter_best_cost.npy", best_cost)
     np.save("max_iter_best_params.npy", [best_theta_x, best_theta_y, best_theta_z, best_phi])


"""
# %% [markdown]
# ## Adam optimizer

# %%
# Initialization of tolerance, parameters, and learning rate
tolerance_for_convergence = 1.e-2
theta_x, theta_y, theta_z, phi = [np.random.uniform(-np.pi, np.pi) for _ in range(4)]
print("Initial parameters =", theta_x, theta_y, theta_z, phi)

initial_learning_rate = 1.0
number_of_iterations = 0
max_iterations = 100
beta1, beta2, epsilon = 0.9, 0.999, 1e-8
m, v = [0]*4, [0]*4
max_gradient = 1.0  # Gradient clipping threshold

# Track the best parameters and the corresponding iteration number
best_cost = float("inf")
best_theta_x, best_theta_y, best_theta_z, best_phi = theta_x, theta_y, theta_z, phi
best_iteration = 0

while number_of_iterations < max_iterations:
     num_layers = 1
     st_instance = shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y, theta_z, phi)], num_layers))
     cost_value = np.real(cost_function(b, gamma_1, gamma_2, st_instance))
     
     # Update best parameters if a new minimum cost is found
     if cost_value < best_cost:
          best_cost = cost_value
          best_theta_x, best_theta_y, best_theta_z, best_phi = theta_x, theta_y, theta_z, phi
          best_iteration = number_of_iterations  # Update iteration number for best cost
     
     # Check for convergence
     if cost_value <= tolerance_for_convergence:
          print("\n[CONVERGED] Cost function =", cost_value)
          print("Optimized parameters =", theta_x, theta_y, theta_z, phi)
          print("Convergence reached at iteration =", number_of_iterations)
          break
     
     # Decaying learning rate for Adam
     learning_rate = initial_learning_rate / np.sqrt(number_of_iterations + 1)
     
     # Calculate gradients using parameter shift
     gradients = [
          0.5 * (np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x + np.pi/2, theta_y, theta_z, phi)], num_layers)))) - 
               np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x - np.pi/2, theta_y, theta_z, phi)], num_layers))))),
          0.5 * (np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y + np.pi/2, theta_z, phi)], num_layers)))) - 
               np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y - np.pi/2, theta_z, phi)], num_layers))))),
          0.5 * (np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y, theta_z + np.pi/2, phi)], num_layers)))) - 
               np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y, theta_z - np.pi/2, phi)], num_layers))))),
          0.5 * (np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y, theta_z, phi + np.pi/2)], num_layers)))) - 
               np.real(cost_function(b, gamma_1, gamma_2, shadow_tomography_clifford(1, anstaz_circuit([(theta_x, theta_y, theta_z, phi - np.pi/2)], num_layers)))))
     ]
     
     # Adam update with gradient clipping
     for i, grad in enumerate(gradients):
          grad = np.clip(grad, -max_gradient, max_gradient)  # Clipping
          m[i] = beta1 * m[i] + (1 - beta1) * grad
          v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
          m_hat = m[i] / (1 - beta1 ** (number_of_iterations + 1))
          v_hat = v[i] / (1 - beta2 ** (number_of_iterations + 1))
          
          if i == 0:
               theta_x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
          elif i == 1:
               theta_y -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
          elif i == 2:
               theta_z -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
          elif i == 3:
               phi -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
     
     number_of_iterations += 1
     print(f"Iteration {number_of_iterations}: Cost =", cost_value)

# After the loop, print the best found values and corresponding iteration number
print("\n[RESULT] Minimum cost encountered =", best_cost)
print("Parameters corresponding to minimum cost =", best_theta_x, best_theta_y, best_theta_z, best_phi)
print("Iteration number for minimum cost =", best_iteration)"""