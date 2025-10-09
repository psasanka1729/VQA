# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""An algorithm to implement a Trotterization real time-evolution."""

from __future__ import annotations

"""
MODIFICATION NOTICE: This code is a modified version of the original code from Qiskit.
The original code is available at https://github.com/qiskit-community/qiskit-algorithms/blob/main/qiskit_algorithms/time_evolvers/trotterization/trotter_qrte.py. This modification includes the implementation of the Lindblad operators to simulate open quantum systems.
The modified code is denoted by comments labeled [SD]. These comments indicate where
changes were made to the original source code to incorporate the Lindblad operators.
The modification of the source code is necessary because the Lindblad operators need to
be applied at each Trotter step, which cannot be achieved by simply using the existing Trotter
circuit and modifying it later. Instead, the Lindblad operators must be integrated into
the circuit at each time step in order to accurately model the non-unitary dynamics
of the open quantum system.
"""
# Modified by: Sasanka Dowarah, Email: Sasanka.Dowarah@utdallas.edu.

import math
from qiskit import QuantumCircuit, transpile
# Qiskit algorithm is moved to qiskit_algorithms.
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parametertable import ParameterView
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.synthesis import ProductFormula, LieTrotter

from qiskit_algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.time_evolution_result import TimeEvolutionResult
from qiskit_algorithms.time_evolvers.real_time_evolver import RealTimeEvolver
from qiskit_algorithms.observables_evaluator import estimate_observables


class TrotterQRTE(RealTimeEvolver):
    """Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ``ProductFormula`` provided.

    Examples:

        .. code-block:: python

            from qiskit.opflow import PauliSumOp
            from qiskit.quantum_info import Pauli, SparsePauliOp
            from qiskit import QuantumCircuit
            from qiskit.algorithms import TimeEvolutionProblem
            from qiskit.algorithms.time_evolvers import TrotterQRTE
            from qiskit.primitives import Estimator

            operator = PauliSumOp(SparsePauliOp([Pauli("X"), Pauli("Z")]))
            initial_state = QuantumCircuit(1)
            time = 1
            evolution_problem = TimeEvolutionProblem(operator, time, initial_state)
            # LieTrotter with 1 rep
            estimator = Estimator()
            trotter_qrte = TrotterQRTE(estimator=estimator)
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    def __init__(
        self,
        product_formula: ProductFormula | None = None,
        estimator: BaseEstimator | None = None,
        num_timesteps: int = 1,
        # [SD] Additional parameters for the Lindblad operators.
        LUMO_qubit_index: int = 1,
        gamma_out: float = 0,
        gamma_in: float = 0,
    
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. If ``None`` provided, the
                Lie-Trotter first order product formula with a single repetition is used. ``reps``
                should be 1 to obtain a number of time-steps equal to ``num_timesteps`` and an
                evaluation of :attr:`.TimeEvolutionProblem.aux_operators` at every time-step. If ``reps``
                is larger than 1, the true number of time-steps will be ``num_timesteps * reps``.
            num_timesteps: The number of time-steps the full evolution time is devided into
                (repetitions of ``product_formula``)
            estimator: An estimator primitive used for calculating expectation values of
                ``TimeEvolutionProblem.aux_operators``.  
            LUMO_qubit_index: The index of the qubit corresponding to the LUMO.    
            gamma_out: coefficient of the Lindblad operators corresponding to L_out.
            gamma_in: coefficient of the Lindblad operators corresponding to L_in.

        """

        self.product_formula = product_formula
        self.num_timesteps = num_timesteps
        # [SD] Additional parameters for the Lindblad operators.
        self.LUMO_qubit_index = LUMO_qubit_index
        self.gamma_out = gamma_out
        self.gamma_in = gamma_in

        self.estimator = estimator

    @property
    def product_formula(self) -> ProductFormula:
        """Returns a product formula."""
        return self._product_formula

    @product_formula.setter
    def product_formula(self, product_formula: ProductFormula | None):
        """Sets a product formula. If ``None`` provided, sets the Lie-Trotter first order product
        formula with a single repetition."""
        if product_formula is None:
            product_formula = LieTrotter()
        self._product_formula = product_formula

    @property
    def estimator(self) -> BaseEstimator | None:
        """
        Returns an estimator.
        """
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator) -> None:
        """
        Sets an estimator.
        """
        self._estimator = estimator

    @property
    def num_timesteps(self) -> int:
        """Returns the number of timesteps."""
        return self._num_timesteps


    @num_timesteps.setter
    def num_timesteps(self, num_timesteps: int) -> None:
        """
        Sets the number of time-steps.

        Raises:
            ValueError: If num_timesteps is not positive.
        """
        if num_timesteps <= 0:
            raise ValueError(
                f"Number of time steps must be positive integer, {num_timesteps} provided"
            )
        self._num_timesteps = num_timesteps

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """
        Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            ``True`` if ``aux_operators`` expectations in the ``TimeEvolutionProblem`` can be
            evaluated, ``False`` otherwise.
        """
        return True       

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:
        """
        Evolves a quantum state for a given time using the Trotterization method
        based on a product formula provided. The result is provided in the form of a quantum
        circuit. If auxiliary operators are included in the ``evolution_problem``, they are
        evaluated on the ``init_state`` and on the evolved state at every step (``num_timesteps``
        times) using an estimator primitive provided.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``Pauli`` or ``PauliSumOp`` are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state as a quantum circuit and, optionally,
            auxiliary operators evaluated for a resulting state on an estimator primitive.

        Raises:
            ValueError: If ``t_param`` is not set to ``None`` in the ``TimeEvolutionProblem``
                (feature not currently supported).
            ValueError: If ``aux_operators`` provided in the time evolution problem but no estimator
                provided to the algorithm.
            ValueError: If the ``initial_state`` is not provided in the ``TimeEvolutionProblem``.
            ValueError: If an unsupported Hamiltonian type is provided.
        """
        #evolution_problem.validate_params()

        if evolution_problem.aux_operators is not None and self.estimator is None:
            raise ValueError(
                "The time evolution problem contained ``aux_operators`` but no estimator was "
                "provided. The algorithm continues without calculating these quantities. "
            )

        # ensure the hamiltonian is a sparse pauli op
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (Pauli, SparsePauliOp)):
            raise ValueError(
                f"TrotterQRTE only accepts Pauli | PauliSumOp | SparsePauliOp, {type(hamiltonian)} "
                "provided."
            )

        if isinstance(hamiltonian, Pauli):
            hamiltonian = SparsePauliOp(hamiltonian)

        t_param = evolution_problem.t_param
        free_parameters = hamiltonian.parameters
        if t_param is not None and free_parameters != ParameterView([t_param]):
            raise ValueError(
                f"Hamiltonian time parameters ({free_parameters}) do not match "
                f"evolution_problem.t_param ({t_param})."
            )

        # make sure PauliEvolutionGate does not implement more than one Trotter step
        dt = evolution_problem.time / self.num_timesteps

        """
        Sasanka: Here we will create a quantum circuit with number of qubits = L+1,
        with one ancilla qubit and call it "open_system_circuit".
        After each step in the trotterization; we will append the trotterized
        circuit produced by Qiskit to "open_system_circuit" and then the two
        Lindblad operators and repeat the process "num_timesteps" times.

        """

        if evolution_problem.initial_state is not None:
            initial_state = evolution_problem.initial_state
        else:
            raise ValueError("``initial_state`` must be provided in the ``TimeEvolutionProblem``.")

        """
        [SD] We will create a quantum circuit with number of qubits = L+1, where the zeroth qubit
        is the ancilla qubit and the rest L qubits are the qubits of the system. The Troteerized
        circuit generated by Qiskit will be appended to the 1 to L qubits of the created cicuit.
        """
        open_system_circuit = QuantumCircuit(initial_state.num_qubits+1)
        evolved_state = QuantumCircuit(initial_state.num_qubits)
        evolved_state.append(initial_state, evolved_state.qubits)

        # [SD] Append the initial state to the open_system_circuit.
        # [SD] We do not need to initialize the ancilla qubit since it is already in the |0> state.
        open_system_circuit.append(evolved_state, open_system_circuit.qubits[1:initial_state.num_qubits+1])

        if evolution_problem.aux_operators is not None:
            observables = []
            observables.append(
                estimate_observables(
                    self.estimator,
                    evolved_state,
                    evolution_problem.aux_operators,
                    None,
                    evolution_problem.truncation_threshold,
                )
            )
        else:
            observables = None
        #

        if t_param is None:
            # the evolution gate
            single_step_evolution_gate = PauliEvolutionGate(
                hamiltonian, dt, synthesis=self.product_formula
            )

        for n in range(self.num_timesteps):
            # if hamiltonian is time-dependent, bind new time-value at every step to construct
            # evolution for next step
            if t_param is not None:
                time_value = (n + 1) * dt
                bound_hamiltonian = hamiltonian.assign_parameters([time_value])
                single_step_evolution_gate = PauliEvolutionGate(
                    bound_hamiltonian,
                    dt,
                    synthesis=self.product_formula,
                )
                
            # [SD] Append the trotterized circuit to the open_system_circuit.
            open_system_circuit.append(single_step_evolution_gate, open_system_circuit.qubits[1:initial_state.num_qubits+1])

            """
            [SD] In this part we will create the circuit for the Lindblad operator corresponding to L_out.
            """
            # if math.sqrt(dt * self.gamma_out) <= 1:
            #     theta_out = 2*math.asin(math.sqrt(dt * self.gamma_out))
            # else:
            #     raise ValueError(" dt* gamma_out must be <= 1. Got = ", math.sqrt(dt*self.gamma_out))

            theta_out = self.gamma_out

            open_system_circuit.cry(theta_out, self.LUMO_qubit_index + 1 , 0)

            # [SD] ZZZ.....Z string part in the Lindblad operator.
            # [SD] If LUMO is the zeroth qubit then no need to apply the ZZZ...Z string part.

            if self.LUMO_qubit_index == 0:
                pass
            else:
                for i in range(1,self.LUMO_qubit_index):
                    open_system_circuit.cx(i,i+1)
                open_system_circuit.cz(self.LUMO_qubit_index,0)
                for i in range(self.LUMO_qubit_index,1,-1):
                    open_system_circuit.cx(i-1,i)

            open_system_circuit.cx(0,self.LUMO_qubit_index+1)
            open_system_circuit.reset(0)


            """
            [SD] In this part we will create the circuit for the Lindblad operator corresponding to L_in.
            """
            # if math.sqrt(dt * self.gamma_in) <= 1:
            #     theta_in = 2*math.asin(math.sqrt(dt * self.gamma_in))
            # else:          
            #     raise ValueError(" dt* gamma_in must be <= 1. Got = ", math.sqrt(dt*self.gamma_in))

            theta_in = self.gamma_in

            open_system_circuit.x(self.LUMO_qubit_index + 1)
            open_system_circuit.cry(theta_in,self.LUMO_qubit_index + 1,0)

            # [SD] ZZZ.....Z string part in the Lindblad operator.
            # [SD] If LUMO is the zeroth qubit then no need to apply the ZZZ...Z string part.
            if self.LUMO_qubit_index == 0:
                pass
            else:
                for i in range(1,self.LUMO_qubit_index):
                    open_system_circuit.cx(i,i+1)
                open_system_circuit.cz(self.LUMO_qubit_index,0)
                for i in range(self.LUMO_qubit_index,1,-1):
                    open_system_circuit.cx(i-1,i) 

            open_system_circuit.cx(0,self.LUMO_qubit_index + 1)    
            open_system_circuit.x(self.LUMO_qubit_index + 1) 
            
            open_system_circuit.reset(0)       



            if evolution_problem.aux_operators is not None:
                observables.append(
                    estimate_observables(
                        self.estimator,
                        evolved_state,
                        evolution_problem.aux_operators,
                        None,
                        evolution_problem.truncation_threshold,
                    )
                )
            #
        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = observables[-1]

        # [SD] Return the open_system_circuit.
        evolved_state = open_system_circuit

        return TimeEvolutionResult(open_system_circuit, evaluated_aux_ops, observables)