# PennyLane libraries for quantum machine learning.
import pennylane as qml
import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer

dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev)
def qcircuit_DR(params, x, y, entanglement = True):
    """A variational quantum circuit representing the Universal classifier.

    Args:
        params (array[float]): array of parameters
        x (array[float]): single input vector
        y (array[float]): single output state density matrix

    Returns:
        float: fidelity between output state and input
    """

    size_list = list(range(0,int(np.floor(len(x)/3.0))))
    #print("Number of parameters: ", size)
    #print("Parameters: ", params)
    qubits = 4
    # Parameters of each layer
    for p in params:
        # Rotation of features
        for i in size_list:
            start = i*3
            end = start + 3
            sub_x = x[start:end]
            qml.Rot(*sub_x, wires=0)
            qml.Rot(*sub_x, wires=1)
            qml.Rot(*sub_x, wires=2)
            qml.Rot(*sub_x, wires=3)
        # Rotation of the parameters
        for q in range(qubits):
            param_qubits = p[(q*3):((q+1)*3)]
            qml.Rot(*param_qubits, wires=q)
        # Entanglement of the qubits
        if entanglement:
            qml.CZ(wires=[0,1])
            qml.CZ(wires=[2,3])
            qml.CZ(wires=[1,2])
            qml.CZ(wires=[3,0])
    fidelity = qml.expval(qml.PauliZ(0))
    return fidelity


