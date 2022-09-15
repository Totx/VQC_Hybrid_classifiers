"""_summary_

    Returns:
        _type_: _description_
"""

import pennylane as qml # PennyLane library for quantum machine learning
import numpy as np
import tensorflow as tf

# Definition of a quantum system using 5 qubits
dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev, interface='tf')
def qcircuit_data_reuploading(qcircuit_parameters, x_data, layers, n_qubits):
    """Quantum variation classifier using the data reuploading technique.

    Args:
        params (array[float]): Tuning parameters of the quantum variational classifier.
        x_data (array[float]): Features of a single row of the dataset.
        n_qubits (int): Number of qubits of the quantum circuit to be used.
        entanglement (bool, optional): Whether entanglement should be incorporated via CNOT gates.
        Defaults to True.

    Returns:
        float: The expectation value of the selected operator in the final quantum state of the
        first qubit.
    """
    # Creates a list containing the amount of rotation given the size of the feature space
    size_list = list(range(0,int(np.floor(len(x_data)/3.0))))
    # print("Number of parameters: ", size)
    # print("Parameters: ", params)
    qubits = n_qubits
    # Parameters of each layer
    for layer in range(layers):
        # Rotation of features
        for i in size_list:
            # Features are taken is sets of three
            start = i*3
            end = start + 3
            sub_x = x_data[start:end]
            for qubit in range(n_qubits):
                qml.Rot(*sub_x, wires=qubit)
        # Rotation of the parameters.
        for qubit in range(qubits):
            # Select the 3 parameters that are going to be used in the rotation in each qubit
            phi, theta, omega = qcircuit_parameters[layer, qubit]
            qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
        # Entanglement of the qubits
        for control_qubit in range(n_qubits):
            target_qubit = (control_qubit + 1) % n_qubits
            qml.CNOT(wires=[control_qubit, target_qubit])
    # The expectation value of the sigma_z operator is calculated.
    expectation_value = qml.expval(qml.PauliZ(0))
    return expectation_value

@qml.qnode(dev, interface='tf')
def qcircuit_amplitude_embedding(qcircuit_parameters, x_data, layers, n_qubits):
    
    # Amplitude embedding using the input data normalized and padded with 0.
    # n_qubits = math.log2(len(x_data))
    # layers = params.shape[0]
    qml.templates.AmplitudeEmbedding([x_i for x_i in x_data], wires=range(n_qubits),
                                     pad_with = 0.,  normalize = True)
    #qml.AmplitudeEmbedding(features=x_data, wires=range(n_qubits), pad_with=0.0, normalize=True)
    for layer in range(layers):
        for qubit in range(n_qubits):
            phi, theta, omega = qcircuit_parameters[layer, qubit]
            qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
        for control_qubit in range(n_qubits):
            target_qubit = (control_qubit + 1) % n_qubits
            qml.CNOT(wires=[control_qubit, target_qubit])
    expectation_value = qml.expval(qml.PauliZ(0))
    return expectation_value

@qml.qnode(dev, interface='tf')
def qcircuit_data_reuploading_kl(inputs, qcircuit_parameters):
    """Quantum variation classifier using the data reuploading technique.

    Args:
        params (array[float]): Tuning parameters of the quantum variational classifier.
        x_data (array[float]): Features of a single row of the dataset.
        n_qubits (int): Number of qubits of the quantum circuit to be used.
        entanglement (bool, optional): Whether entanglement should be incorporated via CNOT gates.
        Defaults to True.

    Returns:
        float: The expectation value of the selected operator in the final quantum state of the
        first qubit.
    """
    
    padding = tf.zeros((3-(inputs.shape[0] % 3)) % 3)
    organized_inputs = tf.reshape(tf.concat([inputs, padding], 0), shape=[-1,3])
    layers, n_qubits, _ = tf.shape(qcircuit_parameters)
    # Parameters of each layer
    for layer in range(layers):
        # Rotation of the parameters.
        for qubit in range(n_qubits):
            for phi, theta, omega in organized_inputs:
                qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
            # Select the 3 parameters that are going to be used in the rotation in each qubit
            phi, theta, omega = qcircuit_parameters[layer, qubit]
            qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
        # Entanglement of the qubits
        for control_qubit in range(n_qubits):
            target_qubit = (control_qubit + 1) % int(n_qubits)
            qml.CNOT(wires=[control_qubit, target_qubit])
    # The expectation value of the sigma_z operator is calculated.
    expectation_value = [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
    return expectation_value

@qml.qnode(dev, interface='tf')
def qcircuit_amplitude_embedding_kl(inputs, qcircuit_parameters):
    
    # Amplitude embedding using the input data normalized and padded with 0.
    # n_qubits = math.log2(len(x_data))
    # layers = params.shape[0]
    layers, n_qubits, _ = tf.shape(qcircuit_parameters)
    # print(tf.shape(qcircuit_parameters))
    # print(qcircuit_parameters)
    qml.templates.AmplitudeEmbedding([x_i for x_i in inputs], wires=range(n_qubits),
                                     pad_with = 0.,  normalize = True)
    #qml.AmplitudeEmbedding(features=x_data, wires=range(n_qubits), pad_with=0.0, normalize=True)
    for layer in range(layers):
        for qubit in range(n_qubits):
            phi, theta, omega = qcircuit_parameters[layer, qubit]
            qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
        for control_qubit in range(n_qubits):
            target_qubit = (control_qubit + 1) % int(n_qubits)
            qml.CNOT(wires=[control_qubit, target_qubit])
    expectation_value = [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
    return expectation_value

@qml.qnode(dev)
def qcircuit_amplitude_embedding_kl_hybrid(inputs, qcircuit_parameters):
    
    _, n_qubits, _ = tf.shape(qcircuit_parameters)
    qml.AmplitudeEmbedding(features=[x_i for x_i in inputs], wires=range(n_qubits), pad_with = 0.0,  normalize = True)
    qml.StronglyEntanglingLayers(weights=qcircuit_parameters, wires=range(n_qubits))
    expectation_value = [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
    return expectation_value

@qml.qnode(dev)
def qcircuit_angle_embedding_kl(inputs, qcircuit_parameters):
    
    layers, n_qubits, _ = tf.shape(qcircuit_parameters)
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
    # qml.StronglyEntanglingLayers(weights=qcircuit_parameters, wires=range(n_qubits))
    for layer in range(layers):
        for qubit in range(n_qubits):
            phi, theta, omega = qcircuit_parameters[layer, qubit]
            qml.Rot(phi=phi, theta=theta, omega=omega, wires=qubit)
        for control_qubit in range(n_qubits):
            target_qubit = (control_qubit + 1) % int(n_qubits)
            qml.CNOT(wires=[control_qubit, target_qubit])
    expectation_value = [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
    return expectation_value

@qml.qnode(dev)
def qcircuit_angle_embedding_kl_hybrid(inputs, qcircuit_parameters):
    
    _, n_qubits, _ = tf.shape(qcircuit_parameters)
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights=qcircuit_parameters, wires=range(n_qubits))
    expectation_value = [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
    return expectation_value

def cost_function_qcircuit(qcircuit_parameters, x_dataset, y_dataset, layers, n_qubits, qcircuit_type = 'AE'):
    
    total_cost = 0
    if qcircuit_type == 'AE':
        qcircuit_function = qcircuit_amplitude_embedding
    elif qcircuit_type == 'DR':
        qcircuit_function = qcircuit_data_reuploading
    for i in range(x_dataset.shape[0]):
        expectation_value = qcircuit_function(qcircuit_parameters, x_data=x_dataset[i],
                                              layers=layers, n_qubits=n_qubits)
        total_cost = total_cost + (y_dataset[i] - expectation_value)**2
    print(x_dataset.shape[0])
    return total_cost / x_dataset.shape[0]
