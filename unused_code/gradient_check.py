# gradient_check.py

""" This script checks to make sure the calculated gradient for the neural net are correct  """

import numpy as np



# this fuction checks the gradient of B
def check_B_gradient(B, A, label, x, Y_hat, hidden):
    print("**Checking B gradient")
    
    gradient = np.dot(np.subtract(Y_hat.T, label).T, hidden.T)
    
    eps = 0.0001
    
    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            
            # Copy the parameter matrix and change the current parameter slightly
            B_matrix_min = B.copy()
            B_matrix_min[row,col] -= eps
            B_matrix_plus = B.copy()
            B_matrix_plus[row,col] += eps
            
            # Compute the numerical gradient
            grad_num = (loss_function(x, A, B_matrix_plus, label) - loss_function(x, A, B_matrix_min, label))/(2*eps)
            
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, gradient[row,col]):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(gradient[row,col])))
            
    print('No B gradient errors found')


# this fuction checks the gradient of A
def check_A_gradient(B, A, label, x, Y_hat):
    print("**Checking A gradient")

    first = np.dot(np.subtract(Y_hat.T, label), B)
    sec = x * (1.0/np.sum(x))
    gradient = sparse.csr_matrix.dot(first.T, sec)
   
    eps = 0.0001
    
    for row in range(A.shape[0]):
        print("row ", row)
        for col in range(A.shape[1]):
            # Copy the parameter matrix and change the current parameter slightly
            A_matrix_min = A.copy()
            A_matrix_min[row,col] -= eps
            A_matrix_plus = A.copy()
            A_matrix_plus[row,col] += eps
            
            # Compute the numerical gradient
            grad_num = (loss_function(x, A_matrix_plus, B, label) - loss_function(x, A_matrix_min, B, label))/(2*eps)
            
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, gradient[row,col]):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(gradient[row,col])))
            
    print('No A gradient errors found')
