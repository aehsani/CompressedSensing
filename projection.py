import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_PD(A, bound=30, T=25):
    m, d = A.shape
    for i in range(m):
        v_i = np.random.uniform(-bound, bound, d)
        # power method - change limits of loop
        for t in range(T):
            v_i = 

        
