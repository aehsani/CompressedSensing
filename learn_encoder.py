import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from projection import ProjectPD
from reconstruction import SmoothGreedy

class GreedyCS:
    def __init__(self, data, n, k):
        self.data = data
        self.k = k
        self.d, self.points = data.shape
        tf.random.set_seed(5)
        self.A = tf.random.normal(
            [n, d], stddev=1/math.sqrt(self.n)
        )
         
    def learn_matrix(self, learn_params=dict()):
        learning_rate = learn_params.get(learning_rate, 0.1)
        epochs = learn_params.get(epochs, 500)
        A_tf = tf.Variable(self.A)
        for i in range(epochs):
            with tf.GradientTape() as tape:
                l2_loss = tf.math.square(
                    tf.norm(A_tf, ord='euclidean'
                )

        
    def compute_gradient
                
                
                
        
    
    def recover_estimate(self, x):
        pass

if __name__ == "__main__":
    pass
    # data = None
    # gcs = GreedyCS(data)
    # gcs.learn_matrix(self)
