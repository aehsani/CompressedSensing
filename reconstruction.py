import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import entr
from scipy import sparse
import gurobipy as gp

class SmoothGreedy:
    def __init__(self, A, Y, k, data=None, eps = 0.1):
        self.k = k
        self.d, self.points = self.data.shape
        self.current_objective_val = 0
        self.A = A
        self.Y = Y
        self.eps = eps
        self.data = data
        self.reconstruction

        # self.reconstruction = np.zeros_like(A)
    def evaluate(self):
        if not data:
            raise ValueError("No data to evaluate error")
    
    def compute_reconstruction(self):
        for i in range(self.points):
            self.recontruction[:,i] = self.compute_estimate_vector(i)

    def compute_best_indices(self, i):
        current_indices = set()
        remaining = set(range(d))
        measurement = y[:,i]
        l0 = 1/2*np.inner(measurement, measurement)
        current_obj_val = 0
        best_x = None

        for _ in range(k):
            r = len(remaining)
            g = dict()
            objective = dict()
            for j in remaining:
                current_indices.add(j)

                best_x, loss = self.find_best_x(current_indices)
                objective[j] = l0 - loss
                g[j] = objective[j] - current_obj_val

                current_indices.remove(j)
            best_j = self.choose_index(g)
            remaining.remove(best_j)
            current_indices.add(best_j)
            current_obj_val = objective[best_j]
        return current_indices
        

                
    def choose_index(self, g):
        g_vec = [value for key,value in sorted(attributes.items())]
        p_vec = self.get_p_quadratic(g_vec)
        chosen = np.random.choice(list(range(len(g_vec))), p=p_vec)
        return chosen

    def get_p_quadratic(self, g_vec):
        N = len(g_vec)
        model = gp.Model("qp")
        m_vars = []
        for i in range(N):
            y = m.addVar(lb=0, up=1, vtype=GRB.CONTINUOUS)
            m_vars.append(y)

        qexpr = gp.GRBQuadExpr()
        qexpr.addTerms([1.0 for _ in range(N)], y)
        model.addQConstr(qexpr, GRB.EQUAL, 1)

        objexpr = gp.GRBQuadExpr()
        objexpr.addTerms(g_vec, m_vars)
        objexpr.addTerms([-self.eps for _ in range(N)], m_vars, m_vars)
        model.setObjective(objexpr, GRB.MAXIMIZE)

        model.optimize()

        p_vec = [v.x for v in m_vars]
        return p_vec

    def get_p_entr(self, g_vec):
        p_vec = np.exp(g_vec/self.eps)
        p_vec = p_vec/p_vec.sum()
        return p_vec
                
    def find_best_x(self, indices):
        indices = sorted(indices)
        new_A = A[:indices]
        new_y = y[:i]
        x_dict = np.linalg.lstsq(new_A, new_y)
        return x_dict[0], x_dict[1].sum()/2

if __name__ == "__main__":
    data = sparse.load_npz('data/data_1.npz')
    A = np.loadtxt('data/m_matrix_1', delimiter=',')

    print('data shape')
    print(data.shape)
    print('measurment matrix shape')
    print(A.shape)
    Y = data.dot(A)
    print("Y")
    print(Y.shape)

    sg = SmoothGreedy

    
