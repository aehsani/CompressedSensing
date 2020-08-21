import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import entr
from scipy import sparse
import gurobipy as gp

class SmoothGreedy:
    def __init__(self, A, Y, data=None, k=12, eps=0.01):
        self.A = A
        self.Y = Y
        self.eps = eps
        self.data = data

        self.n = Y.shape[1] # number of vectors
        self.d = A.shape[1] # dimension of vectors
        self.m = A.shape[0] # dimension of measurements
        self.k = min(k, self.d) # cardinality constraint

        self.reconstruction = np.zeros((self.d, self.n))
        self.current_objective_val = 0
        self.compute_reconstruction()
        

    def evaluate_old(self):
        if data is None:
            raise ValueError("No data to evaluate error")
        unscaled = np.linalg.norm(self.data - self.reconstruction, ord='fro')
        return unscaled/np.sqrt(self.n)
    
    def compute_reconstruction(self):
        for i in range(self.n):
            if i % 50 == 0:
                print("Doing vector: ", i)
            entries, indices = self.compute_best_indices(i)
            self.reconstruction[indices,i] = entries

    def compute_best_indices_old(self, i):
        current_indices = set()
        remaining = set(range(self.d))
        measurement = Y[i,:]
        l0 = 1/2*np.inner(measurement, measurement)
        current_obj_val = 0
        best_x = None

        for _ in range(self.k):
            g = dict()
            objective = dict()
            for j in remaining:
                current_indices.add(j)

                best_x, loss = self.find_best_x(current_indices, measurement)
                objective[j] = l0 - loss
                g[j] = objective[j] - current_obj_val

                current_indices.remove(j)
            best_j = self.choose_index(g)
            remaining.remove(best_j)
            current_indices.add(best_j)
            current_obj_val = objective[best_j]

        best_x, _ = self.find_best_x(current_indices, measurement)
        return best_x, sorted(current_indices)

    def compute_best_indices(self, i):
        current_indices = set()
        pseudo_inv = np.zeros((self.k,self.m))
        A_sub = np.zeros((self.m,self.k))
        remaining = set(range(self.d))
        measurement = Y[:,i]
        l0 = 1/2*np.inner(measurement, measurement)
        current_obj_val = 0
        best_x = None
        chosen = []

        for _ in range(self.k):
            g = dict()
            objective = dict()

            for j in remaining:
                new_A_sub, new_pseudo = self.update_matrices(j, chosen, A_sub, pseudo_inv)
                best_x, loss = self.r1_find_best_x(j, current_indices, measurement, A_sub, psuedo_inv)
                objective[j] = l0 - loss
                g[j] = objective[j] - current_obj_val

            best_j = self.choose_index(g)
            remaining.remove(best_j)
            current_indices.add(best_j)
            current_obj_val = objective[best_j]

        best_x, _ = self.find_best_x(current_indices, measurement)
        return best_x, sorted(current_indices)

    def update_matrices(self, j, chosen, A_sub, pseudo_inv):
        
        return new_A_sub, new_pseudo_inv

    def r1_find_best_x(self, j, indices, measurement, A_sub, pseudo_inv):
        q = len(indices) + 1

        sorted_indices = sorted(indices)
        pos = sorted_indices.index(j)

        new_A = self.A[indices,:]
        x_dict = np.linalg.lstsq(new_A.transpose(), measurement.transpose(), rcond=None)
        return x_dict[0], x_dict[1].sum()/2

                
    def choose_index(self, g):
        g_items = sorted(g.items())
        g_vec = np.array([value for key,value in g_items])
        p_vec = self.get_p_entr(g_vec)
        chosen = np.random.choice(len(g_vec), p=p_vec)
        return g_items[chosen][0]

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
                
    def find_best_x(self, indices, measurement):
        indices = sorted(indices)
        new_A = self.A[indices,:]
        x_dict = np.linalg.lstsq(new_A.transpose(), measurement.transpose(), rcond=None)
        return x_dict[0], x_dict[1].sum()/2

def compute_avg_nonzero(arr):
    pass

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

    sg = SmoothGreedy(A, Y, data, k=20)
    rmse = sg.evaluate()
    print("RMSE: ", rmse)
    np.savetxt("data/recovered_1", sg.reconstruction, delimiter=',')

    
