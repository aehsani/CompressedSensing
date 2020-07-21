import numpy as np
import matplotlib.pyplot as plt

class ProjectPD:
    @classmethod
    def project_pd(cls, A, eps=0.1):
        u, s, vh = np.linalg.svd(A)
        np.maximum(s, eps, out=s)
        return u @ np.diag(s) @ vh
        
    @classmethod
    def project_pd_diff(A, epsilon):
        # TODO see paper by Piotr Indyk on low rank learning
        pass

if __name__ == "__main__":
    pass


