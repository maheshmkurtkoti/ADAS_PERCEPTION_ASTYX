import numpy as np

class KalmanFilterCV:
    def __init__(self, dt =0.1):
        self.dt = dt

        self.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.Q = np.eye(4)*0.05
        self.R = np.eye(2)*0.5

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P
    
    def update(self, x, P, z):
        y = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ self.H) @ P
        return x, P