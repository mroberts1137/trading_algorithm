import numpy as np
import matplotlib.pyplot as plt


class AutoRegression:
    def __init__(self, phi, theta):
        # Array of coefficients
        self.phi = phi if len(phi) > 0 else [0]
        self.theta = theta if len(theta) > 0 else [0]

        # Order of ARIMA(p, q)
        self.p = len(phi)
        self.q = len(theta)

        self.noise = np.zeros(self.q) if self.q > 0 else [0]

        # Reverse the vectors for use in dot product. This is just a convenience.
        self.vec_phi = self.phi[::-1]
        self.vec_theta = self.theta[::-1]

    def __repr__(self):
        return f'ARIMA({self.p}, {self.q}):\nPHI: {self.phi}\nTHETA: {self.theta}\nRoots: {self.get_roots()}'

    def step(self, data, error):
        '''
        Generates next data point following the ARIMA(p, q) model:
        X_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} + epsilon_t + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q}
        :param data: [X_{t-p}, ..., X_{t-1}], past values. Needs to be reversed in dot product with phi
        :param noise: [epsilon_{t-q}, ..., epsilon_{t-1}]. Same as above, but for theta.
        :param error: epsilon_t
        :return:
        '''
        # Ensure len(data) = len(phi)
        if len(data) < self.p:
            data = np.pad(data, (self.p - len(data), 0))
        if len(data) > self.p:
            data = data[-self.p:]
        if self.p == 0:
            data = [0]

        # vec_phi and vec_theta are the reversed phi and theta. Just to make dot product easier.
        new_x = np.dot(self.vec_phi, data) + np.dot(self.vec_theta, self.noise) + error

        # Insert epsilon_t into noise[epsilon_{t-q}, ..., epsilon_{t-1}] -> [epsilon_{t-q+1}, ..., epsilon_t]
        self.noise = np.roll(self.noise, -1)
        self.noise[-1] = error
        return new_x

    def get_roots(self):
        if self.p >= 1:
            characteristic = np.polynomial.polynomial.Polynomial(np.concatenate(([1], -self.phi)))
            return characteristic.roots()
        else:
            return []

    def plot_roots(self):
        roots = self.get_roots()
        if len(roots) == 0:
            return

        # Extract real and imaginary parts
        real_parts = np.real(roots)
        imaginary_parts = np.imag(roots)

        # Plotting on the complex plane
        plt.figure(figsize=(6, 6))

        # Draw the unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        plt.plot(x_circle, y_circle, color='red', linestyle='--', linewidth=1.5, label='Unit Circle')

        # Scatter plot for the roots
        plt.scatter(real_parts, imaginary_parts, color='blue', label='Roots')

        # Add labels, grid, and axis lines
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.title('Roots of the Characteristic Equation')

        plt.show()