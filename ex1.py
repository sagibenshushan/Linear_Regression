import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    matrix_x = data.iloc[:, :9].values
    vector_y = data.iloc[:, -1].values
    return matrix_x, vector_y

def normalize_data(matrix_x):
    mean = np.mean(matrix_x, axis=0)
    std_dev = np.std(matrix_x, axis=0)
    normalized_x = (matrix_x - mean) / std_dev
    return normalized_x

def add_ones_column(matrix_x):
    ones = np.ones((matrix_x.shape[0], 1))
    x_with_ones = np.hstack((ones, matrix_x))
    return x_with_ones

def h_theta(theta, vector_x):
    return np.dot(vector_x, theta)

def compute_cost(theta, matrix_x, vector_y):
    m = len(vector_y)
    predictions = matrix_x @ theta
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - vector_y))
    return cost

def compute_gradient(theta, matrix_x, vector_y):
    m = len(vector_y)
    predictions = matrix_x @ theta
    gradient = (1 / m) * (matrix_x.T @ (predictions - vector_y))
    return gradient

def gradient_descent(matrix_x, vector_y, theta, alpha, num_iterations):
    m = len(vector_y)
    cost_history = []

    for i in range(num_iterations):
        gradient = compute_gradient(theta, matrix_x, vector_y)
        theta = theta.copy() - alpha * gradient
        cost = compute_cost(theta, matrix_x, vector_y)
        cost_history.append(cost)

    return theta, cost_history


# def plot_cost_history(alphas, cost_histories, num_iterations):
#     plt.figure(figsize=(12, 8))

#     for alpha, cost_history in zip(alphas, cost_histories):
#         plt.plot(range(num_iterations), cost_history, label=f'alpha = {alpha}')

#     plt.xlabel('Time steps')
#     plt.ylabel('J(Theta)')
#     plt.title('Decrease in J(Theta) with Gradient Descent')
#     plt.legend()
#     plt.grid()
#     plt.show()
def plot_cost_history(alphas, cost_histories, num_iterations, title):
    plt.figure(figsize=(12, 8))

    for alpha, cost_history in zip(alphas, cost_histories):
        plt.plot(range(num_iterations), cost_history, label=f'alpha = {alpha}')

    plt.xlabel('Time steps')
    plt.ylabel('J(Theta)')
    plt.title(f'Decrease in J(Theta) with {title}')
    plt.legend()
    plt.grid()
    plt.show()

def mini_batch_gradient_descent(matrix_x, vector_y, theta, alpha, num_iterations, batch_size=20):
    m = len(vector_y)
    cost_history = []

    for i in range(num_iterations):
        cost_iteration = 0
        shuffled_indices = np.random.permutation(m)
        matrix_x_shuffled = matrix_x[shuffled_indices]
        vector_y_shuffled = vector_y[shuffled_indices]

        for j in range(0, m, batch_size):
            x_batch = matrix_x_shuffled[j:j + batch_size]
            y_batch = vector_y_shuffled[j:j + batch_size]

            gradient = compute_gradient(theta, x_batch, y_batch)
            theta = theta.copy() - alpha * gradient
            cost = compute_cost(theta, x_batch, y_batch)
            cost_iteration += cost

        cost_history.append(cost_iteration / (m / batch_size))

    return theta, cost_history

def momentum_gradient_descent(matrix_x, vector_y, theta, alpha, num_iterations, beta=0.9):
    m = len(vector_y)
    cost_history = []
    v = np.zeros(theta.shape)

    for i in range(num_iterations):
        gradient = compute_gradient(theta, matrix_x, vector_y)
        v = beta * v + (1 - beta) * gradient
        theta = theta.copy() - alpha * v
        cost = compute_cost(theta, matrix_x, vector_y)
        cost_history.append(cost)

    return theta, cost_history


def adagrad_gradient_descent(matrix_x, vector_y, theta, alpha, num_iterations, eps=1e-8):
    m = len(vector_y)
    cost_history = []
    v = np.zeros(theta.shape)

    for i in range(num_iterations):
        gradient = compute_gradient(theta, matrix_x, vector_y)
        v = v + gradient ** 2
        theta = theta.copy() - (alpha / (np.sqrt(v) + eps)) * gradient
        cost = compute_cost(theta, matrix_x, vector_y)
        cost_history.append(cost)

    return theta, cost_history


def adam_gradient_descent(matrix_x, vector_y, theta, alpha, num_iterations, beta1=0.9, beta2=0.999, eps=1e-8):
    m = len(vector_y)
    cost_history = []
    m_t = np.zeros(theta.shape)
    v_t = np.zeros(theta.shape)

    for i in range(1, num_iterations + 1):
        gradient = compute_gradient(theta, matrix_x, vector_y)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_hat = m_t / (1 - beta1 ** i)
        v_hat = v_t / (1 - beta2 ** i)
        theta = theta.copy() - (alpha / (np.sqrt(v_hat) + eps)) * m_hat
        cost = compute_cost(theta, matrix_x, vector_y)
        cost_history.append(cost)

    return theta, cost_history






def main():
    # Load data from CSV file
    filename = 'cancer_data.csv'
    matrix_x, vector_y = load_data(filename)

    # Normalize the data
    normalized_data = normalize_data(matrix_x)

    # Add a column of ones
    x_with_ones = add_ones_column(normalized_data)

    # Define learning rates and number of iterations
    alphas = [1, 0.1, 0.01, 0.001]
    num_iterations = 1000

    # Initialize a dictionary to store the cost histories for different optimization algorithms
    cost_histories = {'GD': [], 'Momentum': [], 'Adagrad': [], 'Adam': []}
    thetas = {'GD': [], 'Momentum': [], 'Adagrad': [], 'Adam': []}

    # Run the optimization algorithms with different learning rates
    for alpha in alphas:
        initial_theta = np.zeros(x_with_ones.shape[1])

        # Gradient Descent
        theta_gd, gd_cost_history = gradient_descent(x_with_ones, vector_y, initial_theta, alpha, num_iterations)
        cost_histories['GD'].append(gd_cost_history)
        thetas['GD'].append(theta_gd)

        # Momentum Gradient Descent
        theta_momentum, momentum_cost_history = momentum_gradient_descent(x_with_ones, vector_y, initial_theta, alpha, num_iterations)
        cost_histories['Momentum'].append(momentum_cost_history)
        thetas['Momentum'].append(theta_momentum)

        # Adagrad Gradient Descent
        theta_adagrad, adagrad_cost_history = adagrad_gradient_descent(x_with_ones, vector_y, initial_theta, alpha, num_iterations)
        cost_histories['Adagrad'].append(adagrad_cost_history)
        thetas['Adagrad'].append(theta_adagrad)

        # Adam Gradient Descent
        theta_adam, adam_cost_history = adam_gradient_descent(x_with_ones, vector_y, initial_theta, alpha, num_iterations)
        cost_histories['Adam'].append(adam_cost_history)
        thetas['Adam'].append(theta_adam)

    # Calculate and print the output for each optimization algorithm
    for algo_name in thetas.keys():
        print(f'{algo_name} Algorithm:')

        for idx, alpha in enumerate(alphas):
            theta = thetas[algo_name][idx]
            prediction = h_theta(theta, x_with_ones)
            cost = compute_cost(theta, x_with_ones, vector_y)
            gradient = compute_gradient(theta, x_with_ones, vector_y)

            print(f'Alpha: {alpha}')
            print('Prediction:', prediction)
            print('Cost:', cost)
            print('Gradient:', gradient)
            print('---' * 10)

    # Plot the cost histories for Gradient Descent, Momentum, Adagrad, and Adam algorithms
    for algo_name, algo_cost_histories in cost_histories.items():
        plt.figure(figsize=(12, 8))

        for idx, alpha in enumerate(alphas):
            plt.plot(algo_cost_histories[idx], label=f'alpha={alpha}')

        plt.title(f'{algo_name} Algorithm')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    main()