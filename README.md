# Five-most-used-algorithms-in-AI

1. Gradient Descent: The Mountain Climber*
 
 *Code: Gradient Descent*
 
```import numpy as np
def gradient_descent(start, gradient, learn_rate, n_iter=100, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

# Example usage for finding the minimum of f(x) = x^2 + 5 \\
def gradient(x):
    return 2 * x

minimum = gradient_descent(start=10.0, gradient=gradient, learn_rate=0.1)
print(f"The minimum occurs at: {minimum}")```

 *2. Backpropagation: The Blame Game*

*Code for Backprogation technique*

```import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Usage \\
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)

for _ in range(1500):
    nn.feedforward()
    nn.backprop()

print(nn.output)```

 3. K-Means Clustering: The Party Planner* 

*Code: K-Means Clustering** 

```import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data \\
np.random.seed(42)
X = np.random.randn(300, 2)

# Create K-Means instance \\
kmeans = KMeans(n_clusters=3)

# Fit the model \\
kmeans.fit(X)

# Plot the results \\
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title("K-Means Clustering")
plt.show()

4. Decision Trees: The Questionnaire*

 *Code: Decision Tree*

```from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data \\
iris = load_iris()
X, y = iris.data, iris.target

# Split the data \\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model \\
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy \\
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")```

5. Reinforcement Learning: The Pet Trainer** 

**Code for Reinforcement Learning** 

```import numpy as np

# Simple Q-learning implementation \\
class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

# Usage example (simplified environment) \\
env_size = 5
agent = QLearning(states=env_size, actions=4)  # 4 actions: up, down, left, right

for episode in range(1000):
    state = 0  # Start state
    done = False
    while not done:
        action = agent.get_action(state)
        # Simplified next_state and reward calculation
        next_state = min(max(0, state + action - 1), env_size - 1)
        reward = 1 if next_state == env_size - 1 else 0
        agent.update(state, action, reward, next_state)
        state = next_state
        done = (state == env_size - 1)

print(agent.q_table)
```
