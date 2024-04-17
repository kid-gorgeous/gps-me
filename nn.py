import numpy as np
from tensorflow.keras import Sequential 
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 64)
y = np.random.randn(1000, 1)

# Loss function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Personal LSTM implementation
#  Inputs : 64 dim, 616 dim, 1 output
class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        
        # Initialize biases 
        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    # Create the forward pass generating the output and the hidden and cell states
    def forward(self, inputs):
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        for x in inputs:
            concat_input = np.vstack((h, x))
            f = sigmoid(np.dot(self.Wf, concat_input) + self.bf)
            i = sigmoid(np.dot(self.Wi, concat_input) + self.bi)
            o = sigmoid(np.dot(self.Wo, concat_input) + self.bo)
            g = np.tanh(np.dot(self.Wc, concat_input) + self.bc)
            c = f * c + i * g
            h = o * np.tanh(c)
        y = np.dot(self.Wy, h) + self.by
        return y, h, c

    def backward(self, d_y, inputs, h_prev, c_prev):
        
        # Change the shape of the input
        d_Why = np.dot(d_y, h_prev.T)
        d_by = d_y
        d_h_next = np.zeros_like(h_prev)
        d_c_next = np.zeros_like(c_prev)


        for i in reversed(range(len(inputs))):
            x = inputs[i]
            concat_input = np.vstack((h_prev, x))
            o = sigmoid(np.dot(self.Wo, concat_input) + self.bo)
            g = np.tanh(np.dot(self.Wc, concat_input) + self.bc)
            d_o = np.tanh(c_prev) * d_y
            d_c = o * d_y + d_c_next
            d_f = d_c * c_prev
            d_i = d_c * g
            d_g = d_c * i
            d_concat = np.dot(self.Wf.T, d_f) + np.dot(self.Wi.T, d_i) + np.dot(self.Wo.T, d_o) + np.dot(self.Wc.T, d_g)
            d_h_prev = d_concat[:self.hidden_dim, :]
            d_c_prev = f * d_c
            d_Whh = np.dot(d_concat * (1 - h_prev ** 2), concat_input.T)
            d_Wxh = np.dot(d_concat * (1 - h_prev ** 2), x.T)
            d_bh = (1 - h_prev ** 2) * d_concat
            d_Why += np.outer(d_y, h_prev)
            d_by += d_y
            d_y = np.dot(self.Wy.T, d_y)
            h_prev = h_prev

        return d_Why, d_by

    # Update the weights and biases
    def update(self, d_Why, d_by, lr):
        self.Why -= lr * d_Why
        self.by -= lr * d_by

    def train(self):
        pass
        
# Keras LSTM (Work in Progress)
class K_LSTM:
    def __init__(self):
        self.model = Sequential()


def main():
    # Test models here
    lstm = LSTM(64, 16, 1)




if __name__ == "__main__":
    main()

