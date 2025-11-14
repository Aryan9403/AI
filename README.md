1. OR Recognizer using single neuron.
We have one neuron that takes in float or array of floats and outputs a single float. 
We are writing a test so that it recognizes the OR gate
x1 = 0, x2 = 0; y = 0
x1 = 0, x2 = 1; y = 1
x1 = 1, x2 = 0; y = 1
x1 = 1, x2 = 1; y = 1

So when a is close to 0, the label is 0
when a is close to 1, the label is 1

Lets first define wrongness
if y = 1, and neuron outputs a = 0.2 -> wrong
if y = 0, and neuron outputs a = 0.9 -> wrong

So we need a formulae which gives small number when a is close to y
big number when a is far from y

our loss function is loss = (a-y)^2
That means if neuron outputs 0.5 and y = 1
then (0.5-1)^2 = 0.25 = loss

How is the neuron learning 
if loss is high weight will change a lot
if loss is small weight will change little


Computing weight changes, how and when to change weights
for some weight w_i
if input x_i = 0
that weight must not have affected the output much as z = w.x + b
if x_i is large then that weight heavily affected the output

so rule:
w_i <- w_i - learning_rate * (a-y) * x_i
(a-y) -> how wrong the neuron was
x_i -> how much neuron depended on the input
learning rate -> how strong to adjust

Bias Update
b <- b - learning_rate*(a-y)

compute how each weight contributed to the wrongness
dz = a-y
dW = dz*x
dB = dz

parameter <- parameter - learning_rate x gradient

Wanted to build a single artificial neuron based on sigmoid function


so first got the sigmoid function down which is S = 1 / 1 + e^-x

def sigmoid(z: float | np.ndarray) -> float | np.ndarray:

    """ sigmoid activation function
    maps any real number to (0, 1)"""
    return 1/(1 + np.exp(-z))

Takes input a flow of array of floats, returns a float or array of float between 0 to 1, respectively.

Then defined a class Neuron which has a init function 
which has input n_input and weight_scale
it stores n_input as n_input, class variable
and it take n_input generates random array of number of n_inputs multiplies it by weight_scale and stores it in self.w

then it also declares bias, self.b = 0
 
def __init__(self, n_inputs: int, *, weight_scale: float = 0.01):
        """
        A single neuron with:
        - n_inputs incoming connections
        - weight vector w (shape: (n_inputs,))
        - scalar bias b

        By default, weights are small random numbers, bias starts at 0.
        """
        self.n_inputs = n_inputs
        # small random weights
        self.w = np.random.randn(n_inputs) * weight_scale
        self.b = 0.0

Then we have a forward function 

Forward function takes in the input numpy array
it checks the datatype for float
it checks for the expected shape

then declares z as dot product of w and input x + b
z = np.dot(self.w, x) + self.b
this is the neuron function main

then declares and returns a when sigmoid is applied to z
a = sigmoid(z)








