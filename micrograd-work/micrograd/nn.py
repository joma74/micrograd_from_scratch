import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin, relLayerIdx, relNeuronIdx):
        """
        nin: number of inputs; how many weights; the data will be provided in the __call__
        relLayerIdx: index of the Layer this Neuron belongs to, relative inside it's MLP
        relNeuronIdx: index of the Neuron relative inside it's Layer
        """
        self.w = [Value(random.uniform(-1,1), _label=f'L{relLayerIdx}|N{relNeuronIdx}|w{_}') for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.b._label = f'L{relLayerIdx}|N{relNeuronIdx}|b'

    def __call__(self, xs):
        """
        xs: array of data for the inputs
        return: the out Value of this Neuron
        """
        # convert any x that is not an instance of Value into a Value ...
        xs = [x if isinstance(x, Value) else Value(x) for x in xs]
        # ... so a proper label can be given to it
        for idx, x in enumerate(xs):
            if(len(x._terms)):
                x._label = f'{self.b._label[0:len(self.b._label)-2]}|oi{idx}'
            else:
                x._label = f'{self.b._label[0:len(self.b._label)-2]}|i{idx}'
        # Multiply all the elements of w with the elemnts of x pairweise
        # Therefore we zip up the values of w with x; creates tuples and gives back an iterator
        # We sum them up and add the bias, giving the activation value
        # act = sum(wi * xi for wi,xi in zip(self.w, x))) + self.b
        # Optimize
        act = sum((wi * xi for wi,xi in zip(self.w, xs)), self.b)
        # Then we need to pass that through a non-linear function
        out = act.tanh()
        out._label = f'{self.b._label[0:len(self.b._label)-2]}|o'
        return outself

    def parameters(self):
        """
        return: the Bias and all Weights of this Neuron
        """
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, relLayerIdx):
        """
        nin: number of inputs; number of weights
        nout: number of outputs; how many Neurons in that layer, each initialized with nin
        relLayerIdx: index of the Layer, relative inside it's MLP
        """
        self.neurons = [Neuron(nin, relLayerIdx, _) for _ in range(nout)]

    def __call__(self, xs):
        """
        xs: array of data for the inputs
        return: array of the out Value of this Layer
        """
        # KNOW
        # The Input primitive values are replicated for each application to the Layer 0.
        # Per the Karpathy's micrograd implementation, there is no Layer for the Input.
        # WANT ?
        # Input primitive values to have an identity. So convert any x that is not an instance of 
        # Value into a Value. To enable, uncomment the next line.
        # xs = [x if isinstance(x, Value) else Value(x) for x in xs]
        #
        outs = [n(xs) for n in self.neurons]
        # for convenience, if outs is just an array with a single element, then return the element
        return outs[0] if len(outs) == 1 else outs

class MLP:
    """
    M(ulti)L(evel)P(erceptron)
    """
    def __init__(self, nin, lnout):
        """
        nin: number of inputs of the first layer
        lnout: list of numbers of outputs in each layer; how many Neurons in each layer
        """
        # [nin, nouts0, nouts1, ...]
        sz = [nin] + lnout
        # The outs of the previous layer are the ins of the current layer 
        # [Layer[nin, nouts0], Layer[nouts0, nouts1], Layer[nouts1, ...]
        self.layers = [Layer(sz[i], sz[i+1], i) for i in range(len(lnout))]

    def __call__(self, xs):
        """
        xs: array of data for the inputs for each layer
        return: array of the out Values of the last layer
        Each array of the out Values of the previous layer is forwarded to the next layer.
        Note that the Input primitive values are replicated for each application to the Layer 0.
        """
        # KNOW
        # The Input primitive values are replicated for each application to the Layer 0.
        # Per the Karpathy's micrograd implementation, there is no Layer for the Input.
        # WANT ?
        # Input primitive values to have an identity. So convert any x that is not an instance of 
        # Value into a Value. To enable, uncomment the next line.
        # xs = [x if isinstance(x, Value) else Value(x) for x in xs]
        #
        outs = xs
        for layer in self.layers:
            outs = layer(outs) 
        return outs