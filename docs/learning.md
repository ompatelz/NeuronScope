# Learning notes

## The first architecture boundary

The browser is responsible for presenting an experiment and accepting a configuration. The backend is responsible for turning a valid request into reproducible training work. Keeping those jobs separate means a future command-line runner or notebook can reuse the same model and training code.

When the model arrives, a layer will compute:

`z = Wx + b`

and then apply an activation function:

`a = f(z)`

The React app will visualize the resulting measurements, but it should never be the source of truth for those calculations.

## Dataset and neuron foundations

A feature vector is one observation represented by numbers. NeuronScope starts with two
features, `x` and `y`, so the observation can also be drawn as a point. Binary classification
asks a model to assign each point to class 0 or class 1.

A neuron computes `z = Wx + b`, where weights control how strongly each input matters and the
bias shifts the result. An activation computes `a = f(z)`. Without a nonlinear activation,
stacked layers still collapse to one linear transformation. That is why a single straight
boundary cannot separate XOR's alternating quadrants, while a multilayer nonlinear network can.

## Configurable multilayer perceptron

A layer applies the same affine transformation to a batch of feature vectors. PyTorch represents
that batch as a tensor shaped `(batch, features)`. Forward propagation sends that tensor through
each hidden linear layer and activation, then returns raw output logits.

- ReLU keeps positive values and clips negative values to zero. It is simple and usually works
  well in hidden layers.
- Sigmoid compresses values into `(0, 1)`, but deep sigmoid networks can suffer from very small
  gradients.
- Tanh compresses values into `(-1, 1)` and is zero-centered, but it can also saturate.
- A logit is the final unbounded score. NeuronScope does not put an activation on the output layer;
  later, binary-cross-entropy-with-logits can combine the stable loss and sigmoid calculation.

Weights must start with useful scale and symmetry must be broken. PyTorch's default initialization
is a sound baseline. Xavier initialization scales weights using both fan-in and fan-out and is a
natural fit for tanh or sigmoid. He initialization emphasizes fan-in and is designed for ReLU.
Seeding makes construction repeatable, while NeuronScope isolates model construction so it does
not unexpectedly advance the caller's global random-number stream.
