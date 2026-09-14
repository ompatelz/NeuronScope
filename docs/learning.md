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
