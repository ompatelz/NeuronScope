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

## Instrumentation and gradient flow

During forward propagation, each hidden activation is the tensor passed onward to the next layer.
During backpropagation, each parameter gradient describes how a small change to that parameter
would change the loss. Gradient norms therefore provide a compact view of whether learning signals
are flowing through the network.

NeuronScope registers temporary forward hooks on hidden activations, immediately detaches their
outputs, and stores only scalar summaries. After `backward()`, it reads parameter gradients before
the optimizer clears them. Hooks are disabled during post-update metric evaluation and removed in
a `finally` block, preventing validation contamination, retained computation graphs, and duplicate
observations across repeated runs.

## Transparent training diagnostics

NeuronScope applies deterministic heuristics to recent instrumentation; it does not ask an LLM to
guess what happened. Consecutive tiny gradients suggest vanishing gradients, consecutive large
gradients suggest exploding gradients, and mostly zero ReLU outputs suggest dead units. A non-finite
gradient is critical immediately. The centralized thresholds are useful signals rather than
universal laws, so each result includes its exact observations, threshold, explanation, and possible
experiments to try.

## Loss, backpropagation, and optimization

For binary classification, the output logit is compared with the class label using binary cross
entropy. `BCEWithLogitsLoss` combines the sigmoid conversion and cross-entropy calculation in a
numerically stable operation. A lower loss means the logits better support the observed labels;
accuracy separately counts how often the logit's sign selects the correct class.

Each epoch performs three distinct operations: `optimizer.zero_grad()` clears gradients left by
the previous step, `loss.backward()` applies the chain rule through the computation graph and
stores parameter gradients, and `optimizer.step()` updates the parameters. SGD follows the
current gradient directly. Adam also tracks moving estimates of gradient magnitude, which often
makes it less sensitive to a single learning-rate choice.

NeuronScope's first engine uses one full CPU batch. This is appropriate for its small visual 2D
datasets and removes data-order randomness. The learning rate controls update size, while the
epoch count controls how many complete parameter updates are made. The engine returns post-update
loss and accuracy for every epoch so later visualizations can show real learning history.
