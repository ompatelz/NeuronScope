# Learning notes

## The first architecture boundary

The browser is responsible for presenting an experiment and accepting a configuration. The backend is responsible for turning a valid request into reproducible training work. Keeping those jobs separate means a future command-line runner or notebook can reuse the same model and training code.

When the model arrives, a layer will compute:

`z = Wx + b`

and then apply an activation function:

`a = f(z)`

The React app will visualize the resulting measurements, but it should never be the source of truth for those calculations.
