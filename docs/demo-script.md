# NeuronScope demo script

This flow demonstrates the implemented product in roughly 3–5 minutes. Start the API and frontend
using the root README, then open <http://127.0.0.1:5173>.

## 0:00–0:30 — Establish the problem

Say: “A final accuracy score hides how a neural network reached it. NeuronScope keeps the task
simple—classifying 2D points—so we can inspect the learning process itself.”

Point out the three workbench regions: experiment controls on the left, model/boundary views in the
center, and inspection/diagnostic evidence on the right, with metrics and playback below.

## 0:30–1:30 — Train a healthy baseline

Use the default configuration:

- Dataset: Two Moons
- Samples: 200; noise: 0.12; seed: 42
- Hidden layers: `8, 8`
- Activation: ReLU; initialization: He
- Optimizer: Adam; learning rate: 0.01; epochs: 200

Start training. While it runs, explain that the browser submits a bounded configuration and the
FastAPI service generates data, builds the PyTorch MLP, trains it synchronously, and returns real
measurements.

When it completes:

1. Show the Network view and select a hidden neuron. Point out that its nodes, edges, activation,
   width, and parameter count come from the actual architecture.
2. Switch to Boundary. Identify the two sample classes, the probability field, and the 0.5 contour.
3. Show the loss/accuracy traces and their exact recent values.
4. In Selection, inspect gradient norm, weight norm, activation range, and zero percentage.
5. Open Diagnostics. If no rule fired, emphasize that “no rule triggered” is an observed result,
   not a manufactured success message.

## 1:30–2:20 — Replay learning

Move the training-playback slider from the first recorded epoch toward the final epoch. Show how the
decision boundary, metrics, and available instrumentation correspond to the selected snapshot.

Explain that NeuronScope stores only a capped set of selected CPU model states during training,
converts them into public scalar/grid data, restores the final model, and never sends model weights
to the browser.

## 2:20–3:40 — Create a contrasting experiment

Change several controls to make the comparison meaningful:

- Activation: Sigmoid
- Initialization: Xavier
- Optimizer: SGD
- Learning rate: 0.01
- Hidden layers: `8, 8, 8, 8`

Run the experiment. Do not promise that a warning will always appear: diagnostics depend on the
measured statistics and configured thresholds. Compare the new loss, accuracy, layer signals, and
any emitted diagnostic evidence with the healthy baseline.

Open the run-comparison table and select each run. Point out that the last five completed responses
remain available in browser memory without retraining, and that selecting one restores its graph,
boundary, metrics, signals, and diagnostics.

## 3:40–4:30 — Explain the debugger value

If a diagnostic fired, open it and read its observed layer, epochs, metric, and threshold before its
suggested actions. Say: “These are transparent rules over captured PyTorch statistics, not an LLM
guessing from the final score.”

If no diagnostic fired, use the layer inspector to compare gradient magnitudes and activation zero
percentages directly. The honest absence of a warning is itself part of the design.

Close with: “NeuronScope connects `loss.backward()` to visible evidence: the changing boundary,
metric history, gradient flow, activation behavior, and reproducible configuration behind one
training run.”

## Optional failure-state proof

For a quick validation demonstration, enter an invalid hidden-layer expression such as `0, 8` or
an epoch count above 5,000 and submit. The workbench links the error to the invalid control and does
not send a training request. Restore the valid value before continuing.
