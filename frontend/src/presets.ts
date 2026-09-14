import type { WorkbenchConfig } from "./config";

export interface ExperimentPreset {
  id: string;
  name: string;
  description: string;
  intent: string;
  config: WorkbenchConfig;
}

const baseline: WorkbenchConfig = {
  dataset: "two_moons", samples: 200, noise: 0.12, seed: 42,
  hiddenLayers: "8, 8", activation: "relu", initialization: "he",
  optimizer: "adam", learningRate: 0.01, epochs: 200,
  boundaryResolution: 48, playbackSnapshots: 12, traceSample: 1, diagnosticWindow: 3,
  vanishingGradientNorm: 1e-6, explodingGradientNorm: 100, deadReluPercentage: 95,
};

export const initialConfig: WorkbenchConfig = baseline;

export const experimentPresets: ExperimentPreset[] = [
  {
    id: "baseline", name: "Healthy baseline", description: "Compact ReLU network on Two Moons",
    intent: "A fast, stable reference run for inspecting normal gradient flow.", config: baseline,
  },
  {
    id: "spiral", name: "Spiral challenge", description: "Deeper tanh model on a harder boundary",
    intent: "Exercises a more complex nonlinear dataset and longer optimization path.",
    config: { ...baseline, dataset: "spiral", samples: 600, noise: 0.14, seed: 7, hiddenLayers: "32, 32, 16", activation: "tanh", initialization: "xavier", learningRate: 0.005, epochs: 500, boundaryResolution: 64, playbackSnapshots: 18 },
  },
  {
    id: "vanishing", name: "Gradient stress", description: "Deep sigmoid stack with strict detection",
    intent: "Designed to make weakening early-layer gradients easier to observe; a warning is evidence-dependent.",
    config: { ...baseline, hiddenLayers: "16, 16, 16, 16, 16, 16", activation: "sigmoid", initialization: "xavier", optimizer: "sgd", learningRate: 0.01, epochs: 350, diagnosticWindow: 5, vanishingGradientNorm: 0.0001 },
  },
  {
    id: "dead-relu", name: "ReLU stress", description: "Aggressive SGD against noisy circles",
    intent: "Raises the chance of inactive ReLU units while keeping the diagnostic threshold explicit.",
    config: { ...baseline, dataset: "circles", samples: 400, noise: 0.2, hiddenLayers: "32, 32, 32", optimizer: "sgd", learningRate: 0.35, epochs: 250, deadReluPercentage: 80 },
  },
];
