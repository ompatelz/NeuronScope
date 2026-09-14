export function parseHiddenLayers(value: string): number[] {
  const layers = value.split(",").map((entry) => Number(entry.trim()));
  if (
    layers.length < 1 ||
    layers.length > 8 ||
    layers.some((width) => !Number.isInteger(width) || width < 1 || width > 256)
  ) {
    throw new Error("Use 1–8 comma-separated layer widths between 1 and 256.");
  }
  return layers;
}

export type ConfigField =
  | "samples" | "noise" | "seed" | "hiddenLayers" | "learningRate" | "epochs"
  | "boundaryResolution" | "playbackSnapshots" | "diagnosticWindow"
  | "traceSample" | "vanishingGradientNorm" | "explodingGradientNorm" | "deadReluPercentage";
export interface ConfigValidationError { field: ConfigField; message: string }

export type ValidatableConfig = Pick<WorkbenchConfig,
  "samples" | "noise" | "seed" | "hiddenLayers" | "learningRate" | "epochs"
  | "boundaryResolution" | "playbackSnapshots" | "diagnosticWindow"
  | "traceSample" | "vanishingGradientNorm" | "explodingGradientNorm" | "deadReluPercentage"
>;

export function estimateParameterCount(hiddenLayers: string): number | null {
  try {
    const widths = [2, ...parseHiddenLayers(hiddenLayers), 1];
    return widths.slice(1).reduce((total, width, index) => total + (widths[index]! + 1) * width, 0);
  } catch {
    return null;
  }
}

export function validateConfig(config: ValidatableConfig): ConfigValidationError | null {
  if (!Number.isInteger(config.samples) || config.samples < 40 || config.samples > 2_000) {
    return { field: "samples", message: "Samples must be a whole number from 40 to 2,000." };
  }
  if (!Number.isFinite(config.noise) || config.noise < 0 || config.noise > 0.5) {
    return { field: "noise", message: "Noise must be between 0 and 0.5." };
  }
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 2_147_483_647) {
    return { field: "seed", message: "Seed must be a whole number from 0 to 2,147,483,647." };
  }
  try {
    parseHiddenLayers(config.hiddenLayers);
  } catch (error) {
    return { field: "hiddenLayers", message: error instanceof Error ? error.message : "Invalid hidden layers." };
  }
  if (!Number.isFinite(config.learningRate) || config.learningRate <= 0 || config.learningRate > 1) {
    return { field: "learningRate", message: "Learning rate must be greater than 0 and at most 1." };
  }
  if (!Number.isInteger(config.epochs) || config.epochs < 1 || config.epochs > 5_000) {
    return { field: "epochs", message: "Epochs must be a whole number from 1 to 5,000." };
  }
  if (!Number.isInteger(config.boundaryResolution) || config.boundaryResolution < 24 || config.boundaryResolution > 80) {
    return { field: "boundaryResolution", message: "Boundary resolution must be a whole number from 24 to 80." };
  }
  if (!Number.isInteger(config.playbackSnapshots) || config.playbackSnapshots < 2 || config.playbackSnapshots > 24) {
    return { field: "playbackSnapshots", message: "Playback snapshots must be a whole number from 2 to 24." };
  }
  if (!Number.isInteger(config.traceSample) || config.traceSample < 1 || config.traceSample > config.samples) {
    return { field: "traceSample", message: `Trace sample must be a whole number from 1 to ${config.samples}.` };
  }
  if (!Number.isInteger(config.diagnosticWindow) || config.diagnosticWindow < 2 || config.diagnosticWindow > 20) {
    return { field: "diagnosticWindow", message: "Diagnostic window must be a whole number from 2 to 20." };
  }
  if (!Number.isFinite(config.vanishingGradientNorm) || config.vanishingGradientNorm <= 0) {
    return { field: "vanishingGradientNorm", message: "Vanishing-gradient threshold must be greater than 0." };
  }
  if (!Number.isFinite(config.explodingGradientNorm) || config.explodingGradientNorm <= 0) {
    return { field: "explodingGradientNorm", message: "Exploding-gradient threshold must be greater than 0." };
  }
  if (!Number.isFinite(config.deadReluPercentage) || config.deadReluPercentage < 0 || config.deadReluPercentage > 100) {
    return { field: "deadReluPercentage", message: "Dead-ReLU threshold must be between 0 and 100%." };
  }
  return null;
}
import type {
  ActivationName,
  DatasetKind,
  InitializationName,
  OptimizerName,
} from "./api/experiments";

export interface WorkbenchConfig {
  dataset: DatasetKind;
  samples: number;
  noise: number;
  seed: number;
  hiddenLayers: string;
  activation: ActivationName;
  initialization: InitializationName;
  optimizer: OptimizerName;
  learningRate: number;
  epochs: number;
  boundaryResolution: number;
  playbackSnapshots: number;
  traceSample: number;
  diagnosticWindow: number;
  vanishingGradientNorm: number;
  explodingGradientNorm: number;
  deadReluPercentage: number;
}
