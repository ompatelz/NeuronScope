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

export type ConfigField = "samples" | "noise" | "seed" | "hiddenLayers" | "learningRate" | "epochs";
export interface ConfigValidationError { field: ConfigField; message: string }

export interface ValidatableConfig {
  samples: number;
  noise: number;
  seed: number;
  hiddenLayers: string;
  learningRate: number;
  epochs: number;
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
  return null;
}
