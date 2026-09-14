export type DatasetKind = "two_moons" | "circles" | "xor" | "spiral";
export type ActivationName = "relu" | "sigmoid" | "tanh";
export type InitializationName = "default" | "xavier" | "he";
export type OptimizerName = "sgd" | "adam";

export interface ActivationStatistics {
  mean: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  zero_percentage: number;
  nonfinite_count: number;
}

export interface GradientStatistics {
  norm: number | null;
  mean: number | null;
  std: number | null;
  nonfinite_count: number;
}

export interface LayerInstrumentation {
  layer_name: string;
  activation: ActivationStatistics | null;
  weight_norm: number | null;
  gradients: GradientStatistics;
}

export interface EpochInstrumentation {
  epoch: number;
  layers: LayerInstrumentation[];
}

export interface DiagnosticResult {
  type: "vanishing_gradients" | "exploding_gradients" | "dead_relu";
  severity: "warning" | "critical";
  evidence: Array<{ layer_name: string; epochs: number[]; metric: string; observed_values: number[]; threshold: number }>;
  explanation: string;
  possible_actions: string[];
}

export interface PlaybackSnapshot {
  epoch: number;
  metrics: { epoch: number; loss: number; accuracy: number };
  instrumentation: EpochInstrumentation | null;
  probabilities: number[];
}

export interface ExperimentRequest {
  dataset: { kind: DatasetKind; samples: number; noise: number; seed: number };
  model: {
    input_size: 2;
    hidden_layers: number[];
    output_size: 1;
    activation: ActivationName;
    initialization: InitializationName;
    seed: number;
  };
  training: {
    optimizer: OptimizerName;
    learning_rate: number;
    epochs: number;
    instrumentation: boolean;
  };
  boundary?: { resolution: number };
  diagnostics?: { consecutive_epochs?: number; vanishing_gradient_norm?: number; exploding_gradient_norm?: number; dead_relu_zero_percentage?: number };
  playback?: { max_snapshots: number };
}

export interface ExperimentResponse {
  dataset: {
    config: ExperimentRequest["dataset"];
    points: Array<{ x: number; y: number; label: 0 | 1 }>;
  };
  architecture: {
    input_size: number;
    hidden_layers: number[];
    output_size: number;
    layers: Array<{
      name: string;
      input_size: number;
      output_size: number;
      activation: ActivationName | null;
      parameter_count: number;
    }>;
    total_parameters: number;
  };
  training: {
    config: ExperimentRequest["training"];
    history: Array<{ epoch: number; loss: number; accuracy: number }>;
    instrumentation: EpochInstrumentation[];
    final_loss: number;
    final_accuracy: number;
  };
  boundary: {
    resolution: number;
    x_coordinates: number[];
    y_coordinates: number[];
    probabilities: number[];
  };
  diagnostics: DiagnosticResult[];
  playback: {
    resolution: number;
    x_coordinates: number[];
    y_coordinates: number[];
    snapshots: PlaybackSnapshot[];
  };
}

function errorMessage(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return undefined;
  const detail = payload.detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return undefined;
  return detail
    .map((entry) => {
      if (typeof entry !== "object" || entry === null || !("msg" in entry)) return null;
      return typeof entry.msg === "string" ? entry.msg : null;
    })
    .filter((message): message is string => message !== null)
    .join("; ");
}

export async function createExperiment(
  request: ExperimentRequest,
  signal?: AbortSignal,
): Promise<ExperimentResponse> {
  const response = await fetch("/api/v1/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) {
    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      payload = undefined;
    }
    throw new Error(errorMessage(payload) || `Training request failed (${response.status}).`);
  }
  return (await response.json()) as ExperimentResponse;
}
