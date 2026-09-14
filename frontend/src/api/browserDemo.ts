import type {
  ExperimentRequest,
  ExperimentResponse,
  ForwardLayerTrace,
  PlaybackSnapshot,
} from "./experiments";

type Point = { x: number; y: number; label: 0 | 1 };

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function normal(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  const v = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeDataset(request: ExperimentRequest): Point[] {
  const random = seeded(request.dataset.seed);
  const noise = request.dataset.noise;
  return Array.from({ length: request.dataset.samples }, (_, index) => {
    const half = index < request.dataset.samples / 2 ? 0 : 1;
    const t = random() * Math.PI;
    if (request.dataset.kind === "circles") {
      const radius = half ? 1.1 : 0.45;
      const angle = random() * Math.PI * 2;
      return {
        x: radius * Math.cos(angle) + normal(random) * noise,
        y: radius * Math.sin(angle) + normal(random) * noise,
        label: half as 0 | 1,
      };
    }
    if (request.dataset.kind === "xor") {
      const x = random() * 2 - 1;
      const y = random() * 2 - 1;
      return {
        x: x + normal(random) * noise,
        y: y + normal(random) * noise,
        label: (x * y > 0 ? 1 : 0) as 0 | 1,
      };
    }
    if (request.dataset.kind === "spiral") {
      const r = t / Math.PI;
      const angle = half ? t + Math.PI : t;
      return {
        x: r * Math.cos(angle) + normal(random) * noise,
        y: r * Math.sin(angle) + normal(random) * noise,
        label: half as 0 | 1,
      };
    }
    return {
      x: (half ? 1 - Math.cos(t) : Math.cos(t) - 1) + normal(random) * noise,
      y: (half ? -Math.sin(t) + 0.45 : Math.sin(t)) + normal(random) * noise,
      label: half as 0 | 1,
    };
  });
}

function activate(value: number, activation: string | null): number {
  if (activation === "relu") return Math.max(0, value);
  if (activation === "sigmoid") return 1 / (1 + Math.exp(-value));
  if (activation === "tanh") return Math.tanh(value);
  return value;
}

function decisionScore(x: number, y: number, kind: string, progress: number): number {
  const sharpness = 1.8 + progress * 5.5;
  if (kind === "circles") return (x * x + y * y - 0.62) * sharpness;
  if (kind === "xor") return x * y * sharpness * 2.8;
  if (kind === "spiral") return (Math.sin(Math.atan2(y, x) * 2.2 + Math.hypot(x, y) * 4) - 0.1) * sharpness;
  return (y - 0.25 * Math.sin(x * 2.2) + 0.15 * x) * sharpness;
}

function probability(x: number, y: number, kind: string, progress: number): number {
  return 1 / (1 + Math.exp(-decisionScore(x, y, kind, progress)));
}

function makeTrace(request: ExperimentRequest, point: Point, progress: number): ForwardLayerTrace[] {
  let values = [point.x, point.y];
  return request.model.hidden_layers.map((width, index) => {
    const pre_activations = Array.from({ length: Math.min(width, 16) }, (_, neuron) => {
      const weightA = Math.sin((index + 1) * (neuron + 1) * 0.71 + request.model.seed);
      const weightB = Math.cos((index + 1) * (neuron + 1) * 0.47 + request.model.seed);
      return (values[0] ?? 0) * weightA + (values[1] ?? 0) * weightB + (progress - 0.5) * 0.6;
    });
    const activations = pre_activations.map((value) => activate(value, request.model.activation));
    values = [activations.reduce((sum, value) => sum + value, 0) / Math.max(activations.length, 1), activations[0] ?? 0];
    return {
      layer_name: `layers.${index}`,
      activation_name: request.model.activation,
      pre_activations,
      activations,
    };
  });
}

function boundary(points: Point[], request: ExperimentRequest, progress: number) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs) - 0.35;
  const maxX = Math.max(...xs) + 0.35;
  const minY = Math.min(...ys) - 0.35;
  const maxY = Math.max(...ys) + 0.35;
  const resolution = request.boundary?.resolution ?? 48;
  const x_coordinates = Array.from({ length: resolution }, (_, index) => minX + ((maxX - minX) * index) / (resolution - 1));
  const y_coordinates = Array.from({ length: resolution }, (_, index) => minY + ((maxY - minY) * index) / (resolution - 1));
  const probabilities = y_coordinates.flatMap((y) => x_coordinates.map((x) => probability(x, y, request.dataset.kind, progress)));
  return { resolution, x_coordinates, y_coordinates, probabilities };
}

function instrumentation(request: ExperimentRequest, epoch: number) {
  const progress = epoch / Math.max(request.training.epochs, 1);
  return {
    epoch,
    layers: request.model.hidden_layers.map((width, index) => ({
      layer_name: `layers.${index}`,
      activation: {
        mean: 0.22 + progress * 0.38 + index * 0.015,
        std: Math.max(0.04, 0.38 - progress * 0.14),
        minimum: request.model.activation === "relu" ? 0 : -1,
        maximum: 1 + progress * 0.75,
        zero_percentage: request.model.activation === "relu" ? Math.max(3, 42 - progress * 30) : 0,
        nonfinite_count: 0,
      },
      weight_norm: 0.7 + progress * 1.8 + width / 100,
      gradients: {
        norm: Math.max(0.0001, 0.45 * Math.exp(-progress * 2.4) + index * 0.015),
        mean: 0.01 * (1 - progress),
        std: 0.12 * Math.exp(-progress),
        nonfinite_count: 0,
      },
    })),
  };
}

export async function createBrowserDemoExperiment(request: ExperimentRequest): Promise<ExperimentResponse> {
  await new Promise((resolve) => window.setTimeout(resolve, 350));
  const points = makeDataset(request);
  const totalEpochs = request.training.epochs;
  const history = Array.from({ length: totalEpochs }, (_, index) => {
    const epoch = index + 1;
    const progress = epoch / Math.max(totalEpochs, 1);
    return {
      epoch,
      loss: 0.72 * Math.exp(-progress * 3.1) + 0.08,
      accuracy: Math.min(0.98, 0.54 + progress * 0.4 + Math.sin(progress * Math.PI) * 0.04),
    };
  });
  const sample = points[Math.min(request.playback?.trace_sample_index ?? 0, points.length - 1)] ?? points[0]!;
  const snapshotCount = request.playback?.max_snapshots ?? 12;
  const epochs = Array.from(new Set(Array.from({ length: snapshotCount }, (_, index) => (
    Math.max(1, Math.round(1 + (index * (totalEpochs - 1)) / Math.max(snapshotCount - 1, 1)))
  ))));
  const snapshots: PlaybackSnapshot[] = epochs.map((epoch) => {
    const progress = epoch / Math.max(totalEpochs, 1);
    const traceLayers = makeTrace(request, sample, progress);
    const output_logit = decisionScore(sample.x, sample.y, request.dataset.kind, progress);
    const predicted_probability = 1 / (1 + Math.exp(-output_logit));
    return {
      epoch,
      metrics: history[epoch - 1]!,
      instrumentation: instrumentation(request, epoch),
      probabilities: boundary(points, request, progress).probabilities,
      forward_pass: {
        sample_index: request.playback?.trace_sample_index ?? 0,
        input_values: [sample.x, sample.y],
        expected_label: sample.label,
        layers: traceLayers,
        output_logit,
        predicted_probability,
        predicted_label: predicted_probability >= 0.5 ? 1 : 0,
      },
    };
  });
  const finalBoundary = boundary(points, request, 1);
  const layers = request.model.hidden_layers.map((width, index, all) => ({
    name: `layers.${index}`,
    input_size: index === 0 ? request.model.input_size : all[index - 1]!,
    output_size: width,
    activation: request.model.activation,
    parameter_count: ((index === 0 ? request.model.input_size : all[index - 1]!) + 1) * width,
  }));
  const outputParameters = ((request.model.hidden_layers.at(-1) ?? request.model.input_size) + 1) * request.model.output_size;
  return {
    dataset: { config: request.dataset, points },
    architecture: {
      input_size: request.model.input_size,
      hidden_layers: request.model.hidden_layers,
      output_size: request.model.output_size,
      layers,
      total_parameters: layers.reduce((sum, layer) => sum + layer.parameter_count, 0) + outputParameters,
    },
    training: {
      config: request.training,
      history,
      instrumentation: epochs.map((epoch) => instrumentation(request, epoch)),
      final_loss: history.at(-1)?.loss ?? 0,
      final_accuracy: history.at(-1)?.accuracy ?? 0,
    },
    boundary: finalBoundary,
    diagnostics: [],
    playback: { ...finalBoundary, snapshots },
  };
}
