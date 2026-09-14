import { describe, expect, it } from "vitest";

import type { ExperimentResponse, ForwardPassTrace } from "../api/experiments";
import { architectureToGraph } from "./networkGraph";

function architecture(hiddenLayers: number[]): ExperimentResponse["architecture"] {
  const widths = [2, ...hiddenLayers, 1];
  return {
    input_size: 2,
    hidden_layers: hiddenLayers,
    output_size: 1,
    layers: widths.slice(1).map((width, index) => ({
      name: index < hiddenLayers.length ? `hidden_${index}` : "output",
      input_size: widths[index], output_size: width,
      activation: index < hiddenLayers.length ? "relu" : null,
      parameter_count: (widths[index] + 1) * width,
    })),
    total_parameters: 42,
  };
}

const trace: ForwardPassTrace = {
  sample_index: 0,
  input_values: [0.25, -0.5],
  expected_label: 1,
  layers: [
    { layer_name: "hidden_0", activation_name: "relu", pre_activations: [-1, 0, 2], activations: [0, 0, 2] },
    { layer_name: "output", activation_name: "sigmoid", pre_activations: [1.4], activations: [0.8] },
  ],
  output_logit: 1.4,
  predicted_probability: 0.8,
  predicted_label: 1,
};

describe("architectureToGraph", () => {
  it("creates stable ordered neuron IDs and dense adjacent-layer edges", () => {
    const first = architectureToGraph(architecture([3, 2]));
    const second = architectureToGraph(architecture([3, 2]));
    expect(first.nodes.map((node) => node.id)).toEqual([
      "input-0", "input-1", "hidden-0-0", "hidden-0-1", "hidden-0-2",
      "hidden-1-0", "hidden-1-1", "output-0",
    ]);
    expect(second).toEqual(first);
    expect(first.edges).toHaveLength(14);
    expect(first.edges[0]?.id).toBe("edge-input-0-hidden-0-0");
    expect(first.nodes[2]?.position.x).toBeGreaterThan(first.nodes[0]?.position.x ?? 0);
    expect(first.nodes[5]?.position.x).toBeGreaterThan(first.nodes[2]?.position.x ?? 0);
  });

  it("bounds rendering for wide layers while preserving the actual count", () => {
    const graph = architectureToGraph(architecture([256]));
    const hidden = graph.nodes.filter((node) => node.data.layerName === "Hidden 1");
    expect(hidden).toHaveLength(24);
    expect(hidden.at(-1)?.id).toBe("hidden-0-overflow");
    expect(hidden.at(-1)?.data.width).toBe(256);
    expect(hidden.at(-1)?.ariaLabel).toContain("233 additional neurons hidden");
  });

  it("maps observed activations and propagation state without inventing edge strength", () => {
    const graph = architectureToGraph(architecture([3]), trace, 1);
    const input = graph.nodes.find((node) => node.id === "input-0")!;
    const active = graph.nodes.find((node) => node.id === "hidden-0-2")!;
    const output = graph.nodes.find((node) => node.id === "output-0")!;

    expect(input.data.flowState).toBe("settled");
    expect(active.data).toMatchObject({ observedValue: 2, signalLevel: 1, signalSign: "positive", flowState: "active" });
    expect(active.ariaLabel).toContain("observed activation 2.0000");
    expect(output.data.flowState).toBe("queued");
    expect(graph.edges.find((edge) => edge.source === "input-0")?.data).toEqual({ flowState: "active" });
    expect(graph.edges.every((edge) => !("weight" in (edge.data ?? {})))).toBe(true);
  });

  it("labels overflow evidence as an aggregate for wide layers", () => {
    const wideTrace: ForwardPassTrace = {
      ...trace,
      layers: [
        { layer_name: "hidden_0", activation_name: "relu", pre_activations: Array(256).fill(1), activations: Array.from({ length: 256 }, (_, index) => index) },
        trace.layers[1]!,
      ],
    };
    const overflow = architectureToGraph(architecture([256]), wideTrace, 1).nodes.find((node) => node.id === "hidden-0-overflow")!;
    expect(overflow.data.isAggregate).toBe(true);
    expect(overflow.ariaLabel).toContain("mean absolute activation");
  });
});
