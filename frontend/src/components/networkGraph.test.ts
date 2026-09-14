import { describe, expect, it } from "vitest";

import type { ExperimentResponse } from "../api/experiments";
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
});
