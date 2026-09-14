import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResponse, ForwardPassTrace } from "../api/experiments";
import { NetworkGraph } from "./networkGraph";

class ResizeObserverStub { observe() {} unobserve() {} disconnect() {} }

const architecture: ExperimentResponse["architecture"] = {
  input_size: 2, hidden_layers: [2], output_size: 1, total_parameters: 9,
  layers: [
    { name: "hidden_0", input_size: 2, output_size: 2, activation: "relu", parameter_count: 6 },
    { name: "output", input_size: 2, output_size: 1, activation: null, parameter_count: 3 },
  ],
};
const trace: ForwardPassTrace = {
  sample_index: 2, input_values: [0.4, -0.2], expected_label: 1,
  layers: [
    { layer_name: "hidden_0", activation_name: "relu", pre_activations: [0.7, -0.1], activations: [0.7, 0] },
    { layer_name: "output", activation_name: "sigmoid", pre_activations: [1.2], activations: [0.7685] },
  ],
  output_logit: 1.2, predicted_probability: 0.7685, predicted_label: 1,
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("ResizeObserver", ResizeObserverStub);
});
afterEach(() => { cleanup(); vi.runOnlyPendingTimers(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("NetworkGraph forward-pass animation", () => {
  it("plays, pauses, and steps through the observed layers", () => {
    render(<div style={{ width: 900, height: 600 }}><NetworkGraph architecture={architecture} trace={trace} epoch={12} onSelect={vi.fn()} /></div>);
    expect(screen.getByRole("region", { name: "Forward pass animation" })).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("Input layer");
    expect(screen.getByText("Sample #3")).toBeTruthy();
    act(() => vi.advanceTimersByTime(720));
    expect(screen.getByRole("status").textContent).toContain("Hidden 1 · relu");
    fireEvent.click(screen.getByRole("button", { name: "Pause forward pass" }));
    act(() => vi.advanceTimersByTime(2_000));
    expect(screen.getByRole("status").textContent).toContain("Hidden 1 · relu");
    fireEvent.click(screen.getByRole("button", { name: "Next layer" }));
    expect(screen.getByRole("status").textContent).toContain("Output prediction");
    expect(screen.getByText("76.8%")).toBeTruthy();
  });

  it("honors reduced motion while keeping manual inspection available", () => {
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
    render(<div style={{ width: 900, height: 600 }}><NetworkGraph architecture={architecture} trace={trace} epoch={12} onSelect={vi.fn()} /></div>);
    expect(screen.getByRole("status").textContent).toContain("Output prediction");
    expect(screen.queryByRole("button", { name: "Pause forward pass" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Previous layer" }));
    expect(screen.getByRole("status").textContent).toContain("Hidden 1 · relu");
  });
});
