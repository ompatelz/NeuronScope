import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { parseHiddenLayers } from "./config";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const response = {
  dataset: { config: { kind: "two_moons", samples: 40, noise: 0.1, seed: 42 }, points: Array.from({ length: 40 }, (_, index) => ({ x: index, y: 0, label: index % 2 })) },
  architecture: { input_size: 2, hidden_layers: [4], output_size: 1, layers: [{ name: "hidden_0", input_size: 2, output_size: 4, activation: "relu", parameter_count: 12 }, { name: "output", input_size: 4, output_size: 1, activation: null, parameter_count: 5 }], total_parameters: 17 },
  training: { config: { optimizer: "adam", learning_rate: 0.01, epochs: 2, instrumentation: true }, history: [{ epoch: 1, loss: 0.7, accuracy: 0.5 }, { epoch: 2, loss: 0.25, accuracy: 0.9 }], instrumentation: [], final_loss: 0.25, final_accuracy: 0.9 },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the complete accessible configuration and honest idle state", () => {
    render(<App />);
    expect(screen.getByRole("complementary", { name: "Experiment configuration" })).toBeTruthy();
    expect(screen.getByLabelText("Kind")).toBeTruthy();
    expect(screen.getByLabelText("Hidden layers")).toBeTruthy();
    expect(screen.getByLabelText("Optimizer")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run experiment" })).toBeTruthy();
    expect(screen.getByText("No observed architecture yet")).toBeTruthy();
  });

  it("posts selected configuration and presents observed results", async () => {
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => response });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    fireEvent.change(screen.getByLabelText("Samples"), { target: { value: "40" } });
    fireEvent.change(screen.getByLabelText("Hidden layers"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByText("Run completed");
    expect(screen.getAllByText("90.0%", { exact: false })).toHaveLength(2);
    expect(screen.getByText("h1.1")).toBeTruthy();
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as { dataset: { samples: number }; model: { hidden_layers: number[] }; training: { epochs: number } };
    expect(body.dataset.samples).toBe(40);
    expect(body.model.hidden_layers).toEqual([4]);
    expect(body.training.epochs).toBe(2);
    const neuron = await screen.findByLabelText("Hidden 1, neuron 1 of 4");
    fireEvent.click(neuron);
    expect(screen.getByText("Hidden 1 · h1.1")).toBeTruthy();
  });

  it("shows API failures as an alert", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: "Invalid experiment" }) }));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Invalid experiment"));
  });
});

describe("parseHiddenLayers", () => {
  it("parses valid widths and rejects invalid architecture input", () => {
    expect(parseHiddenLayers("16, 8, 4")).toEqual([16, 8, 4]);
    expect(() => parseHiddenLayers("16, 0")).toThrow(/between 1 and 256/);
    expect(() => parseHiddenLayers(" ")).toThrow(/between 1 and 256/);
  });
});
