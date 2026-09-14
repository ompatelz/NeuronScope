import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { parseHiddenLayers, validateConfig } from "./config";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const response = {
  dataset: { config: { kind: "two_moons", samples: 40, noise: 0.1, seed: 42 }, points: Array.from({ length: 40 }, (_, index) => ({ x: index, y: 0, label: index % 2 })) },
  architecture: { input_size: 2, hidden_layers: [4], output_size: 1, layers: [{ name: "hidden_0", input_size: 2, output_size: 4, activation: "relu", parameter_count: 12 }, { name: "output", input_size: 4, output_size: 1, activation: null, parameter_count: 5 }], total_parameters: 17 },
  training: { config: { optimizer: "adam", learning_rate: 0.01, epochs: 2, instrumentation: true }, history: [{ epoch: 1, loss: 0.7, accuracy: 0.5 }, { epoch: 2, loss: 0.25, accuracy: 0.9 }], instrumentation: [], final_loss: 0.25, final_accuracy: 0.9 },
  boundary: { resolution: 2, x_coordinates: [0, 39], y_coordinates: [-1, 1], probabilities: [0.1, 0.2, 0.8, 0.9] },
  diagnostics: [],
  playback: { resolution: 2, x_coordinates: [0, 39], y_coordinates: [-1, 1], snapshots: [
    { epoch: 1, metrics: { epoch: 1, loss: 0.7, accuracy: 0.5 }, instrumentation: null, probabilities: [0.2, 0.3, 0.7, 0.8] },
    { epoch: 2, metrics: { epoch: 2, loss: 0.25, accuracy: 0.9 }, instrumentation: null, probabilities: [0.1, 0.2, 0.8, 0.9] },
  ] },
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
    expect(screen.getAllByText("90.0%", { exact: false }).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("h1.1")).toBeTruthy();
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as { dataset: { samples: number }; model: { hidden_layers: number[] }; training: { epochs: number } };
    expect(body.dataset.samples).toBe(40);
    expect(body.model.hidden_layers).toEqual([4]);
    expect(body.training.epochs).toBe(2);
    const neuron = await screen.findByLabelText("Hidden 1, neuron 1 of 4");
    fireEvent.click(neuron);
    expect(screen.getByText("Hidden 1 · h1.1")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Boundary" }));
    expect(screen.getByRole("img", { name: /^Final model decision boundary/ })).toBeTruthy();
    fireEvent.change(screen.getByRole("slider", { name: "Training playback epoch" }), { target: { value: "1" } });
    expect(screen.getByLabelText("Run progress and final metrics").textContent).toContain("50.0%");
  });

  it("shows API failures as an alert", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: "Invalid experiment" }) }));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Invalid experiment"));
    fireEvent.change(screen.getByLabelText("Samples"), { target: { value: "41" } });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("Ready to train")).toBeTruthy();
  });

  it("explains invalid configuration inline and does not call the API", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    fireEvent.change(screen.getByLabelText("Hidden layers"), { target: { value: "8, 0" } });
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect((await screen.findByRole("alert")).textContent).toContain("between 1 and 256");
    const hiddenLayers = screen.getByLabelText("Hidden layers");
    expect(hiddenLayers.getAttribute("aria-invalid")).toBe("true");
    await waitFor(() => expect(document.activeElement).toBe(hiddenLayers));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("cancels a request and ignores its late response", async () => {
    let resolveFetch!: (value: { ok: boolean; json: () => Promise<typeof response> }) => void;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      void input;
      void init;
      return new Promise((resolve) => { resolveFetch = resolve; });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    expect(await screen.findByRole("button", { name: "Cancel run" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel run" }));
    expect(await screen.findByText("Run cancelled")).toBeTruthy();
    expect(((fetchMock.mock.calls[0]?.[1] as RequestInit).signal as AbortSignal).aborted).toBe(true);
    await act(async () => {
      resolveFetch({ ok: true, json: async () => response });
      await Promise.resolve();
    });
    expect(screen.queryByText("Run completed")).toBeNull();
  });

  it("selects a recorded run and restores its playback evidence", async () => {
    vi.stubGlobal("ResizeObserver", ResizeObserverStub);
    const second = {
      ...response,
      training: { ...response.training, final_loss: 0.4, final_accuracy: 0.7, history: [{ epoch: 1, loss: 0.6, accuracy: 0.55 }, { epoch: 2, loss: 0.4, accuracy: 0.7 }] },
      playback: { ...response.playback, snapshots: response.playback.snapshots.map((snapshot) => ({ ...snapshot, metrics: { ...snapshot.metrics, accuracy: snapshot.epoch === 2 ? 0.7 : snapshot.metrics.accuracy } })) },
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => response })
      .mockResolvedValueOnce({ ok: true, json: async () => second });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await screen.findByText("Run completed");
    fireEvent.click(screen.getByRole("button", { name: "Run experiment" }));
    await waitFor(() => expect(screen.getByLabelText("Run progress and final metrics").textContent).toContain("70.0%"));
    fireEvent.click(screen.getByRole("tab", { name: "Runs" }));
    fireEvent.click(screen.getByRole("button", { name: "Run 1" }));
    await waitFor(() => expect(screen.getByLabelText("Run progress and final metrics").textContent).toContain("90.0%"));
  });
});

describe("parseHiddenLayers", () => {
  it("parses valid widths and rejects invalid architecture input", () => {
    expect(parseHiddenLayers("16, 8, 4")).toEqual([16, 8, 4]);
    expect(() => parseHiddenLayers("16, 0")).toThrow(/between 1 and 256/);
    expect(() => parseHiddenLayers(" ")).toThrow(/between 1 and 256/);
  });

  it("validates every bounded numeric control before serialization", () => {
    const valid = { samples: 200, noise: 0.1, seed: 42, hiddenLayers: "8, 8", learningRate: 0.01, epochs: 200 };
    expect(validateConfig(valid)).toBeNull();
    expect(validateConfig({ ...valid, samples: Number.NaN })?.field).toBe("samples");
    expect(validateConfig({ ...valid, noise: 0.6 })?.field).toBe("noise");
    expect(validateConfig({ ...valid, learningRate: 0 })?.field).toBe("learningRate");
    expect(validateConfig({ ...valid, epochs: 5001 })?.field).toBe("epochs");
  });
});
