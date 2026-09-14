import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentRequest, ExperimentResponse } from "../api/experiments";
import { appendRun, resolveRunResult, RunComparison, type RunRecord } from "./runComparison";

afterEach(cleanup);

function record(id: number, accuracy = 0.8): RunRecord {
  const request = {
    dataset: { kind: "xor", samples: 40, noise: 0.1, seed: id },
    model: { input_size: 2, hidden_layers: [4], output_size: 1, activation: "tanh", initialization: "xavier", seed: id },
    training: { optimizer: "sgd", learning_rate: 0.02, epochs: 10, instrumentation: true },
    boundary: { resolution: 24 },
  } satisfies ExperimentRequest;
  const result = {
    training: { config: request.training, history: [], instrumentation: [], final_loss: 0.12345, final_accuracy: accuracy },
    diagnostics: id % 2 ? [{ type: "vanishing_gradient" }] : [],
  } as unknown as ExperimentResponse;
  return { id, label: `Run ${id}`, request, result };
}

describe("appendRun", () => {
  it("appends in order and caps local history at five runs", () => {
    const history = Array.from({ length: 7 }, (_, index) => record(index + 1)).reduce(appendRun, [] as RunRecord[]);
    expect(history.map((run) => run.label)).toEqual(["Run 3", "Run 4", "Run 5", "Run 6", "Run 7"]);
  });

  it("resolves a selected past response without changing the latest response", () => {
    const first = record(1, 0.6);
    const latest = record(2, 0.9);
    expect(resolveRunResult([first, latest], 1, latest.result)).toBe(first.result);
    expect(resolveRunResult([first, latest], null, latest.result)).toBe(latest.result);
  });
});

describe("RunComparison", () => {
  it("shows exact configuration/result metadata and selects a recorded run", () => {
    const onSelect = vi.fn();
    render(<RunComparison runs={[record(1, 0.875)]} selectedId={1} onSelect={onSelect} onClear={() => {}} />);
    const row = screen.getByRole("button", { name: "Run 1" }).closest("tr");
    expect(row?.textContent).toContain("tanh");
    expect(row?.textContent).toContain("xavier");
    expect(row?.textContent).toContain("sgd");
    expect(row?.textContent).toContain("0.02");
    expect(row?.textContent).toContain("0.12345");
    expect(row?.textContent).toContain("87.5%");
    expect(row?.textContent).toContain("1");
    fireEvent.click(screen.getByRole("button", { name: "Run 1" }));
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("clears history through an explicit local action", () => {
    const onClear = vi.fn();
    render(<RunComparison runs={[record(1)]} selectedId={null} onSelect={() => {}} onClear={onClear} />);
    fireEvent.click(screen.getByRole("button", { name: "Clear history" }));
    expect(onClear).toHaveBeenCalledOnce();
  });
});
