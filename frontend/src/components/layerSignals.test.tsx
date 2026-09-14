import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { EpochInstrumentation } from "../api/experiments";
import { formatSignal, LayerSignals } from "./layerSignals";

const instrumentation: EpochInstrumentation[] = [{
  epoch: 7,
  layers: [
    {
      layer_name: "hidden_0", weight_norm: 1.25,
      gradients: { norm: 0.0042, mean: -0.0003, std: 0.0011, nonfinite_count: 0 },
      activation: { mean: 0.42, std: 0.31, minimum: 0, maximum: 1.4, zero_percentage: 37.5, nonfinite_count: 0 },
    },
    {
      layer_name: "output", weight_norm: null,
      gradients: { norm: null, mean: null, std: null, nonfinite_count: 2 },
      activation: null,
    },
  ],
}];

afterEach(cleanup);

describe("LayerSignals", () => {
  it("shows real latest-epoch gradient, weight, and activation values", () => {
    render(<LayerSignals instrumentation={instrumentation} />);
    expect(screen.getByText("Epoch 7")).toBeTruthy();
    expect(screen.getByLabelText("Hidden 1 signals").textContent).toContain("0.0042");
    expect(screen.getByLabelText("Hidden 1 signals").textContent).toContain("37.5% zero");
    expect(screen.getByLabelText("37.5% zero activations")).toBeTruthy();
  });

  it("renders missing output activations and non-finite evidence explicitly", () => {
    render(<LayerSignals instrumentation={instrumentation} selectedLayerName="output" />);
    const output = screen.getByLabelText("Output signals");
    expect(output.textContent).toContain("2 non-finite");
    expect(output.textContent).toContain("Unavailable");
    expect(output.textContent).toContain("Activation statistics unavailable");
    expect(screen.queryByLabelText("Hidden 1 signals")).toBeNull();
  });

  it("handles missing instrumentation and input-layer selection honestly", () => {
    const { rerender } = render(<LayerSignals instrumentation={[]} />);
    expect(screen.getByText(/not collected/i)).toBeTruthy();
    rerender(<LayerSignals instrumentation={instrumentation} selectedLayerName="__input__" />);
    expect(screen.getByText(/input has no learned-layer instrumentation/i)).toBeTruthy();
  });
});

describe("formatSignal", () => {
  it("distinguishes unavailable, ordinary, and very small values", () => {
    expect(formatSignal(null)).toBe("Unavailable");
    expect(formatSignal(1.25)).toBe("1.2500");
    expect(formatSignal(0.000012)).toBe("1.200e-5");
  });
});
