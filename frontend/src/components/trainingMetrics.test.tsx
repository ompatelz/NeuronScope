import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ExperimentResponse } from "../api/experiments";
import { buildMetricSeries, TrainingMetricsChart } from "./trainingMetrics";

type Training = ExperimentResponse["training"];
function training(history: Training["history"]): Training {
  const final = history.at(-1) ?? { epoch: 0, loss: 0, accuracy: 0 };
  return {
    config: { optimizer: "adam", learning_rate: 0.01, epochs: Math.max(1, final.epoch), instrumentation: true },
    history, instrumentation: [], final_loss: final.loss, final_accuracy: final.accuracy,
  };
}

describe("buildMetricSeries", () => {
  it("handles empty and single-epoch histories without invalid domains", () => {
    expect(buildMetricSeries([])).toEqual({ data: [], lossDomain: [0, 1] });
    const single = buildMetricSeries([{ epoch: 1, loss: 0.5, accuracy: 0.75 }]);
    expect(single.data).toEqual([{ epoch: 1, loss: 0.5, accuracy: 75 }]);
    expect(single.lossDomain[0]).toBeLessThan(single.lossDomain[1]);
  });

  it("preserves raw values and chronological point order", () => {
    const history = [
      { epoch: 1, loss: 0.8, accuracy: 0.4 },
      { epoch: 2, loss: 0.43, accuracy: 0.72 },
      { epoch: 3, loss: 0.2, accuracy: 0.91 },
    ];
    const series = buildMetricSeries(history);
    expect(series.data.map((point) => point.loss)).toEqual([0.8, 0.43, 0.2]);
    expect(series.data.map((point) => point.accuracy)).toEqual([40, 72, 91]);
    expect(series.data.map((point) => point.epoch)).toEqual([1, 2, 3]);
  });
});

describe("TrainingMetricsChart", () => {
  it("renders an honest empty response", () => {
    render(<TrainingMetricsChart training={training([])} />);
    expect(screen.getByText(/contains no metric observations/i)).toBeTruthy();
  });

  it("exposes exact raw values and final metadata", () => {
    render(<TrainingMetricsChart training={training([
      { epoch: 1, loss: 0.7, accuracy: 0.5 },
      { epoch: 2, loss: 0.25, accuracy: 0.9 },
    ])} />);
    expect(screen.getByRole("img", { name: /interactive training loss and accuracy/i })).toBeTruthy();
    expect(screen.getAllByText("0.25000")).toHaveLength(2);
    expect(screen.getAllByText("90.0%")).toHaveLength(2);
  });
});
