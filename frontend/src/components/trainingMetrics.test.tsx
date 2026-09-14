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
  it("handles empty and single-epoch histories without invalid coordinates", () => {
    expect(buildMetricSeries([])).toEqual({ loss: [], accuracy: [], lossDomain: [0, 1] });
    const single = buildMetricSeries([{ epoch: 1, loss: 0.5, accuracy: 0.75 }]);
    expect(single.loss).toHaveLength(1);
    expect(Number.isFinite(single.loss[0]?.x)).toBe(true);
    expect(Number.isFinite(single.loss[0]?.y)).toBe(true);
  });

  it("preserves raw values and chronological point order", () => {
    const history = [
      { epoch: 1, loss: 0.8, accuracy: 0.4 },
      { epoch: 2, loss: 0.43, accuracy: 0.72 },
      { epoch: 3, loss: 0.2, accuracy: 0.91 },
    ];
    const series = buildMetricSeries(history);
    expect(series.loss.map((point) => point.value)).toEqual([0.8, 0.43, 0.2]);
    expect(series.accuracy.map((point) => point.value)).toEqual([0.4, 0.72, 0.91]);
    expect(series.loss[0]!.x).toBeLessThan(series.loss[2]!.x);
  });
});

describe("TrainingMetricsChart", () => {
  it("renders an honest empty response", () => {
    render(<TrainingMetricsChart training={training([])} />);
    expect(screen.getByText(/contains no metric observations/i)).toBeTruthy();
  });

  it("exposes exact raw values and final metadata", () => {
    const { container } = render(<TrainingMetricsChart training={training([
      { epoch: 1, loss: 0.7, accuracy: 0.5 },
      { epoch: 2, loss: 0.25, accuracy: 0.9 },
    ])} />);
    expect(screen.getByRole("img", { name: /training loss and accuracy/i })).toBeTruthy();
    expect(screen.getAllByText("0.25000")).toHaveLength(2);
    expect(screen.getAllByText("90.0%")).toHaveLength(2);
    const titles = Array.from(container.querySelectorAll("circle title")).map((node) => node.textContent);
    expect(titles).toContain("Epoch 1 loss: 0.7");
    expect(titles).toContain("Epoch 2 accuracy: 90.00%");
  });
});
