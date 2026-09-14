import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ExperimentResponse } from "../api/experiments";

export interface MetricPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface MetricSeries {
  data: MetricPoint[];
  lossDomain: [number, number];
}

// Pure transformation retained for deterministic unit testing.
// eslint-disable-next-line react-refresh/only-export-components
export function buildMetricSeries(
  history: ExperimentResponse["training"]["history"],
): MetricSeries {
  if (history.length === 0) return { data: [], lossDomain: [0, 1] };
  const losses = history.map((item) => item.loss);
  let minimum = Math.min(...losses);
  let maximum = Math.max(...losses);
  if (minimum === maximum) {
    const padding = Math.max(Math.abs(minimum) * 0.1, 0.01);
    minimum = Math.max(0, minimum - padding);
    maximum += padding;
  }
  return {
    lossDomain: [minimum, maximum],
    data: history.map((item) => ({
      epoch: item.epoch,
      loss: item.loss,
      accuracy: item.accuracy * 100,
    })),
  };
}

export function TrainingMetricsChart({ training }: {
  training: ExperimentResponse["training"];
}) {
  const series = buildMetricSeries(training.history);
  if (!training.history.length) {
    return <p className="panel-copy">This completed response contains no metric observations.</p>;
  }
  const final = training.history.at(-1)!;
  const recent = training.history.slice(-5);
  return (
    <div className="training-metrics">
      <div className="metrics-meta" aria-label="Run progress and final metrics">
        <span>Epoch <strong>{final.epoch}</strong> / {training.config.epochs}</span>
        <span>Final loss <strong>{training.final_loss.toFixed(5)}</strong></span>
        <span>Final accuracy <strong>{(training.final_accuracy * 100).toFixed(1)}%</strong></span>
        <span>{training.config.optimizer.toUpperCase()} · lr {training.config.learning_rate}</span>
      </div>
      <div className="metrics-chart" role="img" aria-label="Interactive training loss and accuracy by epoch">
        <LineChart
          accessibilityLayer
          data={series.data}
          margin={{ top: 10, right: 12, bottom: 2, left: 0 }}
          responsive
          style={{ width: "100%", height: 172 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 5" vertical={false} />
          <XAxis dataKey="epoch" minTickGap={28} stroke="var(--muted)" tick={{ fontSize: 9 }} tickLine={false} />
          <YAxis yAxisId="loss" domain={series.lossDomain} stroke="var(--accent)" tick={{ fontSize: 9 }} tickFormatter={(value: number) => value.toFixed(3)} width={48} />
          <YAxis yAxisId="accuracy" orientation="right" domain={[0, 100]} stroke="var(--success)" tick={{ fontSize: 9 }} tickFormatter={(value: number) => `${value}%`} width={42} />
          <Tooltip
            contentStyle={{ background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 11 }}
            cursor={{ stroke: "var(--muted)", strokeDasharray: "3 3" }}
            labelStyle={{ color: "var(--foreground)", fontFamily: "var(--font-mono)" }}
          />
          <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />
          <ReferenceLine x={final.epoch} stroke="var(--muted)" strokeDasharray="2 3" />
          <Line yAxisId="loss" type="monotone" dataKey="loss" name="Loss" stroke="var(--accent)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          <Line yAxisId="accuracy" type="monotone" dataKey="accuracy" name="Accuracy %" stroke="var(--success)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
        </LineChart>
      </div>
      <div className="metrics-table-wrap">
        <table>
          <caption>Most recent raw observations</caption>
          <thead><tr><th>Epoch</th><th>Loss</th><th>Accuracy</th></tr></thead>
          <tbody>{recent.map((metric) => <tr key={metric.epoch}><td>{metric.epoch}</td><td>{metric.loss.toFixed(5)}</td><td>{(metric.accuracy * 100).toFixed(1)}%</td></tr>)}</tbody>
        </table>
      </div>
    </div>
  );
}
