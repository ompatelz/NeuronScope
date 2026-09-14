import type { ExperimentResponse } from "../api/experiments";

export interface MetricPlotPoint {
  epoch: number;
  value: number;
  x: number;
  y: number;
}

export interface MetricSeries {
  loss: MetricPlotPoint[];
  accuracy: MetricPlotPoint[];
  lossDomain: [number, number];
}

const LEFT = 52;
const RIGHT = 590;
const LOSS_TOP = 18;
const LOSS_BOTTOM = 65;
const ACCURACY_TOP = 91;
const ACCURACY_BOTTOM = 138;

function xPosition(index: number, count: number): number {
  return count === 1 ? (LEFT + RIGHT) / 2 : LEFT + (index / (count - 1)) * (RIGHT - LEFT);
}

// Pure chart transformation is exported for deterministic unit testing.
// eslint-disable-next-line react-refresh/only-export-components
export function buildMetricSeries(
  history: ExperimentResponse["training"]["history"],
): MetricSeries {
  if (history.length === 0) return { loss: [], accuracy: [], lossDomain: [0, 1] };
  const losses = history.map((item) => item.loss);
  let minimum = Math.min(...losses);
  let maximum = Math.max(...losses);
  if (minimum === maximum) {
    const padding = Math.max(Math.abs(minimum) * 0.1, 0.01);
    minimum = Math.max(0, minimum - padding);
    maximum += padding;
  }
  const scale = (value: number, low: number, high: number, top: number, bottom: number) =>
    bottom - ((value - low) / (high - low)) * (bottom - top);
  return {
    lossDomain: [minimum, maximum],
    loss: history.map((item, index) => ({
      epoch: item.epoch, value: item.loss, x: xPosition(index, history.length),
      y: scale(item.loss, minimum, maximum, LOSS_TOP, LOSS_BOTTOM),
    })),
    accuracy: history.map((item, index) => ({
      epoch: item.epoch, value: item.accuracy, x: xPosition(index, history.length),
      y: scale(item.accuracy, 0, 1, ACCURACY_TOP, ACCURACY_BOTTOM),
    })),
  };
}

function points(series: MetricPlotPoint[]): string {
  return series.map((point) => `${point.x},${point.y}`).join(" ");
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
      <svg className="metrics-chart" viewBox="0 0 600 156" role="img" aria-labelledby="metrics-chart-title metrics-chart-description">
        <title id="metrics-chart-title">Training loss and accuracy by epoch</title>
        <desc id="metrics-chart-description">Two separate vertical bands show raw loss and accuracy observations without smoothing.</desc>
        <g className="metric-grid">
          <line x1={LEFT} x2={RIGHT} y1={LOSS_TOP} y2={LOSS_TOP} />
          <line x1={LEFT} x2={RIGHT} y1={LOSS_BOTTOM} y2={LOSS_BOTTOM} />
          <line x1={LEFT} x2={RIGHT} y1={ACCURACY_TOP} y2={ACCURACY_TOP} />
          <line x1={LEFT} x2={RIGHT} y1={ACCURACY_BOTTOM} y2={ACCURACY_BOTTOM} />
        </g>
        <g className="metric-labels" aria-hidden="true">
          <text x="6" y="25">LOSS</text><text x="6" y="98">ACC.</text>
          <text x={RIGHT} y="153" textAnchor="end">EPOCH {final.epoch}</text>
          <text x={LEFT} y="153">{training.history[0]?.epoch}</text>
        </g>
        <polyline className="metric-trace loss-trace" points={points(series.loss)} />
        <polyline className="metric-trace accuracy-trace" points={points(series.accuracy)} />
        {series.loss.map((point) => <circle className="metric-point loss-point" key={`loss-${point.epoch}`} cx={point.x} cy={point.y} r="2.4"><title>{`Epoch ${point.epoch} loss: ${point.value}`}</title></circle>)}
        {series.accuracy.map((point) => <circle className="metric-point accuracy-point" key={`accuracy-${point.epoch}`} cx={point.x} cy={point.y} r="2.4"><title>{`Epoch ${point.epoch} accuracy: ${(point.value * 100).toFixed(2)}%`}</title></circle>)}
      </svg>
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
