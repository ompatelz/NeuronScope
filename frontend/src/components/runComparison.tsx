import type { ExperimentRequest, ExperimentResponse } from "../api/experiments";

export interface RunRecord {
  id: number;
  label: string;
  request: ExperimentRequest;
  result: ExperimentResponse;
}

// Pure history helpers are exported for deterministic unit testing.
// eslint-disable-next-line react-refresh/only-export-components
export function appendRun(history: RunRecord[], record: RunRecord): RunRecord[] {
  return [...history, record].slice(-5);
}

// eslint-disable-next-line react-refresh/only-export-components
export function resolveRunResult(
  runs: RunRecord[], selectedId: number | null, latest: ExperimentResponse | null,
): ExperimentResponse | null {
  return runs.find((run) => run.id === selectedId)?.result ?? latest;
}

export function RunComparison({ runs, selectedId, onSelect, onClear }: {
  runs: RunRecord[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onClear: () => void;
}) {
  if (!runs.length) return <p className="signal-unavailable">Completed runs will appear here for local comparison.</p>;
  return (
    <section className="run-comparison" aria-labelledby="run-comparison-title">
      <div className="comparison-heading"><div><h3 id="run-comparison-title">Run comparison</h3><span>Stored locally · latest five</span></div><button type="button" onClick={onClear}>Clear history</button></div>
      <div className="comparison-table-wrap"><table>
        <caption className="sr-only">Completed experiment comparison</caption>
        <thead><tr><th>Run</th><th>Activation</th><th>Init</th><th>Optimizer</th><th>LR</th><th>Loss</th><th>Accuracy</th><th>Diagnostics</th></tr></thead>
        <tbody>{runs.map((run) => <tr key={run.id} className={run.id === selectedId ? "is-active" : ""}>
          <td><button type="button" aria-pressed={run.id === selectedId} onClick={() => onSelect(run.id)}>{run.label}</button></td>
          <td>{run.request.model.activation}</td><td>{run.request.model.initialization}</td><td>{run.request.training.optimizer}</td><td>{run.request.training.learning_rate}</td>
          <td>{run.result.training.final_loss.toFixed(5)}</td><td>{(run.result.training.final_accuracy * 100).toFixed(1)}%</td><td>{run.result.diagnostics.length}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="comparison-note">Selecting a run reopens its recorded graph, boundary, metrics, and layer signals without retraining.</p>
    </section>
  );
}
