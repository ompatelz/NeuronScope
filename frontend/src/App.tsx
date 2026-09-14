import { Tabs } from "@base-ui/react/tabs";
import {
  Activity, AlertCircle, BarChart3, Binary, Braces, Bug, CheckCircle2,
  CirclePlay, Database, LoaderCircle, Network, SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";
import { type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";

import {
  createExperiment,
  type ActivationName, type DatasetKind, type ExperimentRequest,
  type ExperimentResponse, type InitializationName, type OptimizerName,
  type PlaybackSnapshot,
} from "./api/experiments";
import { DecisionBoundary } from "./components/decisionBoundary";
import { DiagnosticsPanel } from "./components/diagnosticsPanel";
import { LayerSignals } from "./components/layerSignals";
import { NetworkGraph, type ArchitectureNodeData } from "./components/networkGraph";
import { PlaybackScrubber } from "./components/playbackScrubber";
import { appendRun, resolveRunResult, RunComparison, type RunRecord } from "./components/runComparison";
import { TrainingMetricsChart } from "./components/trainingMetrics";
import { parseHiddenLayers, validateConfig, type ConfigValidationError } from "./config";

interface WorkbenchConfig {
  dataset: DatasetKind;
  samples: number;
  noise: number;
  seed: number;
  hiddenLayers: string;
  activation: ActivationName;
  initialization: InitializationName;
  optimizer: OptimizerName;
  learningRate: number;
  epochs: number;
}

type RunState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "cancelled" }
  | { status: "error"; message: string }
  | { status: "completed"; result: ExperimentResponse };

const initialConfig: WorkbenchConfig = {
  dataset: "two_moons", samples: 200, noise: 0.12, seed: 42,
  hiddenLayers: "8, 8", activation: "relu", initialization: "he",
  optimizer: "adam", learningRate: 0.01, epochs: 200,
};

function buildRequest(config: WorkbenchConfig): ExperimentRequest {
  return {
    dataset: { kind: config.dataset, samples: config.samples, noise: config.noise, seed: config.seed },
    model: {
      input_size: 2, hidden_layers: parseHiddenLayers(config.hiddenLayers), output_size: 1,
      activation: config.activation, initialization: config.initialization, seed: config.seed,
    },
    training: {
      optimizer: config.optimizer, learning_rate: config.learningRate,
      epochs: config.epochs, instrumentation: true,
    },
    boundary: { resolution: 48 },
    playback: { max_snapshots: 12 },
  };
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="field-label"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}

function Heading({ icon: Icon, title, description }: { icon: LucideIcon; title: string; description?: string }) {
  return (
    <div className="panel-heading">
      <Icon aria-hidden="true" />
      <div><h2>{title}</h2>{description && <p>{description}</p>}</div>
    </div>
  );
}

function Configuration({ config, onChange, error }: {
  config: WorkbenchConfig;
  onChange: <K extends keyof WorkbenchConfig>(key: K, value: WorkbenchConfig[K]) => void;
  error: ConfigValidationError | null;
}) {
  const invalid = (field: ConfigValidationError["field"]) => error?.field === field;
  const describedBy = (field: ConfigValidationError["field"]) => invalid(field) ? "configuration-error" : undefined;
  return (
    <aside className="config-rail" aria-label="Experiment configuration">
      <Heading icon={SlidersHorizontal} title="Configuration" description="One synchronous observed run" />
      {error && <p id="configuration-error" className="configuration-error" role="alert"><AlertCircle aria-hidden="true" />{error.message}</p>}
      <fieldset className="control-group">
        <legend><Database /> Dataset</legend>
        <Field label="Kind"><select value={config.dataset} onChange={(e) => onChange("dataset", e.target.value as DatasetKind)}>
          <option value="two_moons">Two moons</option><option value="circles">Circles</option>
          <option value="xor">XOR</option><option value="spiral">Spiral</option>
        </select></Field>
        <div className="control-pair">
          <Field label="Samples"><input type="number" min="40" max="2000" value={config.samples} aria-invalid={invalid("samples")} aria-describedby={describedBy("samples")} onChange={(e) => onChange("samples", e.target.valueAsNumber)} /></Field>
          <Field label="Noise"><input type="number" min="0" max="0.5" step="0.01" value={config.noise} aria-invalid={invalid("noise")} aria-describedby={describedBy("noise")} onChange={(e) => onChange("noise", e.target.valueAsNumber)} /></Field>
        </div>
        <Field label="Seed" hint="Shared by dataset and model"><input type="number" min="0" max="2147483647" value={config.seed} aria-invalid={invalid("seed")} aria-describedby={describedBy("seed")} onChange={(e) => onChange("seed", e.target.valueAsNumber)} /></Field>
      </fieldset>

      <fieldset className="control-group">
        <legend><Network /> Network</legend>
        <Field label="Hidden layers" hint="Comma-separated widths; maximum 8">
          <input aria-label="Hidden layers" type="text" inputMode="numeric" value={config.hiddenLayers} aria-invalid={invalid("hiddenLayers")} aria-describedby={describedBy("hiddenLayers")} onChange={(e) => onChange("hiddenLayers", e.target.value)} placeholder="8, 8" />
        </Field>
        <div className="control-pair">
          <Field label="Activation"><select value={config.activation} onChange={(e) => onChange("activation", e.target.value as ActivationName)}>
            <option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option><option value="tanh">Tanh</option>
          </select></Field>
          <Field label="Initialization"><select value={config.initialization} onChange={(e) => onChange("initialization", e.target.value as InitializationName)}>
            <option value="default">Default</option><option value="xavier">Xavier</option><option value="he">He</option>
          </select></Field>
        </div>
      </fieldset>

      <fieldset className="control-group">
        <legend><Binary /> Training</legend>
        <Field label="Optimizer"><select value={config.optimizer} onChange={(e) => onChange("optimizer", e.target.value as OptimizerName)}>
          <option value="adam">Adam</option><option value="sgd">SGD</option>
        </select></Field>
        <div className="control-pair">
          <Field label="Learning rate"><input type="number" min="0.0001" max="1" step="0.0001" value={config.learningRate} aria-invalid={invalid("learningRate")} aria-describedby={describedBy("learningRate")} onChange={(e) => onChange("learningRate", e.target.valueAsNumber)} /></Field>
          <Field label="Epochs"><input type="number" min="1" max="5000" value={config.epochs} aria-invalid={invalid("epochs")} aria-describedby={describedBy("epochs")} onChange={(e) => onChange("epochs", e.target.valueAsNumber)} /></Field>
        </div>
      </fieldset>
    </aside>
  );
}

function StateSummary({ state }: { state: RunState }) {
  const contents = state.status === "loading" ? [LoaderCircle, "Training in progress", "The API is running the configured experiment."] as const
    : state.status === "cancelled" ? [AlertCircle, "Run cancelled", "No result was recorded. You can adjust the configuration and run again."] as const
    : state.status === "error" ? [AlertCircle, "Run failed", state.message] as const
    : state.status === "completed" ? [CheckCircle2, "Run completed", `Accuracy ${(state.result.training.final_accuracy * 100).toFixed(1)}% · Loss ${state.result.training.final_loss.toFixed(4)}`] as const
    : [CirclePlay, "Ready to train", "Review the configuration, then start a real run."] as const;
  const [Icon, title, text] = contents;
  return (
    <div className={`state-summary state-${state.status}`} role={state.status === "error" ? "alert" : "status"}>
      <Icon aria-hidden="true" className={state.status === "loading" ? "animate-spin" : ""} />
      <div><strong>{title}</strong><span>{text}</span></div>
    </div>
  );
}

function EmptyStage({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) {
  return <div className="empty-stage"><Icon aria-hidden="true" /><strong>{title}</strong><p>{text}</p></div>;
}

function Stage({ state, onSelect, snapshot }: { state: RunState; onSelect: (node: ArchitectureNodeData | null) => void; snapshot: PlaybackSnapshot | null }) {
  const result = state.status === "completed" ? state.result : null;
  const boundary = result && snapshot ? {
    resolution: result.playback.resolution,
    x_coordinates: result.playback.x_coordinates,
    y_coordinates: result.playback.y_coordinates,
    probabilities: snapshot.probabilities,
  } : result?.boundary;
  return (
    <section className="stage" aria-label="Visualization stage">
      <Tabs.Root defaultValue="network" className="tab-root">
        <Tabs.List className="tab-list" aria-label="Visualizations">
          <Tabs.Tab value="network"><Network /> Network</Tabs.Tab>
          <Tabs.Tab value="boundary"><Braces /> Boundary</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="network" className="stage-panel">
          {result ? <NetworkGraph architecture={result.architecture} onSelect={onSelect} />
            : <EmptyStage icon={Network} title="No observed architecture yet" text="Run an experiment to inspect the architecture returned by the training API." />}
        </Tabs.Panel>
        <Tabs.Panel value="boundary" className="stage-panel">
          {result && boundary ? <DecisionBoundary boundary={boundary} points={result.dataset.points} />
            : <EmptyStage icon={Braces} title="No decision boundary yet" text="A prediction grid will appear here when that real capability is implemented." />}
        </Tabs.Panel>
      </Tabs.Root>
    </section>
  );
}

function Inspector({ state, selected, runs, selectedRunId, onSelectRun, onClearRuns, snapshot }: {
  state: RunState;
  selected: ArchitectureNodeData | null;
  runs: RunRecord[];
  selectedRunId: number | null;
  onSelectRun: (id: number) => void;
  onClearRuns: () => void;
  snapshot: PlaybackSnapshot | null;
}) {
  const result = state.status === "completed" ? state.result : null;
  const selectedLayerName = selected && result
    ? selected.layerIndex === 0 ? "__input__" : result.architecture.layers[selected.layerIndex - 1]?.name
    : null;
  return (
    <aside className="inspector" aria-label="Debugger inspector">
      <Heading icon={Bug} title="Inspector" description="Evidence from the selected run" />
      <Tabs.Root defaultValue="selection">
        <Tabs.List className="tab-list compact" aria-label="Inspector views"><Tabs.Tab value="selection">Selection</Tabs.Tab><Tabs.Tab value="diagnostics">Diagnostics</Tabs.Tab><Tabs.Tab value="runs">Runs</Tabs.Tab></Tabs.List>
        <Tabs.Panel value="selection" className="inspector-panel">
          {selected ? <><p className="selection-title">{selected.layerName} · {selected.label}</p><dl className="data-list">
            <div><dt>Neuron</dt><dd>{selected.neuronIndex === null ? "Summary" : selected.neuronIndex + 1}</dd></div>
            <div><dt>Layer width</dt><dd>{selected.width}</dd></div>
            <div><dt>Activation</dt><dd>{selected.activation ?? "None"}</dd></div>
            <div><dt>Layer parameters</dt><dd>{selected.parameterCount.toLocaleString()}</dd></div>
          </dl>{result && <LayerSignals instrumentation={snapshot?.instrumentation ? [snapshot.instrumentation] : result.training.instrumentation} selectedLayerName={selectedLayerName} />}</> : result ? <><p className="panel-copy">Select a neuron to focus its signals. All learned layers are shown below.</p><dl className="data-list">
            <div><dt>Dense layers</dt><dd>{result.architecture.layers.length}</dd></div>
            <div><dt>Total parameters</dt><dd>{result.architecture.total_parameters.toLocaleString()}</dd></div>
          </dl><LayerSignals instrumentation={snapshot?.instrumentation ? [snapshot.instrumentation] : result.training.instrumentation} /></> : <p className="panel-copy">Run an experiment to populate observed model metadata.</p>}
        </Tabs.Panel>
        <Tabs.Panel value="diagnostics" className="inspector-panel">{result ? <DiagnosticsPanel diagnostics={result.diagnostics} /> : <p className="panel-copy">Run an instrumented experiment to evaluate diagnostic rules.</p>}</Tabs.Panel>
        <Tabs.Panel value="runs" className="inspector-panel"><RunComparison runs={runs} selectedId={selectedRunId} onSelect={onSelectRun} onClear={onClearRuns} /></Tabs.Panel>
      </Tabs.Root>
    </aside>
  );
}

function Metrics({ state, snapshot, onSelectEpoch }: { state: RunState; snapshot: PlaybackSnapshot | null; onSelectEpoch: (epoch: number) => void }) {
  const result = state.status === "completed" ? state.result : null;
  const observedTraining = result && snapshot ? {
    ...result.training,
    history: result.training.history.filter((metric) => metric.epoch <= snapshot.epoch),
    final_loss: snapshot.metrics.loss,
    final_accuracy: snapshot.metrics.accuracy,
  } : result?.training;
  return (
    <section className="metrics-dock" aria-label="Metrics history">
      <Heading icon={BarChart3} title="Metrics history" description="Raw observations, newest last" />
      {result && snapshot && <PlaybackScrubber playback={result.playback} selectedEpoch={snapshot.epoch} onSelect={onSelectEpoch} />}
      {observedTraining ? <TrainingMetricsChart training={observedTraining} />
        : <p className="panel-copy">Loss and accuracy observations appear after a completed run.</p>}
    </section>
  );
}

export function App() {
  const [config, setConfig] = useState(initialConfig);
  const [state, setState] = useState<RunState>({ status: "idle" });
  const [selectedNode, setSelectedNode] = useState<ArchitectureNodeData | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedPlaybackEpoch, setSelectedPlaybackEpoch] = useState<number | null>(null);
  const [configError, setConfigError] = useState<ConfigValidationError | null>(null);
  const nextRunId = useRef(1);
  const requestSequence = useRef(0);
  const requestController = useRef<AbortController | null>(null);
  useEffect(() => () => {
    requestSequence.current += 1;
    requestController.current?.abort();
  }, []);
  function update<K extends keyof WorkbenchConfig>(key: K, value: WorkbenchConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }));
    setConfigError((current) => current?.field === key ? null : current);
    setState((current) => current.status === "error" || current.status === "cancelled" ? { status: "idle" } : current);
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const validationError = validateConfig(config);
    if (validationError) {
      setConfigError(validationError);
      setState((current) => current.status === "error" || current.status === "cancelled" ? { status: "idle" } : current);
      window.setTimeout(() => document.querySelector<HTMLElement>(`[aria-describedby="configuration-error"]`)?.focus());
      return;
    }
    setConfigError(null);
    let request: ExperimentRequest;
    try { request = buildRequest(config); }
    catch (error) { setState({ status: "error", message: error instanceof Error ? error.message : "Invalid configuration." }); return; }
    setSelectedNode(null);
    setSelectedPlaybackEpoch(null);
    setSelectedRunId(null);
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    const requestId = ++requestSequence.current;
    setState({ status: "loading" });
    try {
      const result = await createExperiment(request, controller.signal);
      if (requestId !== requestSequence.current) return;
      const id = nextRunId.current;
      nextRunId.current += 1;
      setRuns((current) => appendRun(current, { id, label: `Run ${id}`, request, result }));
      setSelectedRunId(id);
      setState({ status: "completed", result });
    }
    catch (error) {
      if (requestId !== requestSequence.current) return;
      if (error instanceof Error && error.name === "AbortError") setState({ status: "cancelled" });
      else setState({ status: "error", message: error instanceof Error ? error.message : "The training request failed." });
    } finally {
      if (requestController.current === controller) requestController.current = null;
    }
  }
  const displayResult = resolveRunResult(runs, selectedRunId, state.status === "completed" ? state.result : null);
  const displayState: RunState = displayResult ? { status: "completed", result: displayResult } : state;
  const playbackSnapshot = displayResult
    ? displayResult.playback.snapshots.find((snapshot) => snapshot.epoch === selectedPlaybackEpoch) ?? displayResult.playback.snapshots.at(-1) ?? null
    : null;
  function clearRuns() {
    setRuns([]);
    setSelectedRunId(null);
    nextRunId.current = 1;
  }
  function cancelRun() {
    requestSequence.current += 1;
    requestController.current?.abort();
    requestController.current = null;
    setState({ status: "cancelled" });
  }
  return (
    <main className="app-shell"><form onSubmit={submit} noValidate aria-busy={state.status === "loading"}>
      <header className="command-bar">
        <div className="brand-lockup"><Activity aria-hidden="true" /><span>NeuronScope</span><small>Training debugger</small></div>
        <div className="run-controls"><span className={`status-dot status-${state.status}`} /><span className="status-label">{state.status}</span>
          {state.status === "loading" ? <button className="run-button cancel-button" type="button" onClick={cancelRun}><AlertCircle />Cancel run</button>
            : <button className="run-button" type="submit"><CirclePlay />Run experiment</button>}
        </div>
      </header>
      <StateSummary state={state} />
      <div className="workbench-grid"><Configuration config={config} onChange={update} error={configError} /><Stage state={displayState} onSelect={setSelectedNode} snapshot={playbackSnapshot} /><Inspector state={displayState} selected={selectedNode} runs={runs} selectedRunId={selectedRunId} onSelectRun={(id) => { setSelectedRunId(id); setSelectedNode(null); setSelectedPlaybackEpoch(null); }} onClearRuns={clearRuns} snapshot={playbackSnapshot} /><Metrics state={displayState} snapshot={playbackSnapshot} onSelectEpoch={setSelectedPlaybackEpoch} /></div>
    </form></main>
  );
}
