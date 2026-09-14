import { Tabs } from "@base-ui/react/tabs";
import {
  Activity, AlertCircle, BarChart3, Binary, Braces, Bug, CheckCircle2,
  ChevronDown, CirclePlay, Database, FlaskConical, Gauge, GripVertical,
  LoaderCircle, Network, SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";
import { lazy, Suspense, type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";

import {
  createExperiment,
  isBrowserDemoEnabled,
  type ActivationName, type DatasetKind, type ExperimentRequest,
  type ExperimentResponse, type InitializationName, type OptimizerName,
  type PlaybackSnapshot,
} from "./api/experiments";
import { createBrowserDemoExperiment } from "./api/browserDemo";
import { DecisionBoundary } from "./components/decisionBoundary";
import { DiagnosticsPanel } from "./components/diagnosticsPanel";
import { LayerSignals } from "./components/layerSignals";
import { NetworkGraph, type ArchitectureNodeData } from "./components/networkGraph";
import { PlaybackScrubber } from "./components/playbackScrubber";
import { appendRun, resolveRunResult, RunComparison, type RunRecord } from "./components/runComparison";
import {
  estimateParameterCount, parseHiddenLayers, validateConfig,
  type ConfigValidationError, type WorkbenchConfig,
} from "./config";
import { experimentPresets, initialConfig, type ExperimentPreset } from "./presets";

const TrainingMetricsChart = lazy(() => import("./components/trainingMetrics").then((module) => ({
  default: module.TrainingMetricsChart,
})));
const browserDemo = isBrowserDemoEnabled();

type RunState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "cancelled" }
  | { status: "error"; message: string }
  | { status: "completed"; result: ExperimentResponse };

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
    boundary: { resolution: config.boundaryResolution },
    diagnostics: {
      consecutive_epochs: config.diagnosticWindow,
      vanishing_gradient_norm: config.vanishingGradientNorm,
      exploding_gradient_norm: config.explodingGradientNorm,
      dead_relu_zero_percentage: config.deadReluPercentage,
    },
    playback: { max_snapshots: config.playbackSnapshots, trace_sample_index: config.traceSample - 1 },
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

function Configuration({ config, onChange, onPreset, error }: {
  config: WorkbenchConfig;
  onChange: <K extends keyof WorkbenchConfig>(key: K, value: WorkbenchConfig[K]) => void;
  onPreset: (preset: ExperimentPreset) => void;
  error: ConfigValidationError | null;
}) {
  const invalid = (field: ConfigValidationError["field"]) => error?.field === field;
  const describedBy = (field: ConfigValidationError["field"]) => invalid(field) ? "configuration-error" : undefined;
  return (
    <aside className="config-rail" aria-label="Experiment configuration">
      <Heading icon={SlidersHorizontal} title="Configuration" description="One synchronous observed run" />
      <section className="preset-library" aria-labelledby="preset-library-title">
        <div className="preset-library-heading">
          <div><FlaskConical aria-hidden="true" /><span id="preset-library-title">Experiment library</span></div>
          <code>{estimateParameterCount(config.hiddenLayers)?.toLocaleString() ?? "—"} params</code>
        </div>
        <div className="preset-grid">{experimentPresets.map((preset) => <button
          key={preset.id}
          type="button"
          title={preset.intent}
          onClick={() => onPreset(preset)}
        ><strong>{preset.name}</strong><span>{preset.description}</span></button>)}</div>
      </section>
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

      <details className="advanced-controls">
        <summary><span><Gauge aria-hidden="true" /> Advanced runtime</span><ChevronDown aria-hidden="true" /></summary>
        <div className="advanced-controls-body">
          <div className="control-pair">
            <Field label="Boundary grid" hint="24–80 cells per axis"><input type="number" min="24" max="80" value={config.boundaryResolution} aria-invalid={invalid("boundaryResolution")} aria-describedby={describedBy("boundaryResolution")} onChange={(e) => onChange("boundaryResolution", e.target.valueAsNumber)} /></Field>
            <Field label="Playback frames" hint="2–24 retained epochs"><input type="number" min="2" max="24" value={config.playbackSnapshots} aria-invalid={invalid("playbackSnapshots")} aria-describedby={describedBy("playbackSnapshots")} onChange={(e) => onChange("playbackSnapshots", e.target.valueAsNumber)} /></Field>
          </div>
          <Field label="Forward-pass sample" hint={`Dataset row 1–${config.samples}`}><input type="number" min="1" max={config.samples} value={config.traceSample} aria-invalid={invalid("traceSample")} aria-describedby={describedBy("traceSample")} onChange={(e) => onChange("traceSample", e.target.valueAsNumber)} /></Field>
          <Field label="Diagnostic window" hint="Consecutive observations required"><input type="number" min="2" max="20" value={config.diagnosticWindow} aria-invalid={invalid("diagnosticWindow")} aria-describedby={describedBy("diagnosticWindow")} onChange={(e) => onChange("diagnosticWindow", e.target.valueAsNumber)} /></Field>
          <div className="control-pair">
            <Field label="Vanishing norm"><input type="number" min="0.000000001" step="0.000001" value={config.vanishingGradientNorm} aria-invalid={invalid("vanishingGradientNorm")} aria-describedby={describedBy("vanishingGradientNorm")} onChange={(e) => onChange("vanishingGradientNorm", e.target.valueAsNumber)} /></Field>
            <Field label="Exploding norm"><input type="number" min="0.000001" step="1" value={config.explodingGradientNorm} aria-invalid={invalid("explodingGradientNorm")} aria-describedby={describedBy("explodingGradientNorm")} onChange={(e) => onChange("explodingGradientNorm", e.target.valueAsNumber)} /></Field>
          </div>
          <Field label="Dead ReLU threshold" hint="Percentage of zero activations"><input type="number" min="0" max="100" step="1" value={config.deadReluPercentage} aria-invalid={invalid("deadReluPercentage")} aria-describedby={describedBy("deadReluPercentage")} onChange={(e) => onChange("deadReluPercentage", e.target.valueAsNumber)} /></Field>
        </div>
      </details>
    </aside>
  );
}

function StateSummary({ state }: { state: RunState }) {
  const contents = state.status === "loading" ? [LoaderCircle, "Training in progress", "The API is running the configured experiment."] as const
    : state.status === "cancelled" ? [AlertCircle, "Run cancelled", "No result was recorded. You can adjust the configuration and run again."] as const
    : state.status === "error" ? [AlertCircle, "Run failed", state.message] as const
    : state.status === "completed" ? [CheckCircle2, "Run completed", `Accuracy ${(state.result.training.final_accuracy * 100).toFixed(1)}% · Loss ${state.result.training.final_loss.toFixed(4)}`] as const
    : [CirclePlay, "Ready to train", browserDemo ? "GitHub Pages demo mode runs a lightweight browser simulation." : "Review the configuration, then start a real run."] as const;
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

function Stage({ state, onSelect, snapshot, traceKey }: { state: RunState; onSelect: (node: ArchitectureNodeData | null) => void; snapshot: PlaybackSnapshot | null; traceKey: string }) {
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
          {result && snapshot ? <NetworkGraph key={traceKey} architecture={result.architecture} trace={snapshot.forward_pass} epoch={snapshot.epoch} onSelect={onSelect} />
            : <EmptyStage icon={Network} title="No observed architecture yet" text="Run an experiment to inspect the architecture returned by the training API." />}
        </Tabs.Panel>
        <Tabs.Panel value="boundary" className="stage-panel">
          {result && boundary ? <DecisionBoundary boundary={boundary} points={result.dataset.points} />
            : <EmptyStage icon={Braces} title="No decision boundary yet" text="Run an experiment to inspect its observed prediction grid." />}
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
            <div><dt>Observed value</dt><dd>{selected.observedValue === null || selected.observedValue === undefined ? "Unavailable" : selected.observedValue.toFixed(5)}</dd></div>
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
      {observedTraining ? <Suspense fallback={<p className="panel-copy">Loading interactive metrics…</p>}><TrainingMetricsChart training={observedTraining} /></Suspense>
        : <p className="panel-copy">Loss and accuracy observations appear after a completed run.</p>}
    </section>
  );
}

function useDesktopWorkbench(): boolean {
  const query = "(min-width: 1024px)";
  const [matches, setMatches] = useState(() => typeof window.matchMedia === "function" && window.matchMedia(query).matches);
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(query);
    const updateMatch = () => setMatches(media.matches);
    updateMatch();
    media.addEventListener("change", updateMatch);
    return () => media.removeEventListener("change", updateMatch);
  }, []);
  return matches;
}

export function App() {
  const desktopWorkbench = useDesktopWorkbench();
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
    if (state.status === "loading") {
      requestSequence.current += 1;
      requestController.current?.abort();
      requestController.current = null;
    }
    setConfig((current) => ({ ...current, [key]: value }));
    setConfigError((current) => current?.field === key ? null : current);
    setState((current) => current.status === "loading" || current.status === "error" || current.status === "cancelled" ? { status: "idle" } : current);
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
      const result = browserDemo
        ? await createBrowserDemoExperiment(request)
        : await createExperiment(request, controller.signal);
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
  function applyPreset(preset: ExperimentPreset) {
    if (state.status === "loading") {
      requestSequence.current += 1;
      requestController.current?.abort();
      requestController.current = null;
    }
    setConfig({ ...preset.config });
    setConfigError(null);
    setSelectedNode(null);
    setSelectedPlaybackEpoch(null);
    setState({ status: "idle" });
  }
  const configuration = <Configuration config={config} onChange={update} onPreset={applyPreset} error={configError} />;
  const stage = <Stage state={displayState} onSelect={setSelectedNode} snapshot={playbackSnapshot} traceKey={`${selectedRunId ?? "latest"}-${playbackSnapshot?.epoch ?? "none"}`} />;
  const inspector = <Inspector state={displayState} selected={selectedNode} runs={runs} selectedRunId={selectedRunId} onSelectRun={(id) => { setSelectedRunId(id); setSelectedNode(null); setSelectedPlaybackEpoch(null); }} onClearRuns={clearRuns} snapshot={playbackSnapshot} />;
  const metrics = <Metrics state={displayState} snapshot={playbackSnapshot} onSelectEpoch={setSelectedPlaybackEpoch} />;
  return (
    <main className="app-shell"><form onSubmit={submit} noValidate aria-busy={state.status === "loading"}>
      <header className="command-bar">
        <div className="brand-lockup"><Activity aria-hidden="true" /><span>NeuronScope</span><small>{browserDemo ? "Browser demo" : "Training debugger"}</small></div>
        <div className="run-controls"><span className={`status-dot status-${state.status}`} /><span className="status-label">{state.status}</span>
          {state.status === "loading" ? <button className="run-button cancel-button" type="button" onClick={cancelRun}><AlertCircle />Cancel run</button>
            : <button className="run-button" type="submit"><CirclePlay />Run experiment</button>}
        </div>
      </header>
      <StateSummary state={state} />
      {desktopWorkbench ? <Group className="workbench-resizable" orientation="vertical">
        <Panel defaultSize="72%" minSize="420px">
          <Group className="workbench-row" orientation="horizontal">
            <Panel defaultSize="22%" minSize="250px" maxSize="390px">{configuration}</Panel>
            <Separator className="resize-separator vertical"><GripVertical aria-hidden="true" /></Separator>
            <Panel defaultSize="53%" minSize="420px">{stage}</Panel>
            <Separator className="resize-separator vertical"><GripVertical aria-hidden="true" /></Separator>
            <Panel defaultSize="25%" minSize="280px" maxSize="440px">{inspector}</Panel>
          </Group>
        </Panel>
        <Separator className="resize-separator horizontal"><span /></Separator>
        <Panel defaultSize="28%" minSize="220px" maxSize="430px">{metrics}</Panel>
      </Group> : <div className="workbench-grid">{configuration}{stage}{inspector}{metrics}</div>}
    </form></main>
  );
}
