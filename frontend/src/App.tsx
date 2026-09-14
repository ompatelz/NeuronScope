import { Tabs } from "@base-ui/react/tabs";
import {
  Activity, AlertCircle, BarChart3, Binary, Braces, Bug, CheckCircle2,
  CirclePlay, Database, LoaderCircle, Network, SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";
import { type FormEvent, type ReactNode, useState } from "react";

import {
  createExperiment,
  type ActivationName, type DatasetKind, type ExperimentRequest,
  type ExperimentResponse, type InitializationName, type OptimizerName,
} from "./api/experiments";
import { parseHiddenLayers } from "./config";

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

function Configuration({ config, onChange }: {
  config: WorkbenchConfig;
  onChange: <K extends keyof WorkbenchConfig>(key: K, value: WorkbenchConfig[K]) => void;
}) {
  return (
    <aside className="config-rail" aria-label="Experiment configuration">
      <Heading icon={SlidersHorizontal} title="Configuration" description="One synchronous observed run" />
      <fieldset className="control-group">
        <legend><Database /> Dataset</legend>
        <Field label="Kind"><select value={config.dataset} onChange={(e) => onChange("dataset", e.target.value as DatasetKind)}>
          <option value="two_moons">Two moons</option><option value="circles">Circles</option>
          <option value="xor">XOR</option><option value="spiral">Spiral</option>
        </select></Field>
        <div className="control-pair">
          <Field label="Samples"><input type="number" min="40" max="2000" value={config.samples} onChange={(e) => onChange("samples", e.target.valueAsNumber)} /></Field>
          <Field label="Noise"><input type="number" min="0" max="0.5" step="0.01" value={config.noise} onChange={(e) => onChange("noise", e.target.valueAsNumber)} /></Field>
        </div>
        <Field label="Seed" hint="Shared by dataset and model"><input type="number" min="0" max="2147483647" value={config.seed} onChange={(e) => onChange("seed", e.target.valueAsNumber)} /></Field>
      </fieldset>

      <fieldset className="control-group">
        <legend><Network /> Network</legend>
        <Field label="Hidden layers" hint="Comma-separated widths; maximum 8">
          <input aria-label="Hidden layers" type="text" inputMode="numeric" value={config.hiddenLayers} onChange={(e) => onChange("hiddenLayers", e.target.value)} placeholder="8, 8" />
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
          <Field label="Learning rate"><input type="number" min="0.0001" max="1" step="0.0001" value={config.learningRate} onChange={(e) => onChange("learningRate", e.target.valueAsNumber)} /></Field>
          <Field label="Epochs"><input type="number" min="1" max="5000" value={config.epochs} onChange={(e) => onChange("epochs", e.target.valueAsNumber)} /></Field>
        </div>
      </fieldset>
    </aside>
  );
}

function StateSummary({ state }: { state: RunState }) {
  const contents = state.status === "loading" ? [LoaderCircle, "Training in progress", "The API is running the configured experiment."] as const
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

function Stage({ state }: { state: RunState }) {
  const result = state.status === "completed" ? state.result : null;
  return (
    <section className="stage" aria-label="Visualization stage">
      <Tabs.Root defaultValue="network" className="tab-root">
        <Tabs.List className="tab-list" aria-label="Visualizations">
          <Tabs.Tab value="network"><Network /> Network</Tabs.Tab>
          <Tabs.Tab value="boundary"><Braces /> Boundary</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="network" className="stage-panel">
          {result ? <div className="observed-summary"><small>Observed architecture</small><strong>{result.architecture.input_size} → {result.architecture.hidden_layers.join(" → ")} → {result.architecture.output_size}</strong><span>{result.architecture.total_parameters.toLocaleString()} trainable parameters</span><p>Interactive graph rendering arrives in Task 7. This summary comes from the API.</p></div>
            : <EmptyStage icon={Network} title="No observed architecture yet" text="Run an experiment to inspect the architecture returned by the training API." />}
        </Tabs.Panel>
        <Tabs.Panel value="boundary" className="stage-panel">
          {result ? <div className="observed-summary"><small>Observed dataset</small><strong>{result.dataset.points.length.toLocaleString()} classified points</strong><span>No decision grid exists in this response.</span><p>Task 8 will render only boundaries computed from real model predictions.</p></div>
            : <EmptyStage icon={Braces} title="No decision boundary yet" text="A prediction grid will appear here when that real capability is implemented." />}
        </Tabs.Panel>
      </Tabs.Root>
    </section>
  );
}

function Inspector({ state }: { state: RunState }) {
  const result = state.status === "completed" ? state.result : null;
  return (
    <aside className="inspector" aria-label="Debugger inspector">
      <Heading icon={Bug} title="Inspector" description="Evidence from the selected run" />
      <Tabs.Root defaultValue="selection">
        <Tabs.List className="tab-list compact" aria-label="Inspector views"><Tabs.Tab value="selection">Selection</Tabs.Tab><Tabs.Tab value="diagnostics">Diagnostics</Tabs.Tab></Tabs.List>
        <Tabs.Panel value="selection" className="inspector-panel">
          {result ? <dl className="data-list">
            <div><dt>Layers</dt><dd>{result.architecture.layers.length}</dd></div>
            <div><dt>Activation</dt><dd>{result.architecture.layers[0]?.activation ?? "None"}</dd></div>
            <div><dt>Optimizer</dt><dd>{result.training.config.optimizer}</dd></div>
            <div><dt>Epochs</dt><dd>{result.training.history.length}</dd></div>
          </dl> : <p className="panel-copy">Run an experiment to populate observed model metadata.</p>}
        </Tabs.Panel>
        <Tabs.Panel value="diagnostics" className="inspector-panel"><p className="panel-copy">Diagnostics stay empty until Task 11 supplies transparent rules and evidence.</p></Tabs.Panel>
      </Tabs.Root>
    </aside>
  );
}

function Metrics({ state }: { state: RunState }) {
  const recent = state.status === "completed" ? state.result.training.history.slice(-5) : [];
  return (
    <section className="metrics-dock" aria-label="Metrics history">
      <Heading icon={BarChart3} title="Metrics history" description="Raw observations, newest last" />
      {recent.length ? <div className="metrics-table-wrap"><table><caption className="sr-only">Most recent training metrics</caption><thead><tr><th>Epoch</th><th>Loss</th><th>Accuracy</th></tr></thead><tbody>
        {recent.map((metric) => <tr key={metric.epoch}><td>{metric.epoch}</td><td>{metric.loss.toFixed(5)}</td><td>{(metric.accuracy * 100).toFixed(1)}%</td></tr>)}
      </tbody></table></div> : <p className="panel-copy">Loss and accuracy observations appear after a completed run.</p>}
    </section>
  );
}

export function App() {
  const [config, setConfig] = useState(initialConfig);
  const [state, setState] = useState<RunState>({ status: "idle" });
  function update<K extends keyof WorkbenchConfig>(key: K, value: WorkbenchConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }));
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    let request: ExperimentRequest;
    try { request = buildRequest(config); }
    catch (error) { setState({ status: "error", message: error instanceof Error ? error.message : "Invalid configuration." }); return; }
    setState({ status: "loading" });
    try { setState({ status: "completed", result: await createExperiment(request) }); }
    catch (error) { setState({ status: "error", message: error instanceof Error ? error.message : "The training request failed." }); }
  }
  return (
    <main className="app-shell"><form onSubmit={submit}>
      <header className="command-bar">
        <div className="brand-lockup"><Activity aria-hidden="true" /><span>NeuronScope</span><small>Training debugger</small></div>
        <div className="run-controls"><span className={`status-dot status-${state.status}`} /><span className="status-label">{state.status}</span>
          <button className="run-button" type="submit" disabled={state.status === "loading"}>{state.status === "loading" ? <LoaderCircle className="animate-spin" /> : <CirclePlay />}{state.status === "loading" ? "Training…" : "Run experiment"}</button>
        </div>
      </header>
      <StateSummary state={state} />
      <div className="workbench-grid"><Configuration config={config} onChange={update} /><Stage state={state} /><Inspector state={state} /><Metrics state={state} /></div>
    </form></main>
  );
}
