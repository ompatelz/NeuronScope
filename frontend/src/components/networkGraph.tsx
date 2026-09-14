import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { Pause, Play, RotateCcw, StepBack, StepForward, Zap } from "lucide-react";
import { memo, useEffect, useMemo, useState, type CSSProperties } from "react";
import "@xyflow/react/dist/style.css";

import type { ExperimentResponse, ForwardPassTrace } from "../api/experiments";

type FlowState = "queued" | "active" | "settled";
type SignalSign = "positive" | "negative" | "zero" | "unavailable";

export interface ArchitectureNodeData extends Record<string, unknown> {
  label: string;
  kind: "input" | "hidden" | "output" | "overflow";
  layerName: string;
  layerIndex: number;
  neuronIndex: number | null;
  width: number;
  activation: string | null;
  parameterCount: number;
  observedValue?: number | null;
  signalLevel: number;
  signalSign: SignalSign;
  flowState: FlowState;
  isAggregate: boolean;
}

export type ArchitectureNode = Node<ArchitectureNodeData, "neuron">;
export interface ArchitectureGraph { nodes: ArchitectureNode[]; edges: Edge[] }

const MAX_VISIBLE_NEURONS = 24;
const COLUMN_GAP = 180;
const ROW_GAP = 46;

function visibleIndices(width: number): Array<number | null> {
  if (width <= MAX_VISIBLE_NEURONS) return Array.from({ length: width }, (_, index) => index);
  return [...Array.from({ length: MAX_VISIBLE_NEURONS - 1 }, (_, index) => index), null];
}

// Pure graph transformation is exported for deterministic unit testing.
// eslint-disable-next-line react-refresh/only-export-components
export function architectureToGraph(
  architecture: ExperimentResponse["architecture"],
  trace?: ForwardPassTrace,
  activePhase = 0,
): ArchitectureGraph {
  const widths = [architecture.input_size, ...architecture.hidden_layers, architecture.output_size];
  const columns = widths.map((width, layerIndex) => {
    const kind = layerIndex === 0 ? "input" : layerIndex === widths.length - 1 ? "output" : "hidden";
    const metadata = kind === "input" ? undefined : architecture.layers[layerIndex - 1];
    const tracedValues = layerIndex === 0
      ? trace?.input_values
      : trace?.layers[layerIndex - 1]?.activations;
    const finiteValues = (tracedValues ?? []).filter((value): value is number => value !== null && Number.isFinite(value));
    const maxMagnitude = Math.max(0, ...finiteValues.map((value) => Math.abs(value)));
    const visible = visibleIndices(width);
    return visible.map((neuronIndex, rowIndex): ArchitectureNode => {
      const overflow = neuronIndex === null;
      const prefix = kind === "hidden" ? `hidden-${layerIndex - 1}` : kind;
      const id = overflow ? `${prefix}-overflow` : `${prefix}-${neuronIndex}`;
      const hiddenNumber = layerIndex;
      const label = overflow ? `+${width - (MAX_VISIBLE_NEURONS - 1)}` : kind === "input" ? `x${neuronIndex! + 1}` : kind === "output" ? "ŷ" : `h${hiddenNumber}.${neuronIndex! + 1}`;
      const layerName = kind === "input" ? "Input" : kind === "output" ? "Output" : `Hidden ${hiddenNumber}`;
      const omittedValues = overflow ? (tracedValues ?? []).slice(MAX_VISIBLE_NEURONS - 1).filter((value): value is number => value !== null && Number.isFinite(value)) : [];
      const observedValue = overflow
        ? omittedValues.length ? omittedValues.reduce((sum, value) => sum + Math.abs(value), 0) / omittedValues.length : null
        : neuronIndex === null ? null : tracedValues?.[neuronIndex];
      const signalLevel = observedValue === null || observedValue === undefined || maxMagnitude === 0
        ? 0
        : Math.min(1, Math.abs(observedValue) / maxMagnitude);
      const signalSign: SignalSign = observedValue === null || observedValue === undefined
        ? "unavailable" : observedValue === 0 ? "zero" : observedValue > 0 ? "positive" : "negative";
      const flowState: FlowState = layerIndex === activePhase ? "active" : layerIndex < activePhase ? "settled" : "queued";
      const evidence = observedValue === null || observedValue === undefined
        ? ""
        : overflow ? `, mean absolute activation ${observedValue.toFixed(4)}` : `, observed activation ${observedValue.toFixed(4)}`;
      return {
        id,
        type: "neuron",
        position: { x: layerIndex * COLUMN_GAP, y: (rowIndex - (visible.length - 1) / 2) * ROW_GAP },
        data: {
          label, kind: overflow ? "overflow" : kind, layerName, layerIndex,
          neuronIndex, width, activation: metadata?.activation ?? null,
          parameterCount: metadata?.parameter_count ?? 0,
          observedValue, signalLevel, signalSign, flowState, isAggregate: overflow,
        },
        draggable: false,
        connectable: false,
        selectable: !overflow,
        ariaLabel: overflow
          ? `${layerName}, ${width - (MAX_VISIBLE_NEURONS - 1)} additional neurons hidden${evidence}`
          : `${layerName}, neuron ${neuronIndex! + 1} of ${width}${evidence}`,
      };
    });
  });

  const edges: Edge[] = [];
  for (let index = 0; index < columns.length - 1; index += 1) {
    for (const source of columns[index]) {
      for (const target of columns[index + 1]) {
        edges.push({
          id: `edge-${source.id}-${target.id}`,
          source: source.id,
          target: target.id,
          selectable: false,
          focusable: false,
          type: "straight",
          animated: activePhase === index + 1,
          className: `signal-edge flow-${activePhase === index + 1 ? "active" : activePhase > index + 1 ? "settled" : "queued"}`,
          data: { flowState: activePhase === index + 1 ? "active" : activePhase > index + 1 ? "settled" : "queued" },
        });
      }
    }
  }
  return { nodes: columns.flat(), edges };
}

const NeuronNode = memo(function NeuronNode({ data, selected }: NodeProps<ArchitectureNode>) {
  const isInput = data.kind === "input";
  const isOutput = data.kind === "output";
  const style = {
    "--signal-level": data.signalLevel,
    "--signal-mix": `${Math.round(18 + data.signalLevel * 52)}%`,
    "--signal-glow": `${Math.round(6 + data.signalLevel * 15)}px`,
  } as CSSProperties;
  return (
    <div
      className={`neuron-node neuron-${data.kind}${selected ? " is-selected" : ""}`}
      data-flow-state={data.flowState}
      data-activation-sign={data.signalSign}
      data-activation-intensity={data.signalLevel.toFixed(3)}
      style={style}
      title={data.observedValue === null || data.observedValue === undefined ? undefined : `${data.isAggregate ? "Mean absolute activation" : "Observed activation"}: ${data.observedValue.toFixed(5)}`}
    >
      {!isInput && <Handle type="target" position={Position.Left} isConnectable={false} />}
      <span>{data.label}</span>
      {data.neuronIndex === 0 && <small>{data.layerName}</small>}
      {!isOutput && <Handle type="source" position={Position.Right} isConnectable={false} />}
    </div>
  );
});

const nodeTypes = { neuron: NeuronNode };

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function forwardPhaseLabel(trace: ForwardPassTrace, phase: number): string {
  if (phase === 0) return "Input layer";
  const layer = trace.layers[phase - 1];
  if (!layer) return "Output result";
  return layer.layer_name === "output" ? "Output prediction" : `Hidden ${phase} · ${layer.activation_name ?? "linear"}`;
}

function signalSummary(trace: ForwardPassTrace, phase: number): string {
  const values = (phase === 0 ? trace.input_values : trace.layers[phase - 1]?.activations ?? [])
    .filter((value): value is number => value !== null && Number.isFinite(value));
  if (!values.length) return "No finite activation values";
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return `range ${Math.min(...values).toFixed(3)} to ${Math.max(...values).toFixed(3)} · mean ${mean.toFixed(3)}`;
}

function ForwardPassControls({ trace, epoch, phase, playing, speed, onPhase, onPlaying, onSpeed }: {
  trace: ForwardPassTrace; epoch: number; phase: number; playing: boolean; speed: number;
  onPhase: (phase: number) => void; onPlaying: (playing: boolean) => void; onSpeed: (speed: number) => void;
}) {
  const lastPhase = trace.layers.length;
  const complete = phase === lastPhase;
  return <section className="forward-pass-player" aria-label="Forward pass animation">
    <div className="forward-pass-heading"><span><Zap aria-hidden="true" /> Observed forward pass</span><code>epoch {epoch}</code></div>
    <div className="forward-pass-meta"><strong>Sample #{trace.sample_index + 1}</strong><span>true class {trace.expected_label}</span></div>
    <div className="forward-pass-status" role="status" aria-live="polite">
      <strong>{forwardPhaseLabel(trace, phase)}</strong><span>step {phase + 1} of {lastPhase + 1}</span><small>{signalSummary(trace, phase)}</small>
    </div>
    <div className="forward-pass-controls">
      <button type="button" aria-label="Restart forward pass" onClick={() => { onPhase(0); onPlaying(!prefersReducedMotion()); }}><RotateCcw /></button>
      <button type="button" aria-label="Previous layer" disabled={phase === 0} onClick={() => { onPlaying(false); onPhase(Math.max(0, phase - 1)); }}><StepBack /></button>
      <button type="button" className="forward-primary" aria-label={playing ? "Pause forward pass" : complete ? "Restart and play forward pass" : "Play forward pass"} onClick={() => { if (complete) onPhase(0); onPlaying(!playing); }}>{playing ? <Pause /> : <Play />}</button>
      <button type="button" aria-label="Next layer" disabled={complete} onClick={() => { onPlaying(false); onPhase(Math.min(lastPhase, phase + 1)); }}><StepForward /></button>
      <label>Speed<select aria-label="Forward pass speed" value={speed} onChange={(event) => onSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label>
    </div>
    <div className={`prediction-chip${complete ? " is-settled" : ""}`}>p(class 1) <strong>{(trace.predicted_probability * 100).toFixed(1)}%</strong><span>predicted {trace.predicted_label}</span></div>
    <p className="forward-pass-note">Neuron values are observed. Edge pulses show execution order, not weight strength.</p>
  </section>;
}

export function NetworkGraph({ architecture, trace, epoch, onSelect }: {
  architecture: ExperimentResponse["architecture"];
  trace: ForwardPassTrace;
  epoch: number;
  onSelect: (node: ArchitectureNodeData | null) => void;
}) {
  const lastPhase = trace.layers.length;
  const reducedMotion = prefersReducedMotion();
  const [phase, setPhase] = useState(reducedMotion ? lastPhase : 0);
  const [playing, setPlaying] = useState(!reducedMotion);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    if (!playing) return;
    if (phase >= lastPhase) return;
    const nextPhase = Math.min(lastPhase, phase + 1);
    const timer = window.setTimeout(() => {
      setPhase(nextPhase);
      if (nextPhase === lastPhase) setPlaying(false);
    }, 720 / speed);
    return () => window.clearTimeout(timer);
  }, [lastPhase, phase, playing, speed]);
  const graph = useMemo(() => architectureToGraph(architecture, trace, phase), [architecture, trace, phase]);
  return (
    <div className="network-graph" aria-label="Neural network architecture graph">
      <ForwardPassControls trace={trace} epoch={epoch} phase={phase} playing={playing} speed={speed} onPhase={setPhase} onPlaying={setPlaying} onSpeed={setSpeed} />
      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        nodeTypes={nodeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesFocusable={false}
        onNodeClick={(_, node) => onSelect(node.data)}
        onPaneClick={() => onSelect(null)}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        minZoom={0.25}
        maxZoom={1.75}
        ariaLabelConfig={{ "controls.ariaLabel": "Graph view controls" }}
      >
        <Background gap={18} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
