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
import { memo, useMemo } from "react";
import "@xyflow/react/dist/style.css";

import type { ExperimentResponse } from "../api/experiments";

export interface ArchitectureNodeData extends Record<string, unknown> {
  label: string;
  kind: "input" | "hidden" | "output" | "overflow";
  layerName: string;
  layerIndex: number;
  neuronIndex: number | null;
  width: number;
  activation: string | null;
  parameterCount: number;
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
): ArchitectureGraph {
  const widths = [architecture.input_size, ...architecture.hidden_layers, architecture.output_size];
  const columns = widths.map((width, layerIndex) => {
    const kind = layerIndex === 0 ? "input" : layerIndex === widths.length - 1 ? "output" : "hidden";
    const metadata = kind === "input" ? undefined : architecture.layers[layerIndex - 1];
    const visible = visibleIndices(width);
    return visible.map((neuronIndex, rowIndex): ArchitectureNode => {
      const overflow = neuronIndex === null;
      const prefix = kind === "hidden" ? `hidden-${layerIndex - 1}` : kind;
      const id = overflow ? `${prefix}-overflow` : `${prefix}-${neuronIndex}`;
      const hiddenNumber = layerIndex;
      const label = overflow ? `+${width - (MAX_VISIBLE_NEURONS - 1)}` : kind === "input" ? `x${neuronIndex! + 1}` : kind === "output" ? "ŷ" : `h${hiddenNumber}.${neuronIndex! + 1}`;
      const layerName = kind === "input" ? "Input" : kind === "output" ? "Output" : `Hidden ${hiddenNumber}`;
      return {
        id,
        type: "neuron",
        position: { x: layerIndex * COLUMN_GAP, y: (rowIndex - (visible.length - 1) / 2) * ROW_GAP },
        data: {
          label, kind: overflow ? "overflow" : kind, layerName, layerIndex,
          neuronIndex, width, activation: metadata?.activation ?? null,
          parameterCount: metadata?.parameter_count ?? 0,
        },
        draggable: false,
        connectable: false,
        selectable: !overflow,
        ariaLabel: overflow
          ? `${layerName}, ${width - (MAX_VISIBLE_NEURONS - 1)} additional neurons hidden`
          : `${layerName}, neuron ${neuronIndex! + 1} of ${width}`,
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
        });
      }
    }
  }
  return { nodes: columns.flat(), edges };
}

const NeuronNode = memo(function NeuronNode({ data, selected }: NodeProps<ArchitectureNode>) {
  const isInput = data.kind === "input";
  const isOutput = data.kind === "output";
  return (
    <div className={`neuron-node neuron-${data.kind}${selected ? " is-selected" : ""}`}>
      {!isInput && <Handle type="target" position={Position.Left} isConnectable={false} />}
      <span>{data.label}</span>
      {data.neuronIndex === 0 && <small>{data.layerName}</small>}
      {!isOutput && <Handle type="source" position={Position.Right} isConnectable={false} />}
    </div>
  );
});

const nodeTypes = { neuron: NeuronNode };

export function NetworkGraph({ architecture, onSelect }: {
  architecture: ExperimentResponse["architecture"];
  onSelect: (node: ArchitectureNodeData | null) => void;
}) {
  const graph = useMemo(() => architectureToGraph(architecture), [architecture]);
  return (
    <div className="network-graph" aria-label="Neural network architecture graph">
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
