import type { EpochInstrumentation, LayerInstrumentation } from "../api/experiments";

function label(name: string): string {
  const hidden = /^hidden_(\d+)$/.exec(name);
  if (hidden) return `Hidden ${Number(hidden[1]) + 1}`;
  return name.charAt(0).toUpperCase() + name.slice(1).replaceAll("_", " ");
}

// Formatting is exported for focused unit tests.
// eslint-disable-next-line react-refresh/only-export-components
export function formatSignal(value: number | null): string {
  if (value === null) return "Unavailable";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 10_000) return value.toExponential(3);
  return value.toFixed(4);
}

function Signal({ name, value }: { name: string; value: number | null }) {
  return <div><dt>{name}</dt><dd className={value === null ? "signal-missing" : ""}>{formatSignal(value)}</dd></div>;
}

function LayerRow({ layer, maxGradient }: { layer: LayerInstrumentation; maxGradient: number }) {
  const gradientWidth = layer.gradients.norm === null || maxGradient === 0 ? 0 : Math.max(2, (layer.gradients.norm / maxGradient) * 100);
  const nonfinite = layer.gradients.nonfinite_count + (layer.activation?.nonfinite_count ?? 0);
  return (
    <article className={`layer-signal-row${nonfinite > 0 ? " has-nonfinite" : ""}`} aria-label={`${label(layer.layer_name)} signals`}>
      <header>
        <strong>{label(layer.layer_name)}</strong>
        {nonfinite > 0 && <span>{nonfinite} non-finite</span>}
      </header>
      <div className="signal-bar" title={`Relative gradient norm ${formatSignal(layer.gradients.norm)}`}><i style={{ width: `${gradientWidth}%` }} /></div>
      <dl className="signal-grid">
        <Signal name="Gradient norm" value={layer.gradients.norm} />
        <Signal name="Gradient mean" value={layer.gradients.mean} />
        <Signal name="Gradient std" value={layer.gradients.std} />
        <Signal name="Weight norm" value={layer.weight_norm} />
      </dl>
      {layer.activation ? <>
        <div className="activation-heading"><span>Activation</span><span>{layer.activation.zero_percentage.toFixed(1)}% zero</span></div>
        <div className="zero-bar" aria-label={`${layer.activation.zero_percentage.toFixed(1)}% zero activations`}><i style={{ width: `${layer.activation.zero_percentage}%` }} /></div>
        <dl className="signal-grid activation-grid">
          <Signal name="Mean" value={layer.activation.mean} /><Signal name="Std" value={layer.activation.std} />
          <Signal name="Minimum" value={layer.activation.minimum} /><Signal name="Maximum" value={layer.activation.maximum} />
        </dl>
      </> : <p className="signal-unavailable">Activation statistics unavailable for this layer.</p>}
    </article>
  );
}

export function LayerSignals({ instrumentation, selectedLayerName }: {
  instrumentation: EpochInstrumentation[];
  selectedLayerName?: string | null;
}) {
  const latest = instrumentation.at(-1);
  if (!latest) return <p className="signal-unavailable">Instrumentation was not collected for this run.</p>;
  const layers = selectedLayerName ? latest.layers.filter((layer) => layer.layer_name === selectedLayerName) : latest.layers;
  if (!layers.length) return <p className="signal-unavailable">The selected input has no learned-layer instrumentation.</p>;
  const maxGradient = Math.max(0, ...layers.map((layer) => layer.gradients.norm ?? 0));
  return (
    <section className="layer-signals" aria-labelledby="layer-signals-title">
      <div className="signals-title"><h3 id="layer-signals-title">Layer signals</h3><span>Epoch {latest.epoch}</span></div>
      {layers.map((layer) => <LayerRow key={layer.layer_name} layer={layer} maxGradient={maxGradient} />)}
    </section>
  );
}
