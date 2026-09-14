import { AlertTriangle, CheckCircle2, CircleX } from "lucide-react";

import type { DiagnosticResult } from "../api/experiments";

function title(value: DiagnosticResult["type"]): string {
  return value.split("_").map((word) => word[0]?.toUpperCase() + word.slice(1)).join(" ");
}

export function DiagnosticsPanel({ diagnostics }: { diagnostics: DiagnosticResult[] }) {
  if (!diagnostics.length) return <div className="diagnostics-clear" role="status"><CheckCircle2 aria-hidden="true" /><div><strong>No diagnostic rules triggered</strong><p>Captured statistics stayed outside the configured thresholds.</p></div></div>;
  return <div className="diagnostics-list" aria-label="Training diagnostics">{diagnostics.map((item, index) => {
    const Icon = item.severity === "critical" ? CircleX : AlertTriangle;
    return <article className={`diagnostic diagnostic-${item.severity}`} key={`${item.type}-${index}`}>
      <header><Icon aria-hidden="true" /><strong>{title(item.type)}</strong><span>{item.severity}</span></header>
      <p>{item.explanation}</p>
      {item.evidence.map((evidence) => <dl key={`${evidence.layer_name}-${evidence.metric}`}><dt>{evidence.layer_name} · {evidence.metric}</dt><dd>{evidence.observed_values.join(", ")}<small> threshold {evidence.threshold}</small></dd></dl>)}
      <ul>{item.possible_actions.map((action) => <li key={action}>{action}</li>)}</ul>
    </article>;
  })}</div>;
}
