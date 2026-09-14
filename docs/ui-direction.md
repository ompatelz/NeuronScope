# UI direction

## Selected foundation

- **React + Vite + TypeScript** for a compact, client-first instrument panel.
- **Tailwind CSS v4** for local design tokens and responsive composition.
- **Base UI** for accessible, unstyled interaction primitives; **Lucide** for a restrained icon set.
- **shadcn/ui** is a source-owned registry we may selectively adopt when a concrete control is needed; its components are copied into this repository rather than hidden behind a design-system dependency.

## Deferred deliberately

- **React Flow / React Flow UI:** excellent for an eventual computation graph, but no graph exists in Task 0.
- **ECharts, Observable Plot, or Visx:** choose against actual diagnostic data and interaction requirements, not placeholder charts.
- **Motion:** use only when an interaction communicates state; no decorative entrance animation.
- **Radix:** mature alternative, but Base UI gives an accessible unstyled foundation with the current React stack. Avoid mixing primitive systems without a reason.

## Workbench information architecture

The application uses a seam-based instrument layout rather than a dashboard of detached cards:

- a 280 px configuration rail owns dataset, network, and training controls;
- the flexible center stage switches between Network and Boundary views;
- a 320 px inspector switches between Selection and Diagnostics evidence;
- a bottom dock owns raw metrics history.

Below desktop width, panels reflow into two columns and then one continuous mobile column. Base UI
Tabs provide keyboard navigation and focus behavior. Native labeled inputs preserve browser validation.
`POST /api/v1/experiments` is the only source of run summaries; idle, loading, error, and completed
states remain explicit, and unavailable visualizations say so instead of displaying sample telemetry.

## Network graph

The Network view uses React Flow only after a completed experiment supplies an architecture. A pure,
deterministic transform maps input, hidden, and output neurons into fixed left-to-right columns with
stable identifiers and adjacent-layer edges. Nodes can be selected but not moved or connected; the
inspector reports the selected neuron's layer, index, width, activation, and layer parameter count.
Very wide layers retain their real count while collapsing excess visual nodes, avoiding an unbounded
DOM and edge explosion. Graph layout remains presentation logic and never changes model execution.

## Visual principles

The workbench is calm, information-first, and dense only when there is evidence to show. It uses neutral surfaces, one purposeful blue accent, readable sans and monospace type, thin borders, and responsive spacing. Light and dark modes follow the system preference through the same semantic tokens.

## Anti-patterns we avoid

No gradient hero, metric-card wallpaper, fake activity feed, glowing glass panels, arbitrary chart data, oversized logo treatment, or animation that does not explain a state change. Controls will earn their place through a real training capability.
