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

## Visual principles

The workbench is calm, information-first, and dense only when there is evidence to show. It uses neutral surfaces, one purposeful blue accent, readable sans and monospace type, thin borders, and responsive spacing. Light and dark modes follow the system preference through the same semantic tokens.

## Anti-patterns we avoid

No gradient hero, metric-card wallpaper, fake activity feed, glowing glass panels, arbitrary chart data, oversized logo treatment, or animation that does not explain a state change. Controls will earn their place through a real training capability.
