import { Activity, PanelLeft, SunMoon } from "lucide-react";

export function App() {
  return (
    <main className="min-h-svh bg-canvas text-foreground">
      <header className="flex h-14 items-center justify-between border-b border-border bg-surface px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <Activity aria-hidden="true" className="size-4 text-accent" />
          <span className="font-mono text-sm font-semibold tracking-tight">NeuronScope</span>
          <span className="hidden text-xs text-muted sm:inline">Training debugger</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted">
          <SunMoon aria-hidden="true" className="size-4" />
          <span>System theme</span>
        </div>
      </header>
      <section className="grid min-h-[calc(100svh-3.5rem)] place-items-center p-4 sm:p-6">
        <div className="w-full max-w-2xl rounded-lg border border-border bg-surface p-6 shadow-workbench sm:p-8">
          <div className="mb-6 flex items-center gap-2 text-xs font-medium text-muted">
            <PanelLeft aria-hidden="true" className="size-4" />
            <span>Workbench</span>
          </div>
          <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
            Ready for an observed training run.
          </h1>
          <p className="mt-3 max-w-prose leading-7 text-muted">
            Dataset, model, and training controls will appear only when backed by real
            instrumentation.
          </p>
        </div>
      </section>
    </main>
  );
}
