import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DiagnosticsPanel } from "./diagnosticsPanel";

afterEach(cleanup);

describe("DiagnosticsPanel", () => {
  it("shows a no-issues state", () => {
    render(<DiagnosticsPanel diagnostics={[]} />);
    expect(screen.getByText("No diagnostic rules triggered")).toBeTruthy();
  });

  it("shows evidence and actions", () => {
    render(<DiagnosticsPanel diagnostics={[{ type: "vanishing_gradients", severity: "warning", evidence: [{ layer_name: "hidden_0", epochs: [1, 2, 3], metric: "gradient_norm", observed_values: [0.001, 0.001, 0.001], threshold: 0.01 }], explanation: "Tiny gradients.", possible_actions: ["Try ReLU."] }]} />);
    expect(screen.getByText("Vanishing Gradients")).toBeTruthy();
    expect(screen.getByText(/hidden_0 · gradient_norm/)).toBeTruthy();
    expect(screen.getByText("Try ReLU.")).toBeTruthy();
  });
});
