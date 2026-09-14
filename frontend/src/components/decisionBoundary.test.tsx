import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DecisionBoundary } from "./decisionBoundary";

afterEach(cleanup);

describe("DecisionBoundary", () => {
  it("renders every real grid probability and overlays dataset samples", () => {
    const { container } = render(
      <DecisionBoundary
        boundary={{
          resolution: 2,
          x_coordinates: [-1, 1],
          y_coordinates: [-1, 1],
          probabilities: [0.1, 0.4, 0.6, 0.9],
        }}
        points={[
          { x: -0.5, y: 0.25, label: 0 },
          { x: 0.5, y: -0.25, label: 1 },
        ]}
      />,
    );

    expect(screen.getByRole("img", { name: /^Final model decision boundary/ })).toBeTruthy();
    expect(container.querySelectorAll(".boundary-cell")).toHaveLength(4);
    expect(container.querySelectorAll(".boundary-point")).toHaveLength(2);
    expect(screen.getByText(/4 real model predictions/)).toBeTruthy();
  });
});
