import { useId } from "react";

import type { ExperimentResponse } from "../api/experiments";

interface DecisionBoundaryProps {
  boundary: ExperimentResponse["boundary"];
  points: ExperimentResponse["dataset"]["points"];
}

const WIDTH = 640;
const HEIGHT = 420;

function scale(value: number, minimum: number, maximum: number, extent: number): number {
  return ((value - minimum) / (maximum - minimum || 1)) * extent;
}

function probabilityColor(probability: number): string {
  const bounded = Math.max(0, Math.min(1, probability));
  const hue = 220 - bounded * 195;
  const lightness = 34 + Math.abs(bounded - 0.5) * 16;
  return `hsl(${hue} 58% ${lightness}%)`;
}

export function DecisionBoundary({ boundary, points }: DecisionBoundaryProps) {
  const titleId = useId();
  const descriptionId = useId();
  const { resolution, x_coordinates: xs, y_coordinates: ys, probabilities } = boundary;
  const xMin = xs[0] ?? 0;
  const xMax = xs.at(-1) ?? 1;
  const yMin = ys[0] ?? 0;
  const yMax = ys.at(-1) ?? 1;
  const cellWidth = WIDTH / resolution;
  const cellHeight = HEIGHT / resolution;

  return (
    <figure className="decision-boundary">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={`${titleId} ${descriptionId}`}
      >
        <title id={titleId}>Final model decision boundary</title>
        <desc id={descriptionId}>
          {resolution} by {resolution} prediction grid. Blue favors class zero, orange favors
          class one, and labeled samples are overlaid as points.
        </desc>
        <g aria-hidden="true">
          {probabilities.map((probability, index) => {
            const column = index % resolution;
            const row = Math.floor(index / resolution);
            return (
              <rect
                className="boundary-cell"
                key={`${row}-${column}`}
                x={column * cellWidth}
                y={HEIGHT - (row + 1) * cellHeight}
                width={cellWidth + 0.4}
                height={cellHeight + 0.4}
                fill={probabilityColor(probability)}
              />
            );
          })}
          <path
            className="boundary-midline"
            d={probabilities
              .map((probability, index) => ({ probability, index }))
              .filter(({ probability }) => Math.abs(probability - 0.5) < 0.025)
              .map(({ index }) => {
                const column = index % resolution;
                const row = Math.floor(index / resolution);
                return `M ${column * cellWidth} ${HEIGHT - (row + 1) * cellHeight} h ${cellWidth} v ${cellHeight} h -${cellWidth} Z`;
              })
              .join(" ")}
          />
          {points.map((point, index) => (
            <circle
              className={`boundary-point boundary-point-${point.label}`}
              key={`${point.x}-${point.y}-${index}`}
              cx={scale(point.x, xMin, xMax, WIDTH)}
              cy={HEIGHT - scale(point.y, yMin, yMax, HEIGHT)}
              r="3.5"
            />
          ))}
        </g>
      </svg>
      <figcaption>
        <span><i className="legend-swatch class-zero" /> Class 0</span>
        <span><i className="legend-swatch uncertain" /> 0.5 boundary</span>
        <span><i className="legend-swatch class-one" /> Class 1</span>
        <small>{boundary.probabilities.length} real model predictions · {points.length} samples</small>
      </figcaption>
    </figure>
  );
}
