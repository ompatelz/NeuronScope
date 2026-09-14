import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PlaybackScrubber } from "./playbackScrubber";
import { nearestRecordedEpoch } from "./playbackUtils";

const forwardPass = { sample_index: 0, input_values: [0, 0], expected_label: 0 as const, layers: [], output_logit: 0, predicted_probability: 0.5, predicted_label: 1 as const };

afterEach(cleanup);

describe("PlaybackScrubber", () => {
  it("chooses the nearest recorded epoch deterministically", () => {
    expect(nearestRecordedEpoch([1, 10, 20], 7)).toBe(10);
    expect(nearestRecordedEpoch([1, 10, 20], 5)).toBe(1);
  });

  it("labels and selects a recorded epoch from the keyboard-accessible range", () => {
    const onSelect = vi.fn();
    render(<PlaybackScrubber playback={{ resolution: 24, x_coordinates: [], y_coordinates: [], snapshots: [
      { epoch: 1, metrics: { epoch: 1, loss: 0.7, accuracy: 0.5 }, instrumentation: null, probabilities: [], forward_pass: forwardPass },
      { epoch: 10, metrics: { epoch: 10, loss: 0.2, accuracy: 0.9 }, instrumentation: null, probabilities: [], forward_pass: forwardPass },
    ] }} selectedEpoch={1} onSelect={onSelect} />);
    expect(screen.getByText("Recorded epoch 1")).toBeTruthy();
    fireEvent.change(screen.getByRole("slider", { name: "Training playback epoch" }), { target: { value: "9" } });
    expect(onSelect).toHaveBeenCalledWith(10);
  });
});
