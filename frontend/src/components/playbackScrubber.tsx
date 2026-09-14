import type { ExperimentResponse } from "../api/experiments";
import { nearestRecordedEpoch } from "./playbackUtils";

export function PlaybackScrubber({ playback, selectedEpoch, onSelect }: {
  playback: ExperimentResponse["playback"];
  selectedEpoch: number;
  onSelect: (epoch: number) => void;
}) {
  const epochs = playback.snapshots.map((snapshot) => snapshot.epoch);
  if (!epochs.length) return null;
  return <div className="playback-control">
    <div><strong>Training playback</strong><span>Recorded epoch {selectedEpoch}</span></div>
    <input
      aria-label="Training playback epoch"
      type="range"
      min={epochs[0]}
      max={epochs.at(-1)}
      step="1"
      value={selectedEpoch}
      onChange={(event) => onSelect(nearestRecordedEpoch(epochs, event.target.valueAsNumber))}
    />
    <small>{epochs.length} bounded snapshots · nearest recorded epoch selected</small>
  </div>;
}
