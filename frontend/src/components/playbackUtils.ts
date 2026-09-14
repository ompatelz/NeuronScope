export function nearestRecordedEpoch(epochs: number[], requested: number): number {
  return epochs.reduce((nearest, epoch) =>
    Math.abs(epoch - requested) < Math.abs(nearest - requested) ? epoch : nearest
  );
}
