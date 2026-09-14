export function parseHiddenLayers(value: string): number[] {
  const layers = value.split(",").map((entry) => Number(entry.trim()));
  if (
    layers.length < 1 ||
    layers.length > 8 ||
    layers.some((width) => !Number.isInteger(width) || width < 1 || width > 256)
  ) {
    throw new Error("Use 1–8 comma-separated layer widths between 1 and 256.");
  }
  return layers;
}
