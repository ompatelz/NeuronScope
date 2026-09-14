import { describe, expect, it } from "vitest";

import { estimateParameterCount, validateConfig } from "./config";
import { experimentPresets } from "./presets";

describe("experiment presets", () => {
  it("uses unique identifiers and only valid bounded configurations", () => {
    expect(new Set(experimentPresets.map((preset) => preset.id)).size).toBe(experimentPresets.length);
    for (const preset of experimentPresets) {
      expect(validateConfig(preset.config), preset.name).toBeNull();
      expect(estimateParameterCount(preset.config.hiddenLayers), preset.name).toBeGreaterThan(0);
    }
  });

  it("covers both healthy and stress-oriented learning scenarios", () => {
    expect(experimentPresets.some((preset) => preset.id === "baseline")).toBe(true);
    expect(experimentPresets.filter((preset) => /stress/i.test(preset.name)).length).toBeGreaterThanOrEqual(2);
    expect(new Set(experimentPresets.map((preset) => preset.config.dataset)).size).toBeGreaterThanOrEqual(3);
  });
});
