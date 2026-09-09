import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

import { App } from "./App";

describe("App", () => {
  it("exposes the workbench purpose", () => {
    expect(renderToStaticMarkup(<App />)).toContain("Training debugger");
  });
});
