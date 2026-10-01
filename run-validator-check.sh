#!/bin/sh
# A1 gate: validate addon.json against the REAL ResonantOS validator.
# Usage: sh run-validator-check.sh <path-to-2.0.0-alpha-clone>
# Creates a temporary spec inside their repo (vitest include limits), runs it,
# removes it, and leaves their tree clean.
set -e
REPO="${1:?usage: run-validator-check.sh <path-to-2.0.0-alpha-clone>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="$REPO/src/sdk/addons/zz-external-stackbench.test.ts"
cat > "$SPEC" <<EOS
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { validateAddOnManifest } from "./validation";
const manifest = JSON.parse(readFileSync("$HERE/addon.json", "utf8"));
describe("external addon.stack-bench vs real validator", () => {
  it("validates with zero errors and zero warnings (sideloaded)", () => {
    const result = validateAddOnManifest(manifest, { source: "sideload" });
    console.log(JSON.stringify(result.issues ?? []));
    expect((result.issues ?? []).filter(i => i.severity === "error")).toEqual([]);
    expect((result.issues ?? []).filter(i => i.severity === "warning")).toEqual([]);
    expect(result.valid).toBe(true);
  });
});
EOS
trap 'rm -f "$SPEC"' EXIT
cd "$REPO" && npx vitest run src/sdk/addons/zz-external-stackbench.test.ts
