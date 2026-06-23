#!/usr/bin/env bun
/** Render outputs/metrics.json into a self-contained HTML dashboard. */

import { parseMetrics, renderDashboard } from "./report";

const input = process.argv[2] ?? "outputs/metrics.json";
const output = process.argv[3] ?? "outputs/dashboard.html";

const raw = JSON.parse(await Bun.file(input).text()) as unknown;
const metrics = parseMetrics(raw);
await Bun.write(output, renderDashboard(metrics));
console.log(`[dashboard] wrote ${output} for ${metrics.run.project}`);
