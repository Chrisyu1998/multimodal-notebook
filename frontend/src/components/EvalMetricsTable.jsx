import { useState } from "react";
import EvalScoreChart from "./EvalScoreChart";

const CATEGORIES = ["all", "factual", "multi-hop", "cross-modal", "adversarial", "out-of-scope"];

function scoreColor(value, metric) {
  if (value == null) return "";
  if (metric === "correctness" || metric === "faithfulness") {
    if (value >= 4) return "bg-green-100 text-green-800";
    if (value >= 2) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  }
  if (metric === "hallucination_rate") {
    if (value <= 0.1) return "bg-green-100 text-green-800";
    if (value <= 0.3) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  }
  if (metric === "context_precision") {
    if (value >= 0.8) return "bg-green-100 text-green-800";
    if (value >= 0.5) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  }
  return "";
}

function Cell({ value, metric, fmt }) {
  if (value == null) return <td className="px-3 py-2 text-gray-400 text-xs">—</td>;
  return (
    <td className="px-3 py-2">
      <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${scoreColor(value, metric)}`}>
        {fmt(value)}
      </span>
    </td>
  );
}

function avgOf(rows, key, excludeCategory = null) {
  const filtered = excludeCategory ? rows.filter((r) => r.category !== excludeCategory) : rows;
  const vals = filtered.map((r) => r.scores?.[key]).filter((v) => v != null);
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export default function EvalMetricsTable({ run }) {
  const [category, setCategory] = useState("all");

  const allResults = run?.results ?? [];
  const filtered = category === "all" ? allResults : allResults.filter((r) => r.category === category);
  const okRows = filtered.filter((r) => r.status === "ok");

  return (
    <div>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <p className="text-sm text-gray-500">
          {filtered.length} queries
          {run?.summary && (
            <span className="ml-2 text-gray-400">
              · p50 {run.summary.p50_latency_ms}ms · p95 {run.summary.p95_latency_ms}ms
            </span>
          )}
        </p>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="text-sm border border-gray-200 rounded px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <EvalScoreChart results={filtered} />

      <div className="overflow-x-auto rounded-lg border border-gray-200 mt-4">
        <table className="min-w-full text-sm divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {["Query", "Category", "Correctness", "Hallucination", "Faithfulness", "Ctx Precision", "Latency (ms)"].map((h) => (
                <th key={h} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {filtered.map((r) => (
              <tr key={r.query_id} className="hover:bg-gray-50">
                <td className="px-3 py-2 text-gray-800 max-w-xs truncate" title={r.query}>
                  {r.query}
                </td>
                <td className="px-3 py-2 text-gray-500 whitespace-nowrap text-xs">{r.category}</td>
                <Cell value={r.scores?.correctness} metric="correctness" fmt={(v) => v.toFixed(1)} />
                <Cell value={r.scores?.hallucination_rate} metric="hallucination_rate" fmt={(v) => v.toFixed(2)} />
                <Cell value={r.scores?.faithfulness} metric="faithfulness" fmt={(v) => v.toFixed(1)} />
                <Cell value={r.scores?.context_precision} metric="context_precision" fmt={(v) => v.toFixed(2)} />
                <td className="px-3 py-2 text-gray-500 text-xs whitespace-nowrap">{r.latency_ms.toFixed(0)}</td>
              </tr>
            ))}
            {/* Average row */}
            <tr className="bg-gray-50 font-semibold border-t-2 border-gray-300">
              <td className="px-3 py-2 text-gray-700 text-xs" colSpan={2}>Avg ({okRows.length} ok)</td>
              <Cell value={avgOf(okRows, "correctness")} metric="correctness" fmt={(v) => v.toFixed(2)} />
              <Cell value={avgOf(okRows, "hallucination_rate")} metric="hallucination_rate" fmt={(v) => v.toFixed(3)} />
              <Cell value={avgOf(okRows, "faithfulness")} metric="faithfulness" fmt={(v) => v.toFixed(2)} />
              <Cell value={avgOf(okRows, "context_precision", "out-of-scope")} metric="context_precision" fmt={(v) => v.toFixed(3)} />
              <td className="px-3 py-2 text-gray-500 text-xs">
                {okRows.length ? (okRows.reduce((s, r) => s + r.latency_ms, 0) / okRows.length).toFixed(0) : "—"} ms
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
