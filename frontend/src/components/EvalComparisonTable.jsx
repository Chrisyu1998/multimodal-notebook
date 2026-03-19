import { useState, useEffect } from "react";
import { fetchRun } from "../api/eval";

function deltaClass(delta) {
  if (delta == null) return "text-gray-400";
  if (delta > 0) return "text-green-700 font-semibold";
  if (delta < 0) return "text-red-700 font-semibold";
  return "text-gray-500";
}

function fmtDelta(v) {
  if (v == null) return "—";
  return (v > 0 ? "+" : "") + v.toFixed(2);
}

function avgDelta(pairs, key) {
  const vals = pairs.map((p) => p[key]).filter((v) => v != null);
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export default function EvalComparisonTable({ runs }) {
  const [runAId, setRunAId] = useState("");
  const [runBId, setRunBId] = useState("");
  const [runA, setRunA] = useState(null);
  const [runB, setRunB] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (runs.length >= 2 && !runAId && !runBId) {
      setRunBId(runs[0].run_id);
      setRunAId(runs[1].run_id);
    }
  }, [runs]);

  useEffect(() => {
    if (!runAId) return;
    fetchRun(runAId).then(setRunA).catch((e) => setError(e.message));
  }, [runAId]);

  useEffect(() => {
    if (!runBId) return;
    fetchRun(runBId).then(setRunB).catch((e) => setError(e.message));
  }, [runBId]);

  const pairs = (() => {
    if (!runA || !runB) return [];
    const mapA = Object.fromEntries(runA.results.map((r) => [r.query_id, r]));
    return runB.results
      .filter((r) => mapA[r.query_id])
      .map((rb) => {
        const ra = mapA[rb.query_id];
        const dCorrect = rb.scores && ra.scores
          ? rb.scores.correctness - ra.scores.correctness
          : null;
        const dHalluc = rb.scores && ra.scores
          ? rb.scores.hallucination_rate - ra.scores.hallucination_rate
          : null;
        return { rb, dCorrect, dHalluc };
      });
  })();

  const aggCorrect = avgDelta(pairs.map((p) => ({ key: p.dCorrect })), "key");
  const aggHalluc = avgDelta(pairs.map((p) => ({ key: p.dHalluc })), "key");

  const tsLabel = (id) => runs.find((r) => r.run_id === id)?.timestamp?.slice(0, 19) ?? id;

  return (
    <div>
      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Run A (baseline)</label>
          <select value={runAId} onChange={(e) => setRunAId(e.target.value)}
            className="text-sm border border-gray-200 rounded px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            <option value="">— select —</option>
            {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.timestamp.slice(0, 19)}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Run B (new)</label>
          <select value={runBId} onChange={(e) => setRunBId(e.target.value)}
            className="text-sm border border-gray-200 rounded px-2 py-1 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            <option value="">— select —</option>
            {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.timestamp.slice(0, 19)}</option>)}
          </select>
        </div>
      </div>

      {error && <p className="text-red-500 text-sm mb-3">{error}</p>}

      {pairs.length > 0 && (
        <>
          <div className="flex gap-6 mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm">
            <span className="text-gray-500 font-medium">Aggregate deltas (B − A):</span>
            <span className={deltaClass(aggCorrect)}>
              Correctness: {fmtDelta(aggCorrect)}
            </span>
            <span className={deltaClass(aggHalluc)}>
              Hallucination: {fmtDelta(aggHalluc)}
            </span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="min-w-full text-sm divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Query</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">Answer (B)</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide whitespace-nowrap">Correctness Δ</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide whitespace-nowrap">Hallucination Δ</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 bg-white">
                {pairs.map(({ rb, dCorrect, dHalluc }) => {
                  const highlight =
                    (dCorrect != null && Math.abs(dCorrect) > 0.5) ||
                    (dHalluc != null && Math.abs(dHalluc) > 0.5);
                  return (
                    <tr key={rb.query_id} className={highlight ? "bg-yellow-50" : "hover:bg-gray-50"}>
                      <td className="px-3 py-2 text-gray-800 max-w-xs truncate" title={rb.query}>{rb.query}</td>
                      <td className="px-3 py-2 text-gray-600 max-w-sm text-xs" style={{ maxWidth: "280px" }}>
                        <span className="line-clamp-2">{rb.generated_answer}</span>
                      </td>
                      <td className={`px-3 py-2 text-xs ${deltaClass(dCorrect)}`}>{fmtDelta(dCorrect)}</td>
                      <td className={`px-3 py-2 text-xs ${deltaClass(dHalluc)}`}>{fmtDelta(dHalluc)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">Rows highlighted in yellow changed by &gt;0.5 in either metric.</p>
        </>
      )}

      {!runAId || !runBId ? (
        <p className="text-sm text-gray-400 mt-4">Select two runs above to compare.</p>
      ) : pairs.length === 0 && runA && runB ? (
        <p className="text-sm text-gray-400 mt-4">No overlapping queries between the two runs.</p>
      ) : null}
    </div>
  );
}
