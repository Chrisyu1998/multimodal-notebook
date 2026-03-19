import { useState, useEffect } from "react";
import { fetchLatestRun, fetchRuns } from "../api/eval";
import EvalMetricsTable from "./EvalMetricsTable";
import EvalComparisonTable from "./EvalComparisonTable";

const TABS = ["Metrics", "Comparison"];

export default function EvalDashboard() {
  const [tab, setTab] = useState("Metrics");
  const [latestRun, setLatestRun] = useState(null);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchLatestRun(), fetchRuns()])
      .then(([latest, allRuns]) => {
        setLatestRun(latest);
        setRuns(allRuns);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="mt-8 p-6 bg-white rounded-xl border border-gray-200 text-center text-gray-400 text-sm">
        Loading eval results…
      </div>
    );
  }

  if (error) {
    return (
      <div className="mt-8 p-6 bg-white rounded-xl border border-red-200 text-red-500 text-sm">
        {error}
      </div>
    );
  }

  return (
    <div className="mt-8 bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 pt-5 pb-0 border-b border-gray-200">
        <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Eval Dashboard</h2>
            {latestRun && (
              <p className="text-xs text-gray-400 mt-0.5">
                Latest run: {latestRun.timestamp.slice(0, 19)} ·{" "}
                {latestRun.results.length} queries
              </p>
            )}
          </div>
          {latestRun?.summary && (
            <div className="flex gap-4 text-xs">
              <Stat label="Correctness" value={latestRun.summary.avg_correctness?.toFixed(2)} />
              <Stat label="Hallucination" value={latestRun.summary.avg_hallucination_rate?.toFixed(3)} />
              <Stat label="Faithfulness" value={latestRun.summary.avg_faithfulness?.toFixed(2)} />
            </div>
          )}
        </div>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                tab === t
                  ? "border-indigo-600 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </div>

      {/* Body */}
      <div className="p-6">
        {tab === "Metrics" && <EvalMetricsTable run={latestRun} />}
        {tab === "Comparison" && <EvalComparisonTable runs={runs} />}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="text-right">
      <div className="font-semibold text-gray-700">{value ?? "—"}</div>
      <div className="text-gray-400">{label}</div>
    </div>
  );
}
