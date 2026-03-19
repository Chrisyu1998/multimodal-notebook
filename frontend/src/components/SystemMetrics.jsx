import { LatencyChart, TokenUsageChart } from "./SystemMetricsCharts";

function StatCard({ label, value }) {
  return (
    <div className="bg-gray-50 rounded-lg px-4 py-3 text-center">
      <div className="text-lg font-semibold text-gray-900">{value ?? "—"}</div>
      <div className="text-xs text-gray-500 mt-0.5">{label}</div>
    </div>
  );
}

function fmt(n, decimals = 0) {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export default function SystemMetrics({ summary, timeseries }) {
  if (!summary) {
    return (
      <div className="text-sm text-gray-400 text-center py-8">
        No query data yet. Run some queries to see metrics.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <StatCard
          label="Total Queries"
          value={fmt(summary.total_queries)}
        />
        <StatCard
          label="Avg Latency"
          value={`${fmt(summary.avg_latency_ms, 0)} ms`}
        />
        <StatCard
          label="P95 Latency"
          value={`${fmt(summary.p95_latency_ms, 0)} ms`}
        />
        <StatCard
          label="Est. Total Cost"
          value={`$${fmt(summary.total_cost_usd, 4)}`}
        />
        <StatCard
          label="Avg Cost / Query"
          value={`$${fmt(summary.avg_cost_per_query, 5)}`}
        />
      </div>

      {/* Charts */}
      {timeseries && timeseries.length > 0 ? (
        <div className="space-y-8">
          <LatencyChart data={timeseries} />
          <TokenUsageChart data={timeseries} />
        </div>
      ) : (
        <div className="text-sm text-gray-400 text-center py-4">
          Not enough daily data to render charts.
        </div>
      )}
    </div>
  );
}
