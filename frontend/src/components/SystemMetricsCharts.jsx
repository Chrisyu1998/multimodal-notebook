import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

export function LatencyChart({ data }) {
  const last30 = data.slice(-30);
  return (
    <div>
      <h3 className="text-sm font-medium text-gray-700 mb-3">
        Latency Trend — last 30 days
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={last30} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11 }}
            tickFormatter={(d) => d.slice(5)}
          />
          <YAxis tick={{ fontSize: 11 }} unit="ms" width={56} />
          <Tooltip
            formatter={(v) => `${v.toFixed(0)} ms`}
            labelFormatter={(l) => `Date: ${l}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="avg_latency_ms"
            name="Avg latency"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="p95_latency_ms"
            name="P95 latency"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            strokeDasharray="4 2"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TokenUsageChart({ data }) {
  const last14 = data.slice(-14);
  return (
    <div>
      <h3 className="text-sm font-medium text-gray-700 mb-3">
        Token Usage — last 14 days
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={last14} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11 }}
            tickFormatter={(d) => d.slice(5)}
          />
          <YAxis tick={{ fontSize: 11 }} width={56} />
          <Tooltip
            formatter={(v) => v.toLocaleString()}
            labelFormatter={(l) => `Date: ${l}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="avg_input_tokens" name="Avg input tokens" stackId="a" fill="#6366f1" />
          <Bar dataKey="avg_output_tokens" name="Avg output tokens" stackId="a" fill="#a5b4fc" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
