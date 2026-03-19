import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const CATEGORIES = ["factual", "multi-hop", "cross-modal", "adversarial", "out-of-scope"];

export default function EvalScoreChart({ results }) {
  const data = CATEGORIES.map((cat) => {
    const rows = results.filter((r) => r.category === cat && r.scores);
    if (!rows.length) return null;
    const avg = (key) =>
      Math.round((rows.reduce((s, r) => s + r.scores[key], 0) / rows.length) * 10) / 10;
    return {
      name: cat.replace("-", "\u2011"), // non-breaking hyphen for display
      Correctness: avg("correctness"),
      Faithfulness: avg("faithfulness"),
    };
  }).filter(Boolean);

  if (!data.length) return null;

  return (
    <div className="mt-6 mb-2">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
        Avg scores by category
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 0, right: 16, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} />
          <YAxis domain={[0, 5]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="Correctness" fill="#4ade80" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Faithfulness" fill="#60a5fa" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
