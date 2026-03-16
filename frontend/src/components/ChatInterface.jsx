import { useState } from "react";
import { askQuestion } from "../api/query";

function CitationBadge({ nums, sources }) {
  const [open, setOpen] = useState(false);
  const cited = nums.map((n) => sources[n - 1]).filter(Boolean);

  return (
    <span className="relative inline-block">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs font-semibold text-blue-600 hover:text-blue-800
          bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded
          px-1 mx-0.5 cursor-pointer transition-colors leading-none"
      >
        [{nums.join(", ")}]
      </button>
      {open && (
        <div
          className="absolute z-10 bottom-full left-0 mb-1 bg-white border
            border-gray-200 rounded-lg shadow-lg p-2 space-y-1 min-w-max"
        >
          {cited.length === 0 ? (
            <span className="text-xs text-gray-400">Source not found</span>
          ) : (
            cited.map((src, i) => (
              <div key={i} className="text-xs text-gray-700 whitespace-nowrap">
                <span className="font-semibold text-blue-600">[{nums[i]}]</span>{" "}
                {src.filename}
                {src.page != null ? ` — Page ${src.page}` : ""}
              </div>
            ))
          )}
        </div>
      )}
    </span>
  );
}

function renderAnswer(answer, sources) {
  const parts = [];
  let last = 0;
  const re = /\[(\d+(?:,\s*\d+)*)\]/g;
  let match;
  while ((match = re.exec(answer)) !== null) {
    if (match.index > last) parts.push(answer.slice(last, match.index));
    const nums = match[1].split(",").map((s) => parseInt(s.trim(), 10));
    parts.push(<CitationBadge key={match.index} nums={nums} sources={sources} />);
    last = match.index + match[0].length;
  }
  if (last < answer.length) parts.push(answer.slice(last));
  return parts;
}

export default function ChatInterface({ enabled }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim() || !enabled) return;
    setError("");
    setLoading(true);
    try {
      const data = await askQuestion(question);
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-xl mx-auto mt-8">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={!enabled || loading}
          placeholder={enabled ? "Ask a question about your file..." : "Upload a file first"}
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500
            disabled:bg-gray-100 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={!enabled || loading || !question.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium
            hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Submit
        </button>
      </form>

      {loading && (
        <div className="mt-4 flex items-center justify-center gap-2 text-gray-500 text-sm">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          Generating answer...
        </div>
      )}

      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

      {answer && !loading && (
        <div className="mt-4 border border-gray-200 rounded-lg p-4 bg-white">
          <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
            {renderAnswer(answer, sources)}
          </p>

          {sources.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {sources.map((src, i) => (
                <span
                  key={i}
                  className="text-xs bg-gray-100 text-gray-600 rounded-full px-3 py-1 border border-gray-200"
                >
                  <span className="font-semibold text-blue-600">[{i + 1}]</span>{" "}
                  {src.filename}
                  {src.page != null ? ` — Page ${src.page}` : ""}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
