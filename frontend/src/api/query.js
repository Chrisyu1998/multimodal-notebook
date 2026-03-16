export async function askQuestion(question) {
  const res = await fetch("/api/query/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Query failed" }));
    throw new Error(err.detail || "Query failed");
  }

  return res.json();
}
