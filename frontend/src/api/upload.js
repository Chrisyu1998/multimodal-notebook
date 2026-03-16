const TERMINAL_STATUSES = new Set(["indexed", "already_indexed", "error"]);

/**
 * POST /api/upload — returns the full upload response including num_chunks.
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/upload/", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

/**
 * Poll GET /api/upload/status/{fileId} every 2 seconds until status is
 * "indexed" or "error".
 *
 * If `initialResponse` already carries a terminal status (i.e., the backend
 * completed indexing synchronously during the POST), this resolves immediately
 * without making any additional requests.
 */
export async function pollStatus(fileId, initialResponse = null) {
  if (initialResponse && TERMINAL_STATUSES.has(initialResponse.status)) {
    if (initialResponse.status === "error") {
      throw new Error(initialResponse.message || "Indexing failed");
    }
    return initialResponse;
  }

  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/upload/status/${fileId}`);
        if (!res.ok) {
          clearInterval(interval);
          reject(new Error("Status check failed"));
          return;
        }
        const data = await res.json();
        if (data.status === "indexed" || data.status === "already_indexed") {
          clearInterval(interval);
          resolve(data);
        } else if (data.status === "error") {
          clearInterval(interval);
          reject(new Error(data.message || "Indexing failed"));
        }
      } catch (err) {
        clearInterval(interval);
        reject(err);
      }
    }, 2000);
  });
}
