import { useState, useRef } from "react";
import { uploadFile, pollStatus } from "../api/upload";

const STATUS = {
  IDLE: "idle",
  UPLOADING: "uploading",
  INDEXING: "indexing",
  READY: "ready",
  ERROR: "error",
};

export default function FileUpload({ onReady }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(STATUS.IDLE);
  const [errorMsg, setErrorMsg] = useState("");
  const [dragging, setDragging] = useState(false);
  const [numChunks, setNumChunks] = useState(0);
  const inputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      setStatus(STATUS.IDLE);
      setErrorMsg("");
    }
  }

  function handleDragOver(e) {
    e.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleBrowse(e) {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setStatus(STATUS.IDLE);
      setErrorMsg("");
    }
  }

  async function handleUpload() {
    if (!file) return;
    setErrorMsg("");
    setStatus(STATUS.UPLOADING);
    try {
      const uploadResponse = await uploadFile(file);
      setStatus(STATUS.INDEXING);
      const finalResponse = await pollStatus(uploadResponse.file_id, uploadResponse);
      const chunks = finalResponse.num_chunks ?? 0;
      setNumChunks(chunks);
      setStatus(STATUS.READY);
      onReady?.(chunks);
    } catch (err) {
      setStatus(STATUS.ERROR);
      setErrorMsg(err.message);
    }
  }

  const isProcessing = status === STATUS.UPLOADING || status === STATUS.INDEXING;

  const statusLabel =
    status === STATUS.UPLOADING
      ? "Uploading..."
      : status === STATUS.INDEXING
      ? "Indexing..."
      : status === STATUS.READY
      ? numChunks > 0
        ? `Ready — ${numChunks} chunks indexed`
        : "Ready to query!"
      : null;

  return (
    <div className="w-full max-w-xl mx-auto">
      <div
        onClick={() => !isProcessing && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"}
          ${isProcessing ? "cursor-not-allowed opacity-60" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          onChange={handleBrowse}
          disabled={isProcessing}
        />
        {file ? (
          <div>
            <p className="font-medium text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-500 mt-1">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div>
            <p className="text-gray-500">Drag and drop a file here, or click to browse</p>
            <p className="text-xs text-gray-400 mt-1">PDF, image, video, or audio</p>
          </div>
        )}
      </div>

      {statusLabel && (
        <p
          className={`mt-3 text-sm font-medium text-center
            ${status === STATUS.READY ? "text-green-600" : "text-blue-600"}
          `}
        >
          {statusLabel}
        </p>
      )}

      {status === STATUS.ERROR && (
        <p className="mt-3 text-sm text-red-600 text-center">{errorMsg}</p>
      )}

      {file && status !== STATUS.READY && (
        <button
          onClick={handleUpload}
          disabled={isProcessing}
          className="mt-4 w-full py-2 px-4 bg-blue-600 text-white rounded-lg font-medium
            hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isProcessing ? statusLabel : "Upload"}
        </button>
      )}
    </div>
  );
}
