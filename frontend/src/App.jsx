import { useState } from "react";
import FileUpload from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";

export default function App() {
  const [ready, setReady] = useState(false);
  const [chatKey, setChatKey] = useState(0);

  function handleReady() {
    setReady(true);
    setChatKey((k) => k + 1); // reset chat history on each new upload
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900 text-center mb-8">
          Multimodal Notebook
        </h1>
        <FileUpload onReady={handleReady} />
        <ChatInterface key={chatKey} enabled={ready} />
      </div>
    </div>
  );
}
