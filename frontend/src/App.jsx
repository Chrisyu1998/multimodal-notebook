import { useState } from "react";
import FileUpload from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";
import EvalDashboard from "./components/EvalDashboard";

const TABS = ["Notebook", "Eval Dashboard"];

export default function App() {
  const [activeTab, setActiveTab] = useState("Notebook");
  const [ready, setReady] = useState(false);
  const [chatKey, setChatKey] = useState(0);

  function handleReady() {
    setReady(true);
    setChatKey((k) => k + 1);
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 flex items-center gap-6 h-14">
          <span className="text-base font-bold text-gray-900">Multimodal Notebook</span>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  activeTab === t
                    ? "bg-indigo-50 text-indigo-700"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }`}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-4 py-10">
        {activeTab === "Notebook" && (
          <div className="max-w-xl mx-auto">
            <FileUpload onReady={handleReady} />
            <ChatInterface key={chatKey} enabled={ready} />
          </div>
        )}
        {activeTab === "Eval Dashboard" && <EvalDashboard />}
      </main>
    </div>
  );
}
