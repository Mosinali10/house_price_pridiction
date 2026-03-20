"use client";
import { useState, useEffect } from "react";
import type { HousingRow, Metrics } from "@/lib/data";
import Overview from "@/components/Overview";
import EDA from "@/components/EDA";
import ModelPerformance from "@/components/ModelPerformance";
import Predictor from "@/components/Predictor";

const NAV = [
  { id: "overview", icon: "📊", label: "Overview & Insights" },
  { id: "eda",      icon: "🔍", label: "Exploratory Analysis" },
  { id: "model",    icon: "🤖", label: "Model Performance" },
  { id: "predict",  icon: "🔮", label: "Price Predictor" },
];

export default function Home() {
  const [page, setPage] = useState("overview");
  const [data, setData] = useState<HousingRow[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("/housing.json").then(r => r.json()),
      fetch("/metrics.json").then(r => r.json()),
    ]).then(([d, m]) => { setData(d); setMetrics(m); })
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen" style={{ background: "#080b14" }}>
        <div className="text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <p className="text-sm" style={{ color: "#f43f5e" }}>Failed to load data. Please refresh.</p>
        </div>
      </div>
    );
  }

  if (!data.length || !metrics) {
    return (
      <div className="flex items-center justify-center min-h-screen" style={{ background: "#080b14" }}>
        <div className="text-center">
          <div className="w-10 h-10 rounded-full mx-auto mb-4 animate-spin"
            style={{ border: "3px solid #1f2937", borderTopColor: "#a78bfa" }} />
          <p className="text-sm" style={{ color: "#6b7280" }}>Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#080b14" }}>

      {/* Sidebar */}
      <aside
        className="flex flex-col transition-all duration-300 shrink-0 overflow-y-auto"
        style={{
          width: sidebarOpen ? 256 : 60,
          background: "#080b14",
          borderRight: "1px solid #1f2937",
          height: "100vh",
        }}
      >
        {/* Logo */}
        <div className="flex items-center justify-between px-4 py-5"
          style={{ borderBottom: "1px solid #1f2937" }}>
          {sidebarOpen && (
            <div>
              <div className="font-bold text-base tracking-tight" style={{ color: "#f1f5f9" }}>
                🏠 Boston Housing
              </div>
              <div className="text-xs mt-0.5 font-medium" style={{
                background: "linear-gradient(90deg, #a78bfa, #34d399)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>
                Analytics Platform
              </div>
            </div>
          )}
          <button onClick={() => setSidebarOpen(!sidebarOpen)}
            className="rounded-lg p-1.5 transition-colors"
            style={{ background: "#111827", color: "#6b7280", border: "1px solid #1f2937" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {sidebarOpen
                ? <path d="M15 18l-6-6 6-6" />
                : <path d="M9 18l6-6-6-6" />}
            </svg>
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(n => {
            const active = page === n.id;
            return (
              <button key={n.id} onClick={() => setPage(n.id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-all duration-150"
                style={{
                  background: active ? "#a78bfa18" : "transparent",
                  color: active ? "#a78bfa" : "#6b7280",
                  border: active ? "1px solid #a78bfa33" : "1px solid transparent",
                  fontWeight: active ? 600 : 400,
                }}>
                <span className="text-base shrink-0">{n.icon}</span>
                {sidebarOpen && <span className="text-sm">{n.label}</span>}
              </button>
            );
          })}
        </nav>

        {/* Footer meta */}
        {sidebarOpen && (
          <div className="p-4 mx-3 mb-4 rounded-xl space-y-2"
            style={{ background: "#111827", border: "1px solid #1f2937" }}>
            {[
              ["Dataset", "Boston Housing"],
              ["Records", "506 properties"],
              ["Model", "Random Forest"],
              ["Features", "13 variables"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs">
                <span style={{ color: "#4b5563" }}>{k}</span>
                <span style={{ color: "#9ca3af" }}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto p-6 lg:p-8 page-enter">
        {page === "overview" && <Overview data={data} metrics={metrics} />}
        {page === "eda"      && <EDA data={data} />}
        {page === "model"    && <ModelPerformance metrics={metrics} />}
        {page === "predict"  && <Predictor data={data} />}
      </main>
    </div>
  );
}
