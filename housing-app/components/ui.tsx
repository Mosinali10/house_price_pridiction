"use client";
import React from "react";

// ── Design tokens ──────────────────────────────────────────────────────────
export const C = {
  bg:       "#080b14",
  surface:  "#0e1420",
  card:     "#111827",
  border:   "#1f2937",
  accent:   "#a78bfa",   // violet
  accent2:  "#34d399",   // emerald
  accent3:  "#f59e0b",   // amber
  accent4:  "#f472b6",   // pink
  text:     "#f1f5f9",
  muted:    "#6b7280",
  subtle:   "#374151",
};

export const CHART_COLORS = ["#a78bfa","#34d399","#f59e0b","#f472b6","#38bdf8","#fb923c","#818cf8"];

export const TT = {
  background: "#111827",
  border: "1px solid #1f2937",
  borderRadius: 10,
  color: "#f1f5f9",
  fontSize: 12,
  boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
};

// ── Components ─────────────────────────────────────────────────────────────
export function Card({ children, className = "", glow = false }: {
  children: React.ReactNode; className?: string; glow?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl p-5 ${className}`}
      style={{
        background: "linear-gradient(145deg, #111827, #0e1420)",
        border: `1px solid ${glow ? "#a78bfa44" : "#1f2937"}`,
        boxShadow: glow ? "0 0 24px #a78bfa18" : "0 2px 12px rgba(0,0,0,0.3)",
      }}
    >
      {children}
    </div>
  );
}

export function KPI({ label, value, sub, color = "#a78bfa", icon }: {
  label: string; value: string; sub?: string; color?: string; icon?: string;
}) {
  return (
    <div
      className="rounded-2xl p-5 relative overflow-hidden"
      style={{
        background: "linear-gradient(145deg, #111827, #0e1420)",
        border: "1px solid #1f2937",
        boxShadow: "0 2px 12px rgba(0,0,0,0.3)",
      }}
    >
      {/* glow blob */}
      <div style={{
        position: "absolute", top: -20, right: -20,
        width: 80, height: 80, borderRadius: "50%",
        background: color, opacity: 0.08, filter: "blur(20px)",
      }} />
      {icon && <div className="text-xl mb-2">{icon}</div>}
      <div className="text-xs font-medium mb-1 uppercase tracking-wider" style={{ color: C.muted }}>{label}</div>
      <div className="text-2xl font-bold" style={{ color }}>{value}</div>
      {sub && <div className="text-xs mt-1.5" style={{ color: C.muted }}>{sub}</div>}
    </div>
  );
}

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-base font-semibold mb-5 flex items-center gap-2" style={{ color: C.text }}>
      <span style={{
        display: "inline-block", width: 3, height: 18, borderRadius: 99,
        background: "linear-gradient(to bottom, #a78bfa, #34d399)",
      }} />
      {children}
    </h2>
  );
}

export function InsightCard({ value, text, color }: { value: string; text: string; color: string }) {
  return (
    <div
      className="rounded-2xl p-5 relative overflow-hidden"
      style={{
        background: "linear-gradient(145deg, #111827, #0e1420)",
        border: `1px solid ${color}33`,
        boxShadow: `0 0 20px ${color}12`,
      }}
    >
      <div style={{
        position: "absolute", bottom: -16, right: -16,
        width: 72, height: 72, borderRadius: "50%",
        background: color, opacity: 0.1, filter: "blur(16px)",
      }} />
      <div className="text-2xl font-bold mb-1.5" style={{ color }}>{value}</div>
      <div className="text-sm leading-relaxed" style={{ color: "#9ca3af" }}>{text}</div>
    </div>
  );
}

export function Badge({ children, color = "#a78bfa" }: { children: React.ReactNode; color?: string }) {
  return (
    <span className="px-2 py-0.5 rounded-md text-xs font-medium"
      style={{ background: `${color}18`, color, border: `1px solid ${color}33` }}>
      {children}
    </span>
  );
}
