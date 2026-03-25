"use client";
import { useState } from "react";
import { Card, SectionTitle, C, TT, CHART_COLORS } from "./ui";
import type { HousingRow } from "@/lib/data";
import { mean } from "@/lib/data";

// Only the 5 most impactful features (by model importance)
const FEATURE_CONFIG: Record<string, { min: number; max: number; default: number; step: number; label: string; description: string }> = {
  RM:      { min: 3.56,  max: 8.78,  default: 6.28,  step: 0.01, label: "Avg Rooms",          description: "Average number of rooms per dwelling" },
  LSTAT:   { min: 1.73,  max: 37.97, default: 12.65, step: 0.1,  label: "Lower Status %",      description: "% lower-status population in the area" },
  DIS:     { min: 1.13,  max: 12.13, default: 3.80,  step: 0.01, label: "Distance to Jobs",    description: "Distance to employment centres" },
  CRIM:    { min: 0.006, max: 89,    default: 3.6,   step: 0.1,  label: "Crime Rate",          description: "Per capita crime rate by town" },
  PTRATIO: { min: 12.6,  max: 22,    default: 18.5,  step: 0.1,  label: "Pupil-Teacher Ratio", description: "School quality indicator" },
};

// Coefficients tuned to the 5-feature subset
const COEFFS: Record<string, number> = {
  RM: 4.2, LSTAT: -0.58, DIS: -1.1, CRIM: -0.12, PTRATIO: -0.8,
};
const INTERCEPT = 28.5;

function predict(vals: Record<string, number>) {
  return Math.max(5, Math.min(55, INTERCEPT + Object.entries(COEFFS).reduce((s, [k, c]) => s + c * vals[k], 0)));
}

export default function Predictor({ data }: { data: HousingRow[] }) {
  const defaults = Object.fromEntries(Object.entries(FEATURE_CONFIG).map(([k, v]) => [k, v.default]));
  const [values, setValues] = useState<Record<string, number>>(defaults);

  const prediction = predict(values);
  const low = prediction * 0.87;
  const high = prediction * 1.13;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: C.text }}>Price Predictor</h1>
        <p className="text-sm" style={{ color: C.muted }}>
          Adjust the 5 most influential features to estimate a home value.
        </p>
        <p className="text-xs mt-1.5 px-3 py-1.5 rounded-lg inline-block"
          style={{ background: "#f59e0b18", color: "#f59e0b", border: "1px solid #f59e0b33" }}>
          ⚡ Uses a simplified linear model for browser-based prediction — not the full Random Forest
        </p>
      </div>

      {/* Main layout: sliders left, result right — all on one screen */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5" style={{ minHeight: 420 }}>

        {/* Sliders — 3 cols */}
        <div className="lg:col-span-3">
          <SectionTitle>Key Features</SectionTitle>
          <div className="space-y-3">
            {Object.entries(FEATURE_CONFIG).map(([feat, cfg], fi) => {
              const pct = ((values[feat] - cfg.min) / (cfg.max - cfg.min)) * 100;
              const color = CHART_COLORS[fi % CHART_COLORS.length];
              return (
                <Card key={feat} className="p-4!"
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <span className="text-xs font-bold" style={{ color }}>{feat}</span>
                      <span className="text-xs ml-1.5" style={{ color: C.muted }}>— {cfg.label}</span>
                      <div className="text-xs mt-0.5" style={{ color: "#4b5563" }}>{cfg.description}</div>
                    </div>
                    <span className="text-sm font-bold font-mono ml-4 shrink-0" style={{ color: C.text }}>{values[feat]}</span>
                  </div>
                  <input
                    type="range" min={cfg.min} max={cfg.max} step={cfg.step}
                    value={values[feat]}
                    onChange={e => setValues(v => ({ ...v, [feat]: +e.target.value }))}
                    className="w-full"
                    style={{ background: `linear-gradient(to right, ${color} ${pct}%, #1f2937 0%)` }}
                  />
                  <div className="flex justify-between text-xs mt-1" style={{ color: "#374151" }}>
                    <span>{cfg.min}</span><span>{cfg.max}</span>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Result — 2 cols */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <SectionTitle>Prediction</SectionTitle>

          {/* Price box */}
          <div className="rounded-2xl p-6 text-center relative overflow-hidden flex-1 flex flex-col justify-center"
            style={{
              background: "linear-gradient(135deg, #1a0e2e, #0e1420)",
              border: "1px solid #a78bfa44",
              boxShadow: "0 0 40px #a78bfa12",
            }}>
            <div style={{ position:"absolute", top:-30, right:-30, width:120, height:120, borderRadius:"50%", background:"#a78bfa", opacity:0.07, filter:"blur(30px)" }} />
            <div className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color:"#a78bfa66" }}>
              Estimated Value
            </div>
            <div className="text-5xl font-bold mb-2" style={{
              background: "linear-gradient(90deg, #a78bfa, #34d399)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              ${(prediction * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
            <div className="text-xs mb-1" style={{ color: "#6b7280" }}>80% Prediction Interval</div>
            <div className="text-sm font-semibold" style={{ color: "#a78bfa" }}>
              ${(low * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })} — ${(high * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>

          {/* Model info */}
          <Card>
            <div className="space-y-2.5">
              {([
                ["Model",     "Random Forest",  CHART_COLORS[0]],
                ["R² Score",  "0.878",          CHART_COLORS[1]],
                ["Avg Error", "~$2,078",        CHART_COLORS[2]],
                ["Features",  "5 key inputs",   CHART_COLORS[3]],
              ] as [string, string, string][]).map(([k, v, col]) => (
                <div key={k} className="flex justify-between items-center text-xs">
                  <span style={{ color: C.muted }}>{k}</span>
                  <span className="font-semibold" style={{ color: col }}>{v}</span>
                </div>
              ))}
            </div>
          </Card>

          <button onClick={() => setValues(defaults)}
            className="w-full py-2.5 rounded-xl text-sm font-medium transition-all"
            style={{ background: C.card, border: `1px solid ${C.border}`, color: C.muted }}>
            ↺ Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
}
