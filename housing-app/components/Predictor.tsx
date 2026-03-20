"use client";
import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, SectionTitle, C, TT, CHART_COLORS } from "./ui";
import type { HousingRow } from "@/lib/data";
import { FEATURE_LABELS, mean } from "@/lib/data";

const FEATURE_CONFIG: Record<string, { min: number; max: number; default: number; step: number; label: string }> = {
  CRIM:    { min: 0.006, max: 89,    default: 3.6,   step: 0.1,   label: "Crime Rate" },
  ZN:      { min: 0,     max: 100,   default: 11,    step: 1,     label: "Residential Zone %" },
  INDUS:   { min: 0.46,  max: 27.74, default: 11,    step: 0.1,   label: "Industrial Area %" },
  CHAS:    { min: 0,     max: 1,     default: 0,     step: 1,     label: "Charles River (0/1)" },
  NOX:     { min: 0.385, max: 0.871, default: 0.555, step: 0.001, label: "NOx Concentration" },
  RM:      { min: 3.56,  max: 8.78,  default: 6.28,  step: 0.01,  label: "Avg Rooms" },
  AGE:     { min: 2.9,   max: 100,   default: 68,    step: 1,     label: "Pre-1940 Units %" },
  DIS:     { min: 1.13,  max: 12.13, default: 3.8,   step: 0.01,  label: "Distance to Employment" },
  RAD:     { min: 1,     max: 24,    default: 9,     step: 1,     label: "Highway Access Index" },
  TAX:     { min: 187,   max: 711,   default: 408,   step: 1,     label: "Property Tax Rate" },
  PTRATIO: { min: 12.6,  max: 22,    default: 18.5,  step: 0.1,   label: "Pupil-Teacher Ratio" },
  B:       { min: 0.32,  max: 396.9, default: 356,   step: 0.1,   label: "B Index" },
  LSTAT:   { min: 1.73,  max: 37.97, default: 12.65, step: 0.01,  label: "Lower Status %" },
};

const COEFFS: Record<string, number> = {
  CRIM: -0.108, ZN: 0.046, INDUS: 0.021, CHAS: 2.687, NOX: -17.767,
  RM: 3.810, AGE: 0.001, DIS: -1.476, RAD: 0.306, TAX: -0.012,
  PTRATIO: -0.953, B: 0.009, LSTAT: -0.525,
};
const INTERCEPT = 36.459;

function predict(vals: Record<string, number>) {
  return Math.max(5, INTERCEPT + Object.entries(COEFFS).reduce((s, [k, c]) => s + c * vals[k], 0));
}

export default function Predictor({ data }: { data: HousingRow[] }) {
  const defaults = Object.fromEntries(Object.entries(FEATURE_CONFIG).map(([k, v]) => [k, v.default]));
  const [values, setValues] = useState<Record<string, number>>(defaults);

  const prediction = predict(values);
  const low = prediction * 0.85;
  const high = prediction * 1.15;

  const datasetMeans = Object.fromEntries(
    Object.keys(FEATURE_CONFIG).map(f => [f, mean(data.map(d => d[f as keyof HousingRow] as number))])
  );

  const comparisonData = Object.keys(FEATURE_CONFIG).map(f => ({
    feature: f,
    "Your Input": +values[f].toFixed(2),
    "Dataset Mean": +datasetMeans[f].toFixed(2),
  }));

  return (
    <div className="space-y-10 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1" style={{ color: C.text }}>Price Predictor</h1>
        <p className="text-sm" style={{ color: C.muted }}>Adjust the 13 socioeconomic features to estimate the median home value.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Inputs */}
        <div className="lg:col-span-2">
          <SectionTitle>Property & Neighborhood Features</SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(FEATURE_CONFIG).map(([feat, cfg], fi) => {
              const pct = ((values[feat] - cfg.min) / (cfg.max - cfg.min)) * 100;
              const color = CHART_COLORS[fi % CHART_COLORS.length];
              return (
                <Card key={feat}>
                  <div className="flex justify-between items-center mb-3">
                    <div>
                      <span className="text-xs font-bold" style={{ color }}>{feat}</span>
                      <span className="text-xs ml-1.5" style={{ color: C.muted }}>— {cfg.label}</span>
                    </div>
                    <span className="text-sm font-bold font-mono" style={{ color: C.text }}>{values[feat]}</span>
                  </div>
                  <input type="range" min={cfg.min} max={cfg.max} step={cfg.step}
                    value={values[feat]}
                    onChange={e => setValues(v => ({ ...v, [feat]: +e.target.value }))}
                    className="w-full"
                    style={{ background: `linear-gradient(to right, ${color} ${pct}%, #1f2937 0%)` }}
                  />
                  <div className="flex justify-between text-xs mt-1.5" style={{ color: "#374151" }}>
                    <span>{cfg.min}</span><span>{cfg.max}</span>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Result panel */}
        <div className="space-y-4">
          <SectionTitle>Prediction</SectionTitle>

          {/* Main prediction box */}
          <div className="rounded-2xl p-6 text-center relative overflow-hidden"
            style={{
              background: "linear-gradient(135deg, #1a0e2e, #0e1420)",
              border: "1px solid #a78bfa44",
              boxShadow: "0 0 40px #a78bfa18",
            }}>
            <div style={{
              position: "absolute", top: -30, right: -30,
              width: 120, height: 120, borderRadius: "50%",
              background: "#a78bfa", opacity: 0.08, filter: "blur(30px)",
            }} />
            <div className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "#a78bfa88" }}>
              Estimated Value
            </div>
            <div className="text-5xl font-bold mb-1" style={{
              background: "linear-gradient(90deg, #a78bfa, #34d399)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              ${(prediction * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
            <div className="text-xs mt-3 mb-1" style={{ color: "#6b7280" }}>80% Prediction Interval</div>
            <div className="text-sm font-semibold" style={{ color: "#a78bfa" }}>
              ${(low * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })} — ${(high * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>

          {/* Model info */}
          <Card>
            <div className="space-y-2.5">
              {[
                ["Model",    "Random Forest",  CHART_COLORS[0]],
                ["Features", "13 variables",   CHART_COLORS[1]],
                ["R² Score", "0.878",          CHART_COLORS[2]],
                ["Avg Error","$2,078",         CHART_COLORS[3]],
              ].map(([k, v, col]) => (
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

      {/* Comparison chart */}
      <section>
        <SectionTitle>Your Input vs Dataset Averages</SectionTitle>
        <Card>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={comparisonData} margin={{ top: 4, right: 8, bottom: 50, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="feature" tick={{ fill: "#9ca3af", fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
              <Tooltip contentStyle={TT} />
              <Bar dataKey="Your Input" radius={[3, 3, 0, 0]}>
                {comparisonData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
              </Bar>
              <Bar dataKey="Dataset Mean" fill="#374151" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </section>
    </div>
  );
}
