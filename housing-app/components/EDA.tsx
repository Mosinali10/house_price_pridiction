"use client";
import { useState } from "react";
import { ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card, SectionTitle, C, TT, CHART_COLORS } from "./ui";
import type { HousingRow } from "@/lib/data";
import { FEATURE_LABELS, FEATURE_DESCRIPTIONS, mean, pearsonCorr } from "@/lib/data";

const FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"];

function corrColor(v: number) {
  if (v > 0.6)  return "#a78bfa";
  if (v > 0.3)  return "#818cf8";
  if (v < -0.6) return "#f472b6";
  if (v < -0.3) return "#fb923c";
  return "#1f2937";
}
function corrText(v: number) {
  return Math.abs(v) > 0.3 ? "#f1f5f9" : "#6b7280";
}

export default function EDA({ data }: { data: HousingRow[] }) {
  const [selected, setSelected] = useState("RM");
  const [tab, setTab] = useState<"sample" | "stats">("sample");

  const allCols = [...FEATURES, "MEDV"];

  const corrMatrix = allCols.map(a =>
    allCols.map(b => pearsonCorr(
      data.map(d => d[a as keyof HousingRow] as number),
      data.map(d => d[b as keyof HousingRow] as number)
    ))
  );

  const scatterData = data.filter((_, i) => i % 2 === 0).map(d => ({
    x: +(d[selected as keyof HousingRow] as number).toFixed(3),
    y: d.MEDV,
  }));

  const vals = data.map(d => d[selected as keyof HousingRow] as number);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const step = (mx - mn) / 25;
  const histData = Array.from({ length: 25 }, (_, i) => {
    const lo = mn + i * step;
    return { range: lo.toFixed(1), count: vals.filter(v => v >= lo && v < lo + step).length };
  });

  const corr = pearsonCorr(vals, data.map(d => d.MEDV));
  const direction = corr > 0 ? "positively" : "negatively";
  const strength = Math.abs(corr) > 0.5 ? "strongly" : "weakly";

  return (
    <div className="space-y-10 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1" style={{ color: C.text }}>Exploratory Data Analysis</h1>
        <p className="text-sm" style={{ color: C.muted }}>Dig into individual features, correlations, and distributions.</p>
      </div>

      {/* Tabs */}
      <section>
        <div className="flex gap-2 mb-4">
          {(["sample", "stats"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className="px-4 py-2 rounded-xl text-sm font-medium transition-all"
              style={{
                background: tab === t ? "#a78bfa18" : C.card,
                color: tab === t ? C.accent : C.muted,
                border: tab === t ? "1px solid #a78bfa44" : `1px solid ${C.border}`,
              }}>
              {t === "sample" ? "📋 Raw Data Sample" : "📈 Descriptive Statistics"}
            </button>
          ))}
        </div>
        <Card>
          <div className="overflow-x-auto">
            {tab === "sample" ? (
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    {allCols.map(c => (
                      <th key={c} className="py-2.5 px-2 text-left font-semibold uppercase tracking-wider" style={{ color: C.muted }}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.slice(0, 15).map((row, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 === 0 ? "transparent" : "#0e142008" }}>
                      {allCols.map(c => (
                        <td key={c} className="py-2 px-2 font-mono" style={{ color: C.text }}>
                          {(row[c as keyof HousingRow] as number).toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    {["Feature","Count","Mean","Std","Min","25%","50%","75%","Max"].map(h => (
                      <th key={h} className="py-2.5 px-2 text-left font-semibold uppercase tracking-wider" style={{ color: C.muted }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {allCols.map((c, i) => {
                    const v = data.map(d => d[c as keyof HousingRow] as number).sort((a, b) => a - b);
                    const m = mean(v);
                    const std = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length);
                    return (
                      <tr key={c} style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 === 0 ? "transparent" : "#0e142008" }}>
                        <td className="py-2 px-2 font-mono font-bold" style={{ color: C.accent }}>{c}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v.length}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{m.toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{std.toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v[0].toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v[Math.floor(v.length * 0.25)].toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v[Math.floor(v.length * 0.5)].toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v[Math.floor(v.length * 0.75)].toFixed(3)}</td>
                        <td className="py-2 px-2 font-mono" style={{ color: C.text }}>{v[v.length - 1].toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </section>

      {/* Correlation matrix */}
      <section>
        <SectionTitle>Correlation Matrix</SectionTitle>
        <Card>
          <p className="text-xs mb-4" style={{ color: C.muted }}>
            Purple = positive · Pink = negative · Last row/col (MEDV) shows correlation with price.
          </p>
          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="p-1" />
                  {allCols.map(c => (
                    <th key={c} className="p-1 font-mono font-semibold" style={{ color: C.muted, fontSize: 9 }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allCols.map((row, ri) => (
                  <tr key={row}>
                    <td className="p-1 font-mono font-semibold pr-2" style={{ color: C.muted, fontSize: 9 }}>{row}</td>
                    {corrMatrix[ri].map((v, ci) => (
                      <td key={ci} className="p-0.5 text-center rounded"
                        style={{ background: corrColor(v), color: corrText(v), fontSize: 9, minWidth: 28 }}>
                        {v.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* Feature explorer */}
      <section>
        <SectionTitle>Feature Explorer</SectionTitle>
        <div className="mb-4">
          <select value={selected} onChange={e => setSelected(e.target.value)}
            className="px-4 py-2.5 rounded-xl text-sm"
            style={{ background: C.card, border: `1px solid ${C.border}`, color: C.text, outline: "none" }}>
            {FEATURES.map(f => <option key={f} value={f}>{f} — {FEATURE_LABELS[f]}</option>)}
          </select>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-4">
          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>{selected} vs Home Value</p>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name={FEATURE_LABELS[selected]} tick={{ fill: C.muted, fontSize: 10 }}
                  label={{ value: FEATURE_LABELS[selected], position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
                <YAxis dataKey="y" name="Value ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Scatter data={scatterData} fill={C.accent} fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>{selected} Distribution</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="range" tick={{ fill: C.muted, fontSize: 9 }} interval={4} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Bar dataKey="count" fill={CHART_COLORS[1]} radius={[3, 3, 0, 0]} name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div className="rounded-2xl p-4 text-sm" style={{ background: C.card, border: `1px solid ${C.border}` }}>
          <span style={{ color: C.accent, fontWeight: 600 }}>{selected}</span>
          {" "}is <strong style={{ color: C.text }}>{strength} {direction}</strong> correlated with price
          {" "}(r = <span style={{ color: corr > 0 ? CHART_COLORS[0] : CHART_COLORS[3] }}>{corr.toFixed(3)}</span>).
          {" "}<span style={{ color: "#9ca3af" }}>{FEATURE_DESCRIPTIONS[selected]}</span>
        </div>
      </section>

      {/* All distributions */}
      <section>
        <SectionTitle>All Feature Distributions</SectionTitle>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {FEATURES.map((feat, fi) => {
            const v = data.map(d => d[feat as keyof HousingRow] as number);
            const mn2 = Math.min(...v), mx2 = Math.max(...v), s = (mx2 - mn2) / 15;
            const h = Array.from({ length: 15 }, (_, i) => {
              const lo = mn2 + i * s;
              return { x: lo.toFixed(1), y: v.filter(x => x >= lo && x < lo + s).length };
            });
            return (
              <Card key={feat}>
                <p className="text-xs font-bold mb-2" style={{ color: CHART_COLORS[fi % CHART_COLORS.length] }}>{feat}</p>
                <p className="text-xs mb-2" style={{ color: C.muted }}>{FEATURE_LABELS[feat]}</p>
                <ResponsiveContainer width="100%" height={90}>
                  <BarChart data={h} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                    <XAxis dataKey="x" hide />
                    <YAxis hide />
                    <Tooltip contentStyle={TT} />
                    <Bar dataKey="y" fill={CHART_COLORS[fi % CHART_COLORS.length]} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
