"use client";
import {
  ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { Card, KPI, SectionTitle, C, TT, CHART_COLORS } from "./ui";
import type { Metrics } from "@/lib/data";
import { FEATURE_LABELS } from "@/lib/data";

export default function ModelPerformance({ metrics }: { metrics: Metrics }) {
  const { y_test, y_pred } = metrics;
  const residuals = y_test.map((v, i) => v - y_pred[i]);

  const scatterData = y_test.map((v, i) => ({ actual: +v.toFixed(2), predicted: +y_pred[i].toFixed(2) }));

  const rMin = Math.min(...residuals), rMax = Math.max(...residuals);
  const step = (rMax - rMin) / 25;
  const residHist = Array.from({ length: 25 }, (_, i) => {
    const lo = rMin + i * step;
    return { range: lo.toFixed(1), count: residuals.filter(r => r >= lo && r < lo + step).length };
  });

  const fiData = Object.entries(metrics.feature_importances)
    .map(([k, v]) => ({ feature: FEATURE_LABELS[k] || k, importance: +(v * 100).toFixed(2) }))
    .sort((a, b) => b.importance - a.importance);

  const residScatter = y_pred.map((p, i) => ({ predicted: +p.toFixed(2), residual: +residuals[i].toFixed(2) }));

  const errorTable = y_test.map((v, i) => ({
    actual: v.toFixed(2),
    predicted: y_pred[i].toFixed(2),
    error: (v - y_pred[i]).toFixed(2),
    absError: Math.abs(v - y_pred[i]).toFixed(2),
    pct: (Math.abs(v - y_pred[i]) / v * 100).toFixed(1),
  })).sort((a, b) => +b.absError - +a.absError);

  return (
    <div className="space-y-10 max-w-7xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1" style={{ color: C.text }}>Model Performance</h1>
        <p className="text-sm" style={{ color: C.muted }}>Detailed evaluation of the Random Forest Regressor trained on 13 features.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPI label="R² Score"       value={metrics.r2.toFixed(4)}      sub="Variance explained"    color={CHART_COLORS[0]} icon="🎯" />
        <KPI label="RMSE ($k)"      value={metrics.rmse.toFixed(4)}     sub="Root mean sq error"    color={CHART_COLORS[3]} icon="📐" />
        <KPI label="MAE ($k)"       value={metrics.mae.toFixed(4)}      sub="Mean absolute error"   color={CHART_COLORS[1]} icon="📏" />
        <KPI label="CV R² (5-fold)" value={`${metrics.cv_r2_mean.toFixed(3)} ± ${metrics.cv_r2_std.toFixed(3)}`} sub="Cross-validation" color={CHART_COLORS[2]} icon="🔁" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Card>
          <SectionTitle>Actual vs Predicted</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="actual" name="Actual ($k)" tick={{ fill: C.muted, fontSize: 10 }}
                label={{ value: "Actual ($k)", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
              <YAxis dataKey="predicted" name="Predicted ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
              <Tooltip contentStyle={TT} />
              <Scatter data={scatterData} fill={CHART_COLORS[0]} fillOpacity={0.6} name="Prediction" />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Residuals Distribution</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={residHist} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="range" tick={{ fill: C.muted, fontSize: 9 }} interval={4} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine x="0.0" stroke="#f43f5e" strokeDasharray="4 2" />
              <Bar dataKey="count" fill={CHART_COLORS[1]} radius={[3, 3, 0, 0]} name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Feature Importances</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={fiData} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 110 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" tick={{ fill: C.muted, fontSize: 10 }} unit="%" />
              <YAxis type="category" dataKey="feature" tick={{ fill: "#9ca3af", fontSize: 10 }} width={105} />
              <Tooltip contentStyle={TT} formatter={(v) => [`${v}%`, "Importance"]} />
              <Bar dataKey="importance" radius={[0, 5, 5, 0]} name="Importance %">
                {fiData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Residuals vs Predicted</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="predicted" name="Predicted ($k)" tick={{ fill: C.muted, fontSize: 10 }}
                label={{ value: "Predicted ($k)", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
              <YAxis dataKey="residual" name="Residual ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
              <ReferenceLine y={0} stroke="#f43f5e" strokeDasharray="4 2" />
              <Tooltip contentStyle={TT} />
              <Scatter data={residScatter} fill={CHART_COLORS[3]} fillOpacity={0.6} name="Residual" />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <section>
        <SectionTitle>Prediction Error Analysis</SectionTitle>
        <Card>
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0" style={{ background: C.card }}>
                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                  {["Actual ($k)", "Predicted ($k)", "Error ($k)", "Abs Error ($k)", "Error %"].map(h => (
                    <th key={h} className="py-2.5 px-3 text-left text-xs font-semibold uppercase tracking-wider" style={{ color: C.muted }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {errorTable.map((row, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 === 0 ? "transparent" : "#0e142008" }}>
                    <td className="py-2 px-3 font-mono text-xs" style={{ color: C.text }}>{row.actual}</td>
                    <td className="py-2 px-3 font-mono text-xs" style={{ color: C.text }}>{row.predicted}</td>
                    <td className="py-2 px-3 font-mono text-xs" style={{ color: +row.error > 0 ? CHART_COLORS[1] : CHART_COLORS[3] }}>{row.error}</td>
                    <td className="py-2 px-3 font-mono text-xs" style={{ color: C.text }}>{row.absError}</td>
                    <td className="py-2 px-3 font-mono text-xs" style={{ color: +row.pct > 20 ? CHART_COLORS[2] : C.muted }}>{row.pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>
    </div>
  );
}
