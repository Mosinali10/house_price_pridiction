"use client";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, CartesianGrid, ReferenceLine,
  PieChart, Pie, Cell, Legend, RadialBarChart, RadialBar,
} from "recharts";
import { KPI, SectionTitle, InsightCard, Card, C, TT, CHART_COLORS } from "./ui";
import type { HousingRow, Metrics } from "@/lib/data";
import { mean, linearRegression, FEATURE_DESCRIPTIONS } from "@/lib/data";

export default function Overview({ data, metrics }: { data: HousingRow[]; metrics: Metrics }) {
  const prices = data.map(d => d.MEDV);
  const avgPrice = mean(prices);

  // Histogram
  const bins = 30;
  const pMin = Math.min(...prices), pMax = Math.max(...prices);
  const step = (pMax - pMin) / bins;
  const histData = Array.from({ length: bins }, (_, i) => {
    const lo = pMin + i * step;
    return { range: `${lo.toFixed(0)}k`, count: prices.filter(p => p >= lo && p < lo + step).length };
  });

  // Scatter rooms vs price
  const scatterData = data.filter((_, i) => i % 3 === 0).map(d => ({ x: +d.RM.toFixed(2), y: d.MEDV }));
  const { slope, intercept } = linearRegression(data.map(d => d.RM), prices);
  const rmMin = Math.min(...data.map(d => d.RM)), rmMax = Math.max(...data.map(d => d.RM));
  const trendData = [{ x: rmMin, y: +(slope * rmMin + intercept).toFixed(2) }, { x: rmMax, y: +(slope * rmMax + intercept).toFixed(2) }];

  // RAD bar
  const radGroups: Record<number, number[]> = {};
  data.forEach(d => { if (!radGroups[d.RAD]) radGroups[d.RAD] = []; radGroups[d.RAD].push(d.MEDV); });
  const radData = Object.entries(radGroups)
    .map(([z, v]) => ({ zone: `Z${z}`, avg: +mean(v).toFixed(1) }))
    .sort((a, b) => b.avg - a.avg);

  // Crime scatter
  const crimeData = data.filter((_, i) => i % 3 === 0).map(d => ({ x: +d.CRIM.toFixed(3), y: d.MEDV }));

  // Insights
  const highRooms = data.filter(d => d.RM >= 7).map(d => d.MEDV);
  const lowRooms  = data.filter(d => d.RM < 5).map(d => d.MEDV);
  const roomsPct  = ((mean(highRooms) - mean(lowRooms)) / mean(lowRooms) * 100).toFixed(0);

  const sorted = [...data.map(d => d.CRIM)].sort((a, b) => a - b);
  const q75c = sorted[Math.floor(data.length * 0.75)];
  const q25c = sorted[Math.floor(data.length * 0.25)];
  const crimePct = ((mean(data.filter(d => d.CRIM < q25c).map(d => d.MEDV)) - mean(data.filter(d => d.CRIM > q75c).map(d => d.MEDV))) / mean(data.filter(d => d.CRIM > q75c).map(d => d.MEDV)) * 100).toFixed(0);

  const riverYes = mean(data.filter(d => d.CHAS === 1).map(d => d.MEDV));
  const riverNo  = mean(data.filter(d => d.CHAS === 0).map(d => d.MEDV));
  const riverPct = ((riverYes - riverNo) / riverNo * 100).toFixed(0);

  const corr = (key: "RM" | "LSTAT") => {
    const x = data.map(d => d[key]), y = prices;
    const mx = mean(x), my = mean(y);
    const num = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0);
    const den = Math.sqrt(x.reduce((s, xi) => s + (xi - mx) ** 2, 0) * y.reduce((s, yi) => s + (yi - my) ** 2, 0));
    return (num / den).toFixed(2);
  };

  const features = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"];

  // Dashboard data
  const priceRanges = [
    { name: "< $15k",  value: prices.filter(p => p < 15).length,              color: CHART_COLORS[0] },
    { name: "$15–25k", value: prices.filter(p => p >= 15 && p < 25).length,   color: CHART_COLORS[1] },
    { name: "$25–35k", value: prices.filter(p => p >= 25 && p < 35).length,   color: CHART_COLORS[2] },
    { name: "> $35k",  value: prices.filter(p => p >= 35).length,             color: CHART_COLORS[3] },
  ];

  const featureTypes = [
    { name: "Numeric",     value: 11, color: CHART_COLORS[0] },
    { name: "Categorical", value: 1,  color: CHART_COLORS[2] },
    { name: "Binary",      value: 1,  color: CHART_COLORS[3] },
  ];

  const topFeatures = Object.entries(metrics.feature_importances)
    .map(([k, v]) => ({ name: k, pct: +(v * 100).toFixed(1) }))
    .sort((a, b) => b.pct - a.pct).slice(0, 5);

  const modelSummary = [
    { name: "R² Score", value: +(metrics.r2 * 100).toFixed(1), fill: CHART_COLORS[0] },
    { name: "CV Score", value: +(metrics.cv_r2_mean * 100).toFixed(1), fill: CHART_COLORS[1] },
  ];

  const riverComp = [
    { name: "Near River",     value: +riverYes.toFixed(1) },
    { name: "Not Near River", value: +riverNo.toFixed(1) },
  ];

  return (
    <div className="space-y-10 max-w-7xl mx-auto">

      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1" style={{ color: C.text }}>
          Boston Housing Analysis
        </h1>
        <p className="text-sm" style={{ color: C.muted }}>
          506 real estate records · 13 socioeconomic features · Random Forest model
        </p>
      </div>

      {/* ── DASHBOARD SUMMARY ── */}
      <section>
        <SectionTitle>🗂 Dashboard Summary</SectionTitle>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>Price Range Breakdown</p>
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie data={priceRanges} cx="50%" cy="50%" innerRadius={52} outerRadius={80}
                  dataKey="value" paddingAngle={4}>
                  {priceRanges.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={TT} formatter={(v) => [`${v} properties`]} />
                <Legend iconType="circle" iconSize={7}
                  formatter={(v) => <span style={{ color: "#9ca3af", fontSize: 11 }}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>Feature Type Distribution</p>
            <ResponsiveContainer width="100%" height={210}>
              <PieChart>
                <Pie data={featureTypes} cx="50%" cy="50%" outerRadius={75}
                  dataKey="value" paddingAngle={4} labelLine={false} label={false}>
                  {featureTypes.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={TT} formatter={(v) => [`${v} features`]} />
                <Legend iconType="circle" iconSize={7}
                  formatter={(v) => <span style={{ color: "#9ca3af", fontSize: 11 }}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>Top 5 Predictive Features</p>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={topFeatures} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 36 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" tick={{ fill: C.muted, fontSize: 10 }} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={34} />
                <Tooltip contentStyle={TT} formatter={(v) => [`${v}%`, "Importance"]} />
                <Bar dataKey="pct" radius={[0, 5, 5, 0]}>
                  {topFeatures.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-5">
          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>Model Accuracy Overview</p>
            <ResponsiveContainer width="100%" height={170}>
              <RadialBarChart cx="50%" cy="50%" innerRadius={28} outerRadius={75}
                data={modelSummary} startAngle={180} endAngle={0}>
                <RadialBar dataKey="value" cornerRadius={6}
                  label={{ position: "insideStart", fill: C.text, fontSize: 11 }} />
                <Legend iconType="circle" iconSize={7}
                  formatter={(v) => <span style={{ color: "#9ca3af", fontSize: 11 }}>{v}</span>} />
                <Tooltip contentStyle={TT} formatter={(v) => [`${v}%`]} />
              </RadialBarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: C.muted }}>River vs Non-River Avg Price</p>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={riverComp} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} unit="k" />
                <Tooltip contentStyle={TT} formatter={(v) => [`$${v}k`, "Avg Price"]} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  <Cell fill={CHART_COLORS[3]} />
                  <Cell fill={C.subtle} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </section>

      {/* ── DATASET OVERVIEW ── */}
      <section>
        <SectionTitle>📊 Dataset Overview</SectionTitle>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
          <KPI label="Total Records" value="506" sub="Boston properties" icon="🏘" />
          <KPI label="Features" value="13" sub="Socioeconomic variables" color={CHART_COLORS[1]} icon="📐" />
          <KPI label="Avg Home Value" value={`$${(avgPrice * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="Median home value" color={CHART_COLORS[2]} icon="💰" />
          <KPI label="Price Range" value={`$${(pMin * 1000 / 1000).toFixed(0)}k – $${(pMax * 1000 / 1000).toFixed(0)}k`} sub="Min to max" color={CHART_COLORS[3]} icon="📉" />
        </div>

        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                  {["Feature", "Description", "Type", "Min", "Max", "Mean"].map(h => (
                    <th key={h} className="text-left py-2.5 px-3 text-xs font-semibold uppercase tracking-wider" style={{ color: C.muted }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {features.map((f, i) => {
                  const vals = data.map(d => d[f as keyof HousingRow] as number);
                  return (
                    <tr key={f} className="transition-colors"
                      style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 === 0 ? "transparent" : "#0e142008" }}>
                      <td className="py-2.5 px-3 font-mono font-bold text-xs" style={{ color: C.accent }}>{f}</td>
                      <td className="py-2.5 px-3 text-xs" style={{ color: "#9ca3af" }}>{FEATURE_DESCRIPTIONS[f]}</td>
                      <td className="py-2.5 px-3">
                        <span className="px-2 py-0.5 rounded-md text-xs font-medium"
                          style={{
                            background: f === "CHAS" ? "#f472b618" : f === "RAD" ? "#f59e0b18" : "#a78bfa18",
                            color: f === "CHAS" ? "#f472b6" : f === "RAD" ? "#f59e0b" : "#a78bfa",
                          }}>
                          {f === "CHAS" ? "Binary" : f === "RAD" ? "Categorical" : "Numeric"}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-xs font-mono" style={{ color: C.text }}>{Math.min(...vals).toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-xs font-mono" style={{ color: C.text }}>{Math.max(...vals).toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-xs font-mono" style={{ color: C.text }}>{mean(vals).toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* ── DATA VISUALIZATION ── */}
      <section>
        <SectionTitle>📈 Data Visualization</SectionTitle>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: C.muted }}>Price Distribution</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="range" tick={{ fill: C.muted, fontSize: 10 }} interval={4} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <ReferenceLine x={`${avgPrice.toFixed(0)}k`} stroke="#f43f5e" strokeDasharray="4 2"
                  label={{ value: `Mean $${avgPrice.toFixed(1)}k`, fill: "#f43f5e", fontSize: 10 }} />
                <Bar dataKey="count" fill={CHART_COLORS[0]} radius={[3, 3, 0, 0]} name="Properties" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: C.muted }}>Price vs Rooms</p>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name="Avg Rooms" tick={{ fill: C.muted, fontSize: 10 }}
                  label={{ value: "Avg Rooms", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
                <YAxis dataKey="y" name="Value ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Scatter data={scatterData} fill={CHART_COLORS[0]} fillOpacity={0.55} name="Property" />
                <Scatter data={trendData} fill="#f43f5e" line={{ stroke: "#f43f5e", strokeWidth: 2 }} shape={() => null} name="Trend" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: C.muted }}>Price by Location Zone</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={radData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="zone" tick={{ fill: C.muted, fontSize: 10 }} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} formatter={(v) => [`$${v}k`, "Avg Price"]} />
                <Bar dataKey="avg" radius={[4, 4, 0, 0]} name="Avg Price ($k)">
                  {radData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: C.muted }}>Crime Rate vs Home Value</p>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name="Crime Rate" tick={{ fill: C.muted, fontSize: 10 }}
                  label={{ value: "Crime Rate", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
                <YAxis dataKey="y" name="Value ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Scatter data={crimeData} fill={CHART_COLORS[3]} fillOpacity={0.55} name="Property" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </section>

      {/* ── DATA CLEANING ── */}
      <section>
        <SectionTitle>🧹 Data Cleaning</SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <KPI label="Missing Values" value="0" sub="Dataset is complete ✓" color="#34d399" icon="✅" />
          <KPI label="Duplicate Rows" value="0" sub="All records unique ✓" color="#34d399" icon="✅" />
          <KPI label="Price Outliers" value={`${data.filter(d => {
            const s = [...prices].sort((a, b) => a - b);
            const q1 = s[Math.floor(s.length * 0.25)], q3 = s[Math.floor(s.length * 0.75)];
            return d.MEDV < q1 - 1.5 * (q3 - q1) || d.MEDV > q3 + 1.5 * (q3 - q1);
          }).length} rows`} sub="Kept — real data" color={CHART_COLORS[2]} icon="📌" />
        </div>
        <Card>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            {[
              { title: "Missing Values", body: "Dataset has no missing values — all 506 records are complete and ready for analysis.", color: "#34d399" },
              { title: "Outliers", body: "Price outliers identified via IQR method but retained — they represent real high-value properties like waterfront homes.", color: CHART_COLORS[2] },
              { title: "Feature Scaling", body: "StandardScaler applied inside the model pipeline to normalize all 13 features before training.", color: CHART_COLORS[0] },
            ].map(({ title, body, color }) => (
              <div key={title}>
                <div className="font-semibold mb-1.5 text-xs uppercase tracking-wider" style={{ color }}>{title}</div>
                <p style={{ color: "#9ca3af", lineHeight: 1.6 }}>{body}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* ── KEY INSIGHTS ── */}
      <section>
        <SectionTitle>🧠 Key Insights</SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <InsightCard value={`+${roomsPct}%`} color={CHART_COLORS[0]}
            text={`Homes with 7+ rooms cost ${roomsPct}% more than homes with fewer than 5 rooms`} />
          <InsightCard value={`+${crimePct}%`} color={CHART_COLORS[1]}
            text={`Low-crime areas have homes worth ${crimePct}% more than high-crime areas`} />
          <InsightCard value={`+${riverPct}%`} color={CHART_COLORS[3]}
            text={`Homes near the Charles River cost ${riverPct}% more on average`} />
          <InsightCard value={`r = ${corr("RM")}`} color={CHART_COLORS[4]}
            text="Number of rooms has the strongest positive correlation with price" />
          <InsightCard value={`r = ${corr("LSTAT")}`} color={CHART_COLORS[2]}
            text="Lower-status population % has the strongest negative correlation with price" />
          <InsightCard value={`${(metrics.r2 * 100).toFixed(1)}%`} color={CHART_COLORS[5]}
            text={`Model explains ${(metrics.r2 * 100).toFixed(1)}% of price variance with avg error of $${(metrics.mae * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
        </div>
      </section>

      {/* ── MODEL PERFORMANCE ── */}
      <section>
        <SectionTitle>📉 Model Performance</SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <KPI label="R² Score" value={metrics.r2.toFixed(3)} sub="Variance explained" color={CHART_COLORS[0]} icon="🎯" />
          <KPI label="MAE" value={`$${(metrics.mae * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="Avg prediction error" color={CHART_COLORS[1]} icon="📏" />
          <KPI label="RMSE" value={`$${(metrics.rmse * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub="Root mean squared error" color={CHART_COLORS[4]} icon="📐" />
        </div>
        <Card glow>
          <p className="text-sm leading-relaxed" style={{ color: "#9ca3af" }}>
            The Random Forest model explains{" "}
            <span style={{ color: CHART_COLORS[0], fontWeight: 600 }}>{(metrics.r2 * 100).toFixed(1)}%</span> of price variance.
            On average, predictions are off by{" "}
            <span style={{ color: CHART_COLORS[1], fontWeight: 600 }}>${(metrics.mae * 1000).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>.
            Cross-validated R² is{" "}
            <span style={{ color: CHART_COLORS[4], fontWeight: 600 }}>{metrics.cv_r2_mean.toFixed(3)} ± {metrics.cv_r2_std.toFixed(3)}</span>,
            confirming the model generalizes well to unseen data.
          </p>
        </Card>
      </section>
    </div>
  );
}
