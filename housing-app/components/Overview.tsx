"use client";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, CartesianGrid,
} from "recharts";
import { KPI, SectionTitle, Card, C, TT, CHART_COLORS } from "./ui";
import type { HousingRow, Metrics } from "@/lib/data";
import { mean, linearRegression, FEATURE_DESCRIPTIONS } from "@/lib/data";

export default function Overview({ data, metrics }: { data: HousingRow[]; metrics: Metrics }) {
  const prices = data.map(d => d.MEDV);
  const avgPrice = mean(prices);

  // Price histogram
  const bins = 20;
  const pMin = Math.min(...prices), pMax = Math.max(...prices);
  const step = (pMax - pMin) / bins;
  const histData = Array.from({ length: bins }, (_, i) => {
    const lo = pMin + i * step;
    return { range: `$${lo.toFixed(0)}k`, count: prices.filter(p => p >= lo && p < lo + step).length };
  });

  // Rooms vs Price scatter
  const scatterRooms = data.filter((_, i) => i % 2 === 0).map(d => ({ x: +d.RM.toFixed(1), y: d.MEDV }));

  // Crime vs Price scatter
  const scatterCrime = data.filter((_, i) => i % 2 === 0).map(d => ({ x: +d.CRIM.toFixed(2), y: d.MEDV }));

  // Feature importance top 5
  const topFeatures = Object.entries(metrics.feature_importances)
    .sort((a, b) => b[1] - a[1]).slice(0, 5)
    .map(([k, v]) => ({ name: k, pct: +(v * 100).toFixed(1) }));

  // Simple computed stats (realistic language)
  const highRoomsMean = mean(data.filter(d => d.RM >= 7).map(d => d.MEDV));
  const lowRoomsMean  = mean(data.filter(d => d.RM < 5).map(d => d.MEDV));
  const riverYes = mean(data.filter(d => d.CHAS === 1).map(d => d.MEDV));
  const riverNo  = mean(data.filter(d => d.CHAS === 0).map(d => d.MEDV));

  const corrRM = (() => {
    const x = data.map(d => d.RM), y = prices, mx = mean(x), my = mean(y);
    const num = x.reduce((s,xi,i)=>s+(xi-mx)*(y[i]-my),0);
    const den = Math.sqrt(x.reduce((s,xi)=>s+(xi-mx)**2,0)*y.reduce((s,yi)=>s+(yi-my)**2,0));
    return (num/den).toFixed(2);
  })();
  const corrLSTAT = (() => {
    const x = data.map(d => d.LSTAT), y = prices, mx = mean(x), my = mean(y);
    const num = x.reduce((s,xi,i)=>s+(xi-mx)*(y[i]-my),0);
    const den = Math.sqrt(x.reduce((s,xi)=>s+(xi-mx)**2,0)*y.reduce((s,yi)=>s+(yi-my)**2,0));
    return (num/den).toFixed(2);
  })();

  return (
    <div className="space-y-10 max-w-5xl mx-auto">

      {/* ── HEADER ── */}
      <div>
        <div className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: C.muted }}>
          Data Analysis Project · Boston Housing Dataset
        </div>
        <h1 className="text-2xl font-bold mb-2" style={{ color: C.text }}>
          Boston House Price Analysis
        </h1>
        <p className="text-sm leading-relaxed" style={{ color: "#9ca3af", maxWidth: 620 }}>
          This project explores what factors influence house prices in the Boston area.
          Using a publicly available dataset of 506 properties, I analyzed key features,
          identified patterns, and built a basic prediction model.
        </p>
      </div>

      {/* ── SECTION 1: DATASET OVERVIEW ── */}
      <section>
        <SectionTitle>📦 Dataset Overview</SectionTitle>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
          <KPI label="Total Records"  value="506"   sub="Boston properties"      />
          <KPI label="Features"       value="13"    sub="Input variables"         color={CHART_COLORS[1]} />
          <KPI label="Avg Price"      value={`$${(avgPrice*1000).toLocaleString(undefined,{maximumFractionDigits:0})}`} sub="Median home value" color={CHART_COLORS[2]} />
          <KPI label="Price Range"    value={`$${(pMin*1000/1000).toFixed(0)}k–$${(pMax*1000/1000).toFixed(0)}k`} sub="Min to max" color={CHART_COLORS[3]} />
        </div>

        {/* How it works */}
        <Card>
          <div className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: CHART_COLORS[0] }}>
            How This Project Works
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            {[
              {
                step: "1. Dataset",
                body: "Used the Boston Housing dataset from UCI ML Repository. It contains 506 records with 13 socioeconomic and environmental features collected in 1978.",
                color: CHART_COLORS[0],
              },
              {
                step: "2. Preprocessing",
                body: "Checked for missing values (none found). Applied StandardScaler to normalize features so that variables with large ranges don't dominate the model.",
                color: CHART_COLORS[1],
              },
              {
                step: "3. Model Training",
                body: "Trained a Random Forest model using an 80/20 train-test split. Used 5-fold cross-validation to check if the model generalizes beyond the training data.",
                color: CHART_COLORS[2],
              },
            ].map(({ step, body, color }) => (
              <div key={step}>
                <div className="font-semibold mb-2 text-xs" style={{ color }}>{step}</div>
                <p style={{ color: "#9ca3af", lineHeight: 1.7 }}>{body}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* ── SECTION 2: EDA & CHARTS ── */}
      <section>
        <SectionTitle>📊 Exploratory Data Analysis</SectionTitle>
        <p className="text-sm mb-5" style={{ color: C.muted }}>
          Before building the model, I explored the data visually to understand distributions and relationships.
        </p>

        {/* Chart row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: C.muted }}>Price Distribution</p>
            <p className="text-xs mb-4" style={{ color: "#4b5563" }}>Most homes are priced between $15k–$25k. A few high-value outliers exist above $40k.</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="range" tick={{ fill: C.muted, fontSize: 9 }} interval={3} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Bar dataKey="count" fill={CHART_COLORS[0]} radius={[3,3,0,0]} name="Properties" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: C.muted }}>Rooms vs Price</p>
            <p className="text-xs mb-4" style={{ color: "#4b5563" }}>A clear positive trend — homes with more rooms tend to have higher prices.</p>
            <ResponsiveContainer width="100%" height={200}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name="Avg Rooms" tick={{ fill: C.muted, fontSize: 10 }}
                  label={{ value: "Avg Rooms", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
                <YAxis dataKey="y" name="Price ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Scatter data={scatterRooms} fill={CHART_COLORS[1]} fillOpacity={0.6} name="Property" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Chart row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: C.muted }}>Crime Rate vs Price</p>
            <p className="text-xs mb-4" style={{ color: "#4b5563" }}>Higher crime areas generally show lower home values, though the relationship is non-linear.</p>
            <ResponsiveContainer width="100%" height={200}>
              <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name="Crime Rate" tick={{ fill: C.muted, fontSize: 10 }}
                  label={{ value: "Crime Rate", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 10 }} />
                <YAxis dataKey="y" name="Price ($k)" tick={{ fill: C.muted, fontSize: 10 }} />
                <Tooltip contentStyle={TT} />
                <Scatter data={scatterCrime} fill={CHART_COLORS[3]} fillOpacity={0.55} name="Property" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: C.muted }}>Top 5 Features by Importance</p>
            <p className="text-xs mb-4" style={{ color: "#4b5563" }}>The model relies most on room count and socioeconomic status to make predictions.</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={topFeatures} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis type="number" tick={{ fill: C.muted, fontSize: 10 }} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={38} />
                <Tooltip contentStyle={TT} formatter={(v) => [`${v}%`, "Importance"]} />
                <Bar dataKey="pct" radius={[0,4,4,0]} fill={CHART_COLORS[2]} name="Importance %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </section>

      {/* ── SECTION 3: INSIGHTS ── */}
      <section>
        <SectionTitle>💡 Key Insights</SectionTitle>
        <p className="text-sm mb-5" style={{ color: C.muted }}>
          These are the main patterns I observed from the data. I tried to keep the language realistic rather than overstating findings.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              title: "Room count is the strongest predictor",
              body: `Homes with 7+ rooms have an average price of $${(highRoomsMean*1000).toLocaleString(undefined,{maximumFractionDigits:0})} vs $${(lowRoomsMean*1000).toLocaleString(undefined,{maximumFractionDigits:0})} for homes with under 5 rooms. A strong positive correlation was observed (r = ${corrRM}).`,
              color: CHART_COLORS[0],
            },
            {
              title: "Socioeconomic status strongly affects price",
              body: `LSTAT (% lower-status population) shows the strongest negative correlation with price (r = ${corrLSTAT}). Areas with higher lower-status % consistently show lower home values.`,
              color: CHART_COLORS[1],
            },
            {
              title: "River proximity shows a modest premium",
              body: `Properties near the Charles River average $${(riverYes*1000).toLocaleString(undefined,{maximumFractionDigits:0})} vs $${(riverNo*1000).toLocaleString(undefined,{maximumFractionDigits:0})} for non-river properties. The sample of river-adjacent homes is small (${data.filter(d=>d.CHAS===1).length} records), so this finding should be interpreted cautiously.`,
              color: CHART_COLORS[2],
            },
            {
              title: "Crime has a non-linear negative effect",
              body: "Higher crime rates are associated with lower prices, but the relationship isn't perfectly linear. A few high-crime areas still have moderate prices, possibly due to other compensating factors.",
              color: CHART_COLORS[3],
            },
          ].map(({ title, body, color }) => (
            <Card key={title}>
              <div className="text-xs font-semibold mb-2" style={{ color }}>{title}</div>
              <p className="text-sm" style={{ color: "#9ca3af", lineHeight: 1.7 }}>{body}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* ── SECTION 4: MODEL PERFORMANCE ── */}
      <section>
        <SectionTitle>🤖 Model Performance</SectionTitle>
        <p className="text-sm mb-5" style={{ color: C.muted }}>
          I used a Random Forest model. Below are the results and what each metric means in plain terms.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <KPI label="R² Score" value={metrics.r2.toFixed(3)} sub="Variance explained" color={CHART_COLORS[0]} icon="🎯" />
          <KPI label="MAE" value={`$${(metrics.mae*1000).toLocaleString(undefined,{maximumFractionDigits:0})}`} sub="Avg prediction error" color={CHART_COLORS[1]} icon="📏" />
          <KPI label="RMSE" value={`$${(metrics.rmse*1000).toLocaleString(undefined,{maximumFractionDigits:0})}`} sub="Root mean squared error" color={CHART_COLORS[2]} icon="📐" />
        </div>

        <Card>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            {[
              {
                metric: "R² = " + metrics.r2.toFixed(3),
                explain: "The model explains about 87.8% of the variation in house prices. A score closer to 1.0 is better. This is a reasonably good result for this dataset.",
                color: CHART_COLORS[0],
              },
              {
                metric: "MAE = $" + (metrics.mae*1000).toLocaleString(undefined,{maximumFractionDigits:0}),
                explain: "On average, the model's predictions are off by about $2,078. This means if a house is worth $20,000, the model might predict $18,000 or $22,000.",
                color: CHART_COLORS[1],
              },
              {
                metric: "RMSE = $" + (metrics.rmse*1000).toLocaleString(undefined,{maximumFractionDigits:0}),
                explain: "Similar to MAE but penalizes larger errors more. The fact that RMSE is close to MAE suggests the model doesn't make many extreme mistakes.",
                color: CHART_COLORS[2],
              },
            ].map(({ metric, explain, color }) => (
              <div key={metric}>
                <div className="font-bold mb-2 font-mono" style={{ color }}>{metric}</div>
                <p style={{ color: "#9ca3af", lineHeight: 1.7 }}>{explain}</p>
              </div>
            ))}
          </div>
        </Card>

        <div className="mt-4 p-4 rounded-xl text-sm" style={{ background: "#111827", border: "1px solid #1f2937" }}>
          <span className="font-semibold" style={{ color: CHART_COLORS[0] }}>Why Random Forest?</span>
          <span style={{ color: "#9ca3af" }}> House prices don't follow a straight line — they're influenced by complex interactions between features. Random Forest handles these non-linear relationships better than simple linear regression, and it's less prone to overfitting on small datasets like this one.</span>
        </div>
      </section>

      {/* ── SECTION 5: LIMITATIONS ── */}
      <section>
        <SectionTitle>⚠️ Limitations</SectionTitle>
        <Card>
          <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: "#f59e0b" }}>
            Things I'd improve with more time or data
          </p>
          <div className="space-y-3">
            {[
              "The dataset only has 506 rows — this is quite small for machine learning. Results may not be statistically robust.",
              "Data was collected in 1978. House prices and neighborhood dynamics have changed significantly since then.",
              "The model was trained on Boston data only. It would likely not generalize well to other cities or regions.",
              `Cross-validated R² is ${metrics.cv_r2_mean.toFixed(2)} ± ${metrics.cv_r2_std.toFixed(2)}, which shows some variance across folds — the model's performance isn't perfectly consistent.`,
              "Some features like the B index have ethical concerns and would be excluded in a real-world analysis.",
              "The predictor on this site uses a simplified linear approximation — not the actual Random Forest model — since running Python in the browser isn't straightforward.",
            ].map((text, i) => (
              <div key={i} className="flex gap-3 text-sm" style={{ color: "#9ca3af" }}>
                <span style={{ color: "#f59e0b", marginTop: 2 }}>•</span>
                <span style={{ lineHeight: 1.7 }}>{text}</span>
              </div>
            ))}
          </div>
        </Card>
      </section>

    </div>
  );
}
