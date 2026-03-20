export type HousingRow = {
  CRIM: number; ZN: number; INDUS: number; CHAS: number;
  NOX: number; RM: number; AGE: number; DIS: number;
  RAD: number; TAX: number; PTRATIO: number; B: number;
  LSTAT: number; MEDV: number;
};

export type Metrics = {
  r2: number; rmse: number; mae: number; mse: number;
  cv_r2_mean: number; cv_r2_std: number;
  train_size: number; test_size: number;
  features: string[];
  feature_importances: Record<string, number>;
  y_test: number[]; y_pred: number[];
};

export const FEATURE_LABELS: Record<string, string> = {
  CRIM: "Crime Rate", ZN: "Residential Zone %", INDUS: "Industrial Area %",
  CHAS: "Charles River", NOX: "NOx Concentration", RM: "Avg Rooms",
  AGE: "Pre-1940 Units %", DIS: "Distance to Employment",
  RAD: "Highway Access", TAX: "Property Tax Rate",
  PTRATIO: "Pupil-Teacher Ratio", B: "B Index",
  LSTAT: "Lower Status %", MEDV: "Home Value ($k)",
};

export const FEATURE_DESCRIPTIONS: Record<string, string> = {
  CRIM: "Per capita crime rate by town",
  ZN: "Proportion of residential land zoned for large lots",
  INDUS: "Proportion of non-retail business acres per town",
  CHAS: "Charles River dummy variable (1 if tract bounds river)",
  NOX: "Nitric oxides concentration (parts per 10 million)",
  RM: "Average number of rooms per dwelling",
  AGE: "Proportion of owner-occupied units built prior to 1940",
  DIS: "Weighted distances to five Boston employment centres",
  RAD: "Index of accessibility to radial highways",
  TAX: "Full-value property-tax rate per $10,000",
  PTRATIO: "Pupil-teacher ratio by town",
  B: "Demographic index by town",
  LSTAT: "% lower status of the population",
};

export function pearsonCorr(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const num = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0);
  const den = Math.sqrt(
    x.reduce((s, xi) => s + (xi - mx) ** 2, 0) *
    y.reduce((s, yi) => s + (yi - my) ** 2, 0)
  );
  return den === 0 ? 0 : num / den;
}

export function mean(arr: number[]) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function linearRegression(x: number[], y: number[]) {
  const n = x.length;
  const mx = mean(x), my = mean(y);
  const slope = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0) /
    x.reduce((s, xi) => s + (xi - mx) ** 2, 0);
  const intercept = my - slope * mx;
  return { slope, intercept };
}
