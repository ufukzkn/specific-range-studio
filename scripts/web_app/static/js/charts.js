/* ======================================================================
   Specific Range Studio — Plotly Chart Helpers
   ====================================================================== */

const ChartTheme = {
  bg: 'rgba(0,0,0,0)',
  paper: 'rgba(0,0,0,0)',
  gridColor: 'rgba(148, 163, 184, 0.08)',
  textColor: '#94a3b8',
  font: { family: 'Inter, sans-serif', size: 12, color: '#94a3b8' },
  xgboostColor: '#fbbf24',
  ftColor: '#a78bfa',
  actualColor: '#34d399',
  errorXgbColor: '#f97316',
  errorFtColor: '#c084fc',
  colors: [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#ec4899', '#14b8a6', '#f43f5e', '#6366f1',
  ],
};

const baseLayout = {
  paper_bgcolor: ChartTheme.paper,
  plot_bgcolor: ChartTheme.bg,
  font: ChartTheme.font,
  margin: { l: 60, r: 30, t: 40, b: 50 },
  xaxis: {
    gridcolor: ChartTheme.gridColor,
    zerolinecolor: ChartTheme.gridColor,
    color: ChartTheme.textColor,
  },
  yaxis: {
    gridcolor: ChartTheme.gridColor,
    zerolinecolor: ChartTheme.gridColor,
    color: ChartTheme.textColor,
  },
  legend: {
    bgcolor: 'rgba(0,0,0,0)',
    font: { color: ChartTheme.textColor, size: 11 },
  },
  hoverlabel: {
    bgcolor: '#1e293b',
    bordercolor: '#334155',
    font: { color: '#f1f5f9', family: 'Inter, sans-serif', size: 12 },
  },
};

const plotlyConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

/**
 * Render the 3-trace comparison chart: Actual vs XGBoost vs FT-Transformer.
 */
function renderComparisonScatter(containerId, data) {
  if (!data || data.length === 0) {
    document.getElementById(containerId).innerHTML =
      '<div class="empty-state"><div class="empty-state__icon">📊</div><div class="empty-state__text">Karşılaştırma verisi bulunamadı</div></div>';
    return;
  }

  const actuals = data.map(r => r.actual_specific_range);
  const xgbPreds = data.map(r => r.xgboost_predicted);
  const ftPreds = data.map(r => r.ft_transformer_predicted);
  const indices = data.map((_, i) => i);
  const hoverTexts = data.map(r =>
    `Engine: ${r.engine_type}<br>Alt: ${r.altitude} ft<br>` +
    `Mach: ${r.mach}<br>GW: ${r.gross_weight} lb<br>` +
    `Actual: ${r.actual_specific_range?.toFixed(6)}`
  );

  const traces = [
    {
      x: indices, y: actuals, name: 'Actual',
      mode: 'markers', marker: { color: ChartTheme.actualColor, size: 5, opacity: 0.7 },
      hovertext: hoverTexts, hoverinfo: 'text+y',
    },
    {
      x: indices, y: xgbPreds, name: 'XGBoost',
      mode: 'markers', marker: { color: ChartTheme.xgboostColor, size: 4, symbol: 'cross', opacity: 0.6 },
      hoverinfo: 'y+name',
    },
    {
      x: indices, y: ftPreds, name: 'FT-Transformer',
      mode: 'markers', marker: { color: ChartTheme.ftColor, size: 4, symbol: 'diamond', opacity: 0.6 },
      hoverinfo: 'y+name',
    },
  ];

  const layout = {
    ...baseLayout,
    title: { text: 'Actual vs XGBoost vs FT-Transformer', font: { size: 14, color: '#e2e8f0' } },
    xaxis: { ...baseLayout.xaxis, title: 'Satır İndeksi' },
    yaxis: { ...baseLayout.yaxis, title: 'Specific Range' },
    height: 480,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}

/**
 * Render actual vs predicted scatter plot (45-degree line comparison).
 */
function renderActualVsPredicted(containerId, data) {
  if (!data || data.length === 0) return;

  const actuals = data.map(r => r.actual_specific_range);
  const xgbPreds = data.map(r => r.xgboost_predicted);
  const ftPreds = data.map(r => r.ft_transformer_predicted);
  const minVal = Math.min(...actuals) * 0.95;
  const maxVal = Math.max(...actuals) * 1.05;

  const traces = [
    {
      x: [minVal, maxVal], y: [minVal, maxVal], name: 'Mükemmel Tahmin',
      mode: 'lines', line: { color: 'rgba(148,163,184,0.3)', dash: 'dash', width: 1 },
      hoverinfo: 'skip',
    },
    {
      x: actuals, y: xgbPreds, name: 'XGBoost',
      mode: 'markers', marker: { color: ChartTheme.xgboostColor, size: 5, opacity: 0.6 },
      hoverinfo: 'x+y+name',
    },
    {
      x: actuals, y: ftPreds, name: 'FT-Transformer',
      mode: 'markers', marker: { color: ChartTheme.ftColor, size: 5, opacity: 0.6 },
      hoverinfo: 'x+y+name',
    },
  ];

  const layout = {
    ...baseLayout,
    title: { text: 'Actual vs Predicted (45° Çizgisi)', font: { size: 14, color: '#e2e8f0' } },
    xaxis: { ...baseLayout.xaxis, title: 'Actual Specific Range', range: [minVal, maxVal] },
    yaxis: { ...baseLayout.yaxis, title: 'Predicted Specific Range', range: [minVal, maxVal] },
    height: 480,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}

/**
 * Render error distribution histogram for both models.
 */
function renderErrorHistogram(containerId, data) {
  if (!data || data.length === 0) return;

  const xgbErrors = data.map(r => r.xgboost_absolute_error).filter(v => v != null);
  const ftErrors = data.map(r => r.ft_transformer_absolute_error).filter(v => v != null);

  const traces = [
    {
      x: xgbErrors, name: 'XGBoost', type: 'histogram',
      marker: { color: 'rgba(251, 191, 36, 0.5)', line: { color: '#fbbf24', width: 1 } },
      opacity: 0.7, nbinsx: 40,
    },
    {
      x: ftErrors, name: 'FT-Transformer', type: 'histogram',
      marker: { color: 'rgba(167, 139, 250, 0.5)', line: { color: '#a78bfa', width: 1 } },
      opacity: 0.7, nbinsx: 40,
    },
  ];

  const layout = {
    ...baseLayout,
    title: { text: 'Mutlak Hata Dağılımı', font: { size: 14, color: '#e2e8f0' } },
    barmode: 'overlay',
    xaxis: { ...baseLayout.xaxis, title: 'Absolute Error' },
    yaxis: { ...baseLayout.yaxis, title: 'Frekans' },
    height: 360,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}

/**
 * Render slice-level MAE/RMSE bar chart.
 */
function renderSliceSummaryChart(containerId, data, modelName) {
  if (!data || data.length === 0) return;

  const labels = data.map(r => `${r.engine_type} @ ${r.altitude} ft`);

  const traces = [
    {
      x: labels, y: data.map(r => r.mae), name: 'MAE', type: 'bar',
      marker: { color: 'rgba(59, 130, 246, 0.7)', line: { color: '#3b82f6', width: 1 } },
    },
    {
      x: labels, y: data.map(r => r.rmse), name: 'RMSE', type: 'bar',
      marker: { color: 'rgba(16, 185, 129, 0.7)', line: { color: '#10b981', width: 1 } },
    },
  ];

  const layout = {
    ...baseLayout,
    title: { text: `${modelName}: Slice Bazlı Hata`, font: { size: 14, color: '#e2e8f0' } },
    barmode: 'group',
    xaxis: { ...baseLayout.xaxis, tickangle: -45 },
    yaxis: { ...baseLayout.yaxis, title: 'Error' },
    height: 400,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}

/**
 * Render metric comparison bar chart (two models side by side).
 */
function renderMetricComparisonChart(containerId, xgbMetrics, ftMetrics) {
  const metricNames = ['MAE', 'RMSE', 'MAPE'];
  const xgbValues = [xgbMetrics.mae, xgbMetrics.rmse, xgbMetrics.mape];
  const ftValues = [ftMetrics.mae, ftMetrics.rmse, ftMetrics.mape];

  const traces = [
    {
      x: metricNames, y: xgbValues, name: 'XGBoost', type: 'bar',
      marker: { color: 'rgba(251, 191, 36, 0.7)', line: { color: '#fbbf24', width: 1 } },
      text: xgbValues.map(v => v.toFixed(6)), textposition: 'outside',
      textfont: { size: 10, color: '#fbbf24' },
    },
    {
      x: metricNames, y: ftValues, name: 'FT-Transformer', type: 'bar',
      marker: { color: 'rgba(167, 139, 250, 0.7)', line: { color: '#a78bfa', width: 1 } },
      text: ftValues.map(v => v.toFixed(6)), textposition: 'outside',
      textfont: { size: 10, color: '#a78bfa' },
    },
  ];

  const layout = {
    ...baseLayout,
    barmode: 'group',
    yaxis: { ...baseLayout.yaxis, title: 'Değer' },
    height: 340,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}

/**
 * Render estimated cost components for both models.
 * Lower bars are better; fit score itself is rendered in summary cards.
 */
function renderCostSimulatorChart(containerId, models) {
  const container = document.getElementById(containerId);
  if (!container || !models?.xgboost || !models?.ft_transformer) return;

  const xgb = models.xgboost;
  const ft = models.ft_transformer;
  const categories = ['Accuracy Cost', 'Latency Cost', 'Memory Cost', 'Composite Cost'];

  const traces = [
    {
      x: categories,
      y: [xgb.accuracy_component, xgb.latency_component, xgb.memory_component, xgb.combined_cost],
      name: 'XGBoost',
      type: 'bar',
      marker: { color: 'rgba(251, 191, 36, 0.72)', line: { color: '#fbbf24', width: 1 } },
      text: [xgb.accuracy_component, xgb.latency_component, xgb.memory_component, xgb.combined_cost].map(v => v.toFixed(3)),
      textposition: 'outside',
      textfont: { size: 10, color: '#fbbf24' },
    },
    {
      x: categories,
      y: [ft.accuracy_component, ft.latency_component, ft.memory_component, ft.combined_cost],
      name: 'FT-Transformer',
      type: 'bar',
      marker: { color: 'rgba(167, 139, 250, 0.72)', line: { color: '#a78bfa', width: 1 } },
      text: [ft.accuracy_component, ft.latency_component, ft.memory_component, ft.combined_cost].map(v => v.toFixed(3)),
      textposition: 'outside',
      textfont: { size: 10, color: '#a78bfa' },
    },
  ];

  const layout = {
    ...baseLayout,
    title: { text: 'Tahmini Maliyet Bileşenleri (düşük = iyi)', font: { size: 14, color: '#e2e8f0' } },
    barmode: 'group',
    yaxis: { ...baseLayout.yaxis, title: 'Göreli Maliyet' },
    height: 340,
  };

  Plotly.newPlot(containerId, traces, layout, plotlyConfig);
}
