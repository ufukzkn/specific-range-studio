/* ======================================================================
   Specific Range Studio — Main Application Logic
   ====================================================================== */

// ---------------------------------------------------------------------------
// API Helper
// ---------------------------------------------------------------------------

async function api(endpoint, options = {}) {
  const url = endpoint.startsWith('http') ? endpoint : endpoint;
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || `API error ${resp.status}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------

function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const icons = { success: '✓', error: '✗', info: 'ℹ', warning: '⚠' };
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Tab Navigation
// ---------------------------------------------------------------------------

function initTabs() {
  document.querySelectorAll('.tab-nav__item').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      document.querySelectorAll('.tab-nav__item').forEach(b => b.classList.remove('tab-nav__item--active'));
      btn.classList.add('tab-nav__item--active');

      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('tab-content--active'));
      const panel = document.getElementById(`tab-${target}`);
      if (panel) panel.classList.add('tab-content--active');

      // Lazy-load tab data
      if (target === 'overview' && !window._overviewLoaded) loadOverview();
      if (target === 'compare' && !window._compareLoaded) loadComparison();
      if (target === 'predict' && !window._predictLoaded) loadPredictionTab();
      if (target === 'nomogram' && !window._nomogramLoaded) loadNomogramTab();
      if (target === 'setup' && !window._setupLoaded) loadSetupTab();
    });
  });
}

// ---------------------------------------------------------------------------
// Status Badges
// ---------------------------------------------------------------------------

async function loadStatus() {
  try {
    const status = await api('/api/status');
    const container = document.getElementById('status-badges');
    if (!container) return;

    const badges = [
      { label: 'Veri', ok: status.data_ready },
      { label: 'XGB', ok: status.xgboost_model },
      { label: 'FT', ok: status.ft_transformer_model },
      { label: 'Rapor', ok: status.xgboost_report && status.ft_transformer_report },
    ];

    container.innerHTML = badges.map(b =>
      `<div class="status-badge status-badge--${b.ok ? 'ok' : 'warn'}">
        <span class="status-badge__dot"></span>${b.label}
      </div>`
    ).join('');
  } catch (e) {
    console.warn('Status check failed:', e);
  }
}

// ---------------------------------------------------------------------------
// Tab 1: Overview (Genel Bakış)
// ---------------------------------------------------------------------------

async function loadOverview() {
  window._overviewLoaded = true;

  try {
    // Load metrics for the selected model
    const model = document.getElementById('overview-model-select')?.value || 'xgboost';
    const metrics = await api(`/api/report/${model}/metrics`);

    // Render metric cards
    const grid = document.getElementById('overview-metrics');
    if (grid && metrics) {
      grid.innerHTML = [
        metricCardHTML('Satir Sayisi', formatInt(metrics.rows), '', '', METRIC_DESC.rows),
        metricCardHTML('MAE', formatFloat(metrics.mae, 6), '', 'accent', METRIC_DESC.mae),
        metricCardHTML('RMSE', formatFloat(metrics.rmse, 6), '', 'accent', METRIC_DESC.rmse),
        metricCardHTML('MAPE', formatFloat(metrics.mape, 4) + '%', '', '', METRIC_DESC.mape),
        metricCardHTML('R2', formatFloat(metrics.r2, 6), '', 'success', METRIC_DESC.r2),
      ].join('');
    }

    // Load slice summary
    const slices = await api(`/api/report/${model}/slice-summary`);
    const sliceCols = [
      { key: 'engine_type', label: 'Engine Type' },
      { key: 'altitude', label: 'Altitude (ft)', format: 'int' },
      { key: 'rows', label: 'Rows', format: 'int' },
      { key: 'mae', label: 'MAE', format: 'float6' },
      { key: 'rmse', label: 'RMSE', format: 'float6' },
      { key: 'mape', label: 'MAPE', format: 'float4' },
      { key: 'r2', label: 'R²', format: 'float6' },
    ];
    renderDataTable('overview-slice-table', slices, sliceCols, { errorColumn: 'mae', maxErrorValue: Math.max(...slices.map(r => r.mae), 0.001) });

    // Slice summary chart
    renderSliceSummaryChart('overview-slice-chart', slices, model);

  } catch (err) {
    showToast('Genel bakış yüklenemedi: ' + err.message, 'error');
  }
}

function switchOverviewModel() {
  window._overviewLoaded = false;
  loadOverview();
}

// ---------------------------------------------------------------------------
// Tab 2: Comparison (Karşılaştırma)
// ---------------------------------------------------------------------------

let compareState = { page: 1 };
let costSimDebounceTimer = null;

async function loadComparison() {
  window._compareLoaded = true;
  try {
    // Load metrics comparison
    const metrics = await api('/api/compare/metrics');
    if (metrics.xgboost && metrics.ft_transformer) {
      renderCompareMetrics(metrics);
      renderMetricComparisonChart('compare-metric-chart', metrics.xgboost, metrics.ft_transformer);
    }

    updateCostControlLabels();
    await loadCostSimulation();

    // Load rows
    await loadCompareRows();

  } catch (err) {
    showToast('Karsilastirma yuklenemedi: ' + err.message, 'error');
  }
}

function getCostInputs() {
  return {
    accuracy_weight: parseFloat(document.getElementById('cost-accuracy')?.value || '55'),
    latency_weight: parseFloat(document.getElementById('cost-latency')?.value || '25'),
    memory_weight: parseFloat(document.getElementById('cost-memory')?.value || '20'),
    cpu_speed_factor: parseFloat(document.getElementById('cost-cpu-speed')?.value || '1'),
    ram_budget_gb: parseFloat(document.getElementById('cost-ram-budget')?.value || '8'),
  };
}

function updateCostControlLabels() {
  const inputs = getCostInputs();
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  setText('cost-accuracy-value', `${Math.round(inputs.accuracy_weight)}%`);
  setText('cost-latency-value', `${Math.round(inputs.latency_weight)}%`);
  setText('cost-memory-value', `${Math.round(inputs.memory_weight)}%`);
  setText('cost-cpu-speed-value', `${inputs.cpu_speed_factor.toFixed(2)}x`);
  setText('cost-ram-budget-value', `${inputs.ram_budget_gb.toFixed(1)} GB`);
}

function debounceCostSimulation() {
  updateCostControlLabels();
  clearTimeout(costSimDebounceTimer);
  costSimDebounceTimer = setTimeout(() => {
    loadCostSimulation();
  }, 120);
}

async function loadCostSimulation() {
  const summary = document.getElementById('compare-cost-summary');
  if (summary) {
    summary.innerHTML = `
      <div class="metric-card"><div class="loading-overlay"><div class="spinner"></div></div></div>
      <div class="metric-card"><div class="loading-overlay"><div class="spinner"></div></div></div>
    `;
  }

  try {
    const params = new URLSearchParams(getCostInputs());
    const data = await api(`/api/compare/cost-simulator?${params}`);
    renderCostSummary(data);
    renderCostSimulatorChart('compare-cost-chart', data.models);
  } catch (err) {
    if (summary) {
      summary.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${err.message}</div></div>`;
    }
  }
}

function renderCostSummary(data) {
  const container = document.getElementById('compare-cost-summary');
  if (!container || !data?.models?.xgboost || !data?.models?.ft_transformer) return;

  const renderModelCard = (modelKey, model, accent) => {
    const isWinner = data.winner === modelKey;
    const ramPct = model.ram_budget_utilization * 100;
    return `
      <div class="result-card ${modelKey === 'xgboost' ? 'result-card--xgboost' : 'result-card--ft'}">
        <div class="result-card__model">${model.display_name}${isWinner ? ' • önerilen' : ''}</div>
        <div class="sim-score" style="color:${accent}">${model.fit_score.toFixed(1)}</div>
        <div class="text-muted" style="font-size:0.78rem;margin-top:4px">Uyum skoru / 100</div>
        <div class="sim-meta">
          <div class="sim-meta__item">
            <div class="sim-meta__label">Tahmini Gecikme</div>
            <div class="sim-meta__value">${model.estimated_latency_ms.toFixed(2)} ms</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Tahmini Peak RAM</div>
            <div class="sim-meta__value">${model.estimated_peak_ram_mb.toFixed(0)} MB</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Model Boyutu</div>
            <div class="sim-meta__value">${model.model_size_mb.toFixed(2)} MB</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">RAM Kullanımı</div>
            <div class="sim-meta__value">${ramPct.toFixed(1)}%</div>
          </div>
        </div>
        <div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Accuracy Cost</span>
            <span class="text-mono">${model.accuracy_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Latency Cost</span>
            <span class="text-mono">${model.latency_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Memory Cost</span>
            <span class="text-mono">${model.memory_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:0.78rem;color:var(--text-muted)">Composite Cost</span>
            <span class="text-mono">${model.combined_cost.toFixed(3)}</span>
          </div>
        </div>
        <div class="simulator-help" style="margin-top:var(--space-md)">${model.note}</div>
      </div>
    `;
  };

  container.innerHTML = [
    renderModelCard('xgboost', data.models.xgboost, '#fbbf24'),
    renderModelCard('ft_transformer', data.models.ft_transformer, '#a78bfa'),
  ].join('');
}

function renderCompareMetrics(metrics) {
  const container = document.getElementById('compare-metrics');
  if (!container) return;

  const xgb = metrics.xgboost;
  const ft = metrics.ft_transformer;

  container.innerHTML = `
    <div class="metric-card" style="border-top:3px solid #fbbf24">
      <div class="metric-card__label">XGBoost MAE</div>
      <div class="metric-card__value" style="color:#fbbf24">${formatFloat(xgb.mae, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #a78bfa">
      <div class="metric-card__label">FT-Transformer MAE</div>
      <div class="metric-card__value" style="color:#a78bfa">${formatFloat(ft.mae, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #fbbf24">
      <div class="metric-card__label">XGBoost R2</div>
      <div class="metric-card__value" style="color:#fbbf24">${formatFloat(xgb.r2, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #a78bfa">
      <div class="metric-card__label">FT-Transformer R2</div>
      <div class="metric-card__value" style="color:#a78bfa">${formatFloat(ft.r2, 6)}</div>
    </div>
  `;
}

async function loadCompareRows() {
  const params = new URLSearchParams({
    page: compareState.page,
    per_page: 50,
  });

  const resp = await api(`/api/compare/rows?${params}`);

  const cols = [
    { key: 'row_id', label: '#', format: 'int' },
    { key: 'engine_type', label: 'Engine' },
    { key: 'altitude', label: 'Alt (ft)', format: 'int' },
    { key: 'mach', label: 'Mach', format: 'float4' },
    { key: 'actual_specific_range', label: 'Actual', format: 'float6' },
    { key: 'xgboost_predicted', label: 'XGB Pred', format: 'float6' },
    { key: 'ft_transformer_predicted', label: 'FT Pred', format: 'float6' },
    { key: 'xgboost_absolute_error', label: 'XGB Error', format: 'float6' },
    { key: 'ft_transformer_absolute_error', label: 'FT Error', format: 'float6' },
  ];

  const maxErr = resp.rows.length > 0
    ? Math.max(...resp.rows.map(r => Math.max(r.xgboost_absolute_error || 0, r.ft_transformer_absolute_error || 0)))
    : 0.01;

  renderDataTable('compare-table', resp.rows, cols, {
    errorColumn: 'xgboost_absolute_error',
    maxErrorValue: maxErr,
    onRowClick: 'selectCompareRow',
  });

  window._compareRows = resp.rows;
  window._compareCols = cols;

  renderPagination('compare-pagination', resp, function goToPage(p) {
    compareState.page = p;
    loadCompareRows();
  });
}

function selectCompareRow(idx) {
  if (window._compareRows && window._compareCols) {
    renderRowDetail('compare-row-detail', window._compareRows[idx], window._compareCols);
  }
}

// ---------------------------------------------------------------------------
// Tab 3: Prediction (Tekil Tahmin)
// ---------------------------------------------------------------------------

async function loadPredictionTab() {
  window._predictLoaded = true;
  try {
    const scenarios = await api('/api/scenarios');
    const sel = document.getElementById('predict-scenario');
    if (sel) {
      sel.innerHTML = '';
      scenarios.forEach((s, i) => sel.add(new Option(s.name, i)));
    }
    window._scenarios = scenarios;
    applyScenario();
  } catch (err) {
    showToast('Senaryo yüklenemedi: ' + err.message, 'error');
  }
}

function applyScenario() {
  const idx = parseInt(document.getElementById('predict-scenario')?.value || '0');
  const s = window._scenarios?.[idx];
  if (!s) return;
  document.getElementById('input-altitude').value = s.altitude;
  document.getElementById('input-gross-weight').value = s.gross_weight;
  document.getElementById('input-drag-index').value = s.drag_index;
  document.getElementById('input-mach').value = s.mach;
  document.getElementById('input-fuel-flow').value = s.fuel_flow;
  document.getElementById('input-engine-type').value = s.engine_type;
}

async function runPrediction() {
  const body = {
    altitude: parseFloat(document.getElementById('input-altitude').value),
    gross_weight: parseFloat(document.getElementById('input-gross-weight').value),
    drag_index: parseFloat(document.getElementById('input-drag-index').value),
    mach: parseFloat(document.getElementById('input-mach').value),
    fuel_flow: parseFloat(document.getElementById('input-fuel-flow').value),
    engine_type: document.getElementById('input-engine-type').value,
  };

  const resultContainer = document.getElementById('predict-results');
  resultContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Hesaplaniyor...</span></div>';

  try {
    const result = await api('/api/predict', { method: 'POST', body: JSON.stringify(body) });

    let html = '';

    // ---- Input Summary ----
    html += `<div class="card mb-lg">
      <div class="card__header"><div class="card__title">Girdi Parametreleri</div></div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
        <div class="param-chip"><span class="param-chip__label">Altitude</span><span class="param-chip__value">${body.altitude.toLocaleString('tr-TR')} ft</span></div>
        <div class="param-chip"><span class="param-chip__label">Mach</span><span class="param-chip__value">${body.mach}</span></div>
        <div class="param-chip"><span class="param-chip__label">Gross Weight</span><span class="param-chip__value">${body.gross_weight.toLocaleString('tr-TR')} lb</span></div>
        <div class="param-chip"><span class="param-chip__label">Drag Index</span><span class="param-chip__value">${body.drag_index}</span></div>
        <div class="param-chip"><span class="param-chip__label">Fuel Flow</span><span class="param-chip__value">${body.fuel_flow.toLocaleString('tr-TR')} lb/h</span></div>
        <div class="param-chip"><span class="param-chip__label">Engine Type</span><span class="param-chip__value">${body.engine_type}</span></div>
      </div>
    </div>`;

    const hasExact = result.exact_match != null;
    const actual = hasExact ? result.exact_match.actual : null;

    // ---- Prediction Results ----
    if (hasExact) {
      // === EXACT MATCH FOUND ===
      html += `<div class="card mb-lg" style="border:1px solid rgba(16,185,129,0.3);background:rgba(16,185,129,0.05)">
        <div class="card__header">
          <div class="card__title" style="color:var(--success)">Bire Bir Eslesen Gercek Satir Bulundu</div>
        </div>
        <div style="text-align:center;margin-bottom:var(--space-md)">
          <div style="font-size:0.8rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">Gercek Specific Range</div>
          <div style="font-family:var(--font-mono);font-size:2.2rem;font-weight:700;color:var(--success)">${actual.toFixed(6)}</div>
        </div>
      </div>`;

      // Model predictions compared to actual
      html += '<div class="grid-2 mb-lg">';

      if (result.xgboost != null) {
        const xgbErr = Math.abs(result.xgboost - actual);
        const xgbPct = actual !== 0 ? ((xgbErr / Math.abs(actual)) * 100).toFixed(4) : '-';
        html += `<div class="result-card result-card--xgboost">
          <div class="result-card__model">XGBoost Tahmini</div>
          <div class="result-card__value">${result.xgboost.toFixed(6)}</div>
          <div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Mutlak Hata (AE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:${xgbErr < 0.001 ? 'var(--success)' : xgbErr < 0.005 ? 'var(--warning)' : 'var(--danger)'}">${xgbErr.toFixed(6)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Yuzde Hata (APE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${xgbPct}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:0.8rem;color:var(--text-muted)">Fark (Signed)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${(result.xgboost - actual) >= 0 ? '+' : ''}${(result.xgboost - actual).toFixed(6)}</span>
            </div>
          </div>
        </div>`;
      }

      if (result.ft_transformer != null) {
        const ftErr = Math.abs(result.ft_transformer - actual);
        const ftPct = actual !== 0 ? ((ftErr / Math.abs(actual)) * 100).toFixed(4) : '-';
        html += `<div class="result-card result-card--ft">
          <div class="result-card__model">FT-Transformer Tahmini</div>
          <div class="result-card__value">${result.ft_transformer.toFixed(6)}</div>
          <div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Mutlak Hata (AE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:${ftErr < 0.001 ? 'var(--success)' : ftErr < 0.005 ? 'var(--warning)' : 'var(--danger)'}">${ftErr.toFixed(6)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Yuzde Hata (APE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${ftPct}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:0.8rem;color:var(--text-muted)">Fark (Signed)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${(result.ft_transformer - actual) >= 0 ? '+' : ''}${(result.ft_transformer - actual).toFixed(6)}</span>
            </div>
          </div>
        </div>`;
      }

      html += '</div>';

      // Winner comparison if both models available
      if (result.xgboost != null && result.ft_transformer != null) {
        const xgbErr = Math.abs(result.xgboost - actual);
        const ftErr = Math.abs(result.ft_transformer - actual);
        const winner = xgbErr < ftErr ? 'XGBoost' : ftErr < xgbErr ? 'FT-Transformer' : 'Esit';
        const winnerColor = xgbErr < ftErr ? '#fbbf24' : ftErr < xgbErr ? '#a78bfa' : 'var(--text-secondary)';
        html += `<div class="card mb-lg" style="text-align:center;padding:var(--space-md)">
          <span style="font-size:0.8rem;color:var(--text-muted)">Bu satir icin daha iyi model:</span>
          <span style="font-weight:700;color:${winnerColor};margin-left:8px;font-size:1rem">${winner}</span>
          <span style="font-size:0.8rem;color:var(--text-dim);margin-left:8px">(hata farki: ${Math.abs(xgbErr - ftErr).toFixed(6)})</span>
        </div>`;
      }

    } else {
      // === NO EXACT MATCH ===
      const wAvg = result.weighted_avg_sr;

      html += `<div class="card mb-lg" style="border:1px solid rgba(245,158,11,0.3);background:rgba(245,158,11,0.05)">
        <div class="card__header">
          <div class="card__title" style="color:var(--warning)">Bire Bir Eslesen Satir Bulunamadi</div>
        </div>
        <p style="font-size:0.85rem;color:var(--text-muted);margin:0">Bu girdi kombinasyonu veri setinde yok. Asagida en yakin satirlarin mesafe-agirlikli ortalamasiyla karsilastirma yapiliyor.</p>
      </div>`;

      // Weighted average reference
      if (wAvg != null) {
        html += `<div class="card mb-lg" style="border:1px solid rgba(6,182,212,0.3);background:rgba(6,182,212,0.05)">
          <div class="card__header">
            <div class="card__title" style="color:var(--info)">Tahmini Gercek Deger (Agirlikli Ortalama)</div>
          </div>
          <div style="text-align:center;margin-bottom:var(--space-sm)">
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">En yakin satirlarin mesafe-agirlikli SR ortalamasi</div>
            <div style="font-family:var(--font-mono);font-size:2rem;font-weight:700;color:var(--info)">${wAvg.toFixed(6)}</div>
          </div>
        </div>`;
      }

      html += '<div class="grid-2 mb-lg">';

      if (result.xgboost != null) {
        html += `<div class="result-card result-card--xgboost">
          <div class="result-card__model">XGBoost Tahmini</div>
          <div class="result-card__value">${result.xgboost.toFixed(6)}</div>`;

        if (wAvg != null) {
          const xErr = Math.abs(result.xgboost - wAvg);
          const xPct = wAvg !== 0 ? ((xErr / Math.abs(wAvg)) * 100).toFixed(4) : '-';
          html += `<div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Agirlikli Ort. Farki (AE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:${xErr < 0.001 ? 'var(--success)' : xErr < 0.005 ? 'var(--warning)' : 'var(--danger)'}">${xErr.toFixed(6)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Yuzde Fark (APE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${xPct}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:0.8rem;color:var(--text-muted)">Fark (Signed)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${(result.xgboost - wAvg) >= 0 ? '+' : ''}${(result.xgboost - wAvg).toFixed(6)}</span>
            </div>
          </div>`;
        }

        html += '</div>';
      }

      if (result.ft_transformer != null) {
        html += `<div class="result-card result-card--ft">
          <div class="result-card__model">FT-Transformer Tahmini</div>
          <div class="result-card__value">${result.ft_transformer.toFixed(6)}</div>`;

        if (wAvg != null) {
          const fErr = Math.abs(result.ft_transformer - wAvg);
          const fPct = wAvg !== 0 ? ((fErr / Math.abs(wAvg)) * 100).toFixed(4) : '-';
          html += `<div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Agirlikli Ort. Farki (AE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:${fErr < 0.001 ? 'var(--success)' : fErr < 0.005 ? 'var(--warning)' : 'var(--danger)'}">${fErr.toFixed(6)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-size:0.8rem;color:var(--text-muted)">Yuzde Fark (APE)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${fPct}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:0.8rem;color:var(--text-muted)">Fark (Signed)</span>
              <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${(result.ft_transformer - wAvg) >= 0 ? '+' : ''}${(result.ft_transformer - wAvg).toFixed(6)}</span>
            </div>
          </div>`;
        }

        html += '</div>';
      }

      html += '</div>';

      // Show nearest rows table
      if (result.nearest_rows && result.nearest_rows.length > 0) {
        html += `<div class="card mb-lg">
          <div class="card__header"><div class="card__title">En Yakin Gercek Satirlar</div></div>
          <p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:var(--space-md)">
            Asagidaki satirlarin mesafeye ters oranli agirlikli ortalamasiyla yukaridaki kiyaslama yapildi.
          </p>`;

        const nearCols = [
          { key: 'engine_type', label: 'Engine' },
          { key: 'altitude', label: 'Alt (ft)', format: 'int' },
          { key: 'gross_weight', label: 'GW (lb)', format: 'int' },
          { key: 'drag_index', label: 'DI', format: 'int' },
          { key: 'mach', label: 'Mach', format: 'float4' },
          { key: 'fuel_flow', label: 'FF', format: 'int' },
          { key: 'specific_range', label: 'SR', format: 'float6' },
          { key: 'distance', label: 'Mesafe', format: 'float4' },
        ];
        html += '<div id="predict-nearest-inner"></div></div>';
        resultContainer.innerHTML = html;
        renderDataTable('predict-nearest-inner', result.nearest_rows, nearCols);
        showToast('Tahmin tamamlandi', 'success');
        return;
      }
    }

    resultContainer.innerHTML = html;
    showToast('Tahmin tamamlandi', 'success');
  } catch (err) {
    resultContainer.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${err.message}</div></div>`;
    showToast('Tahmin hatasi: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Tab 4: Nomogram
// ---------------------------------------------------------------------------

async function loadNomogramTab() {
  window._nomogramLoaded = true;
  try {
    const engineTypes = await api('/api/reference/engine-types');
    const sel = document.getElementById('nomo-engine');
    if (sel) {
      sel.innerHTML = '';
      engineTypes.forEach(et => sel.add(new Option(et, et)));
    }
    // Cascade: load altitudes for first engine type
    await loadNomoAltitudes();
  } catch (err) {
    showToast('Nomogram yuklenemedi: ' + err.message, 'warning');
  }
}

async function loadNomoAltitudes() {
  const eng = document.getElementById('nomo-engine')?.value;
  if (!eng) return;
  try {
    const altitudes = await api(`/api/reference/altitudes?engine_type=${encodeURIComponent(eng)}`);
    const sel = document.getElementById('nomo-altitude');
    if (sel) {
      sel.innerHTML = '';
      altitudes.forEach(a => sel.add(new Option(`${Math.round(a)} ft`, a)));
    }
    // Cascade: load weights for first altitude
    await loadNomoWeights();
  } catch (err) {
    console.warn('Altitude load failed:', err);
  }
}

async function loadNomoWeights() {
  const eng = document.getElementById('nomo-engine')?.value;
  const alt = document.getElementById('nomo-altitude')?.value;
  if (!eng || !alt) return;
  try {
    const weights = await api(`/api/reference/gross-weights?engine_type=${encodeURIComponent(eng)}&altitude=${alt}`);
    const sel = document.getElementById('nomo-gross-weight');
    if (sel) {
      sel.innerHTML = '';
      weights.forEach(w => sel.add(new Option(`${Math.round(w)} lb`, w)));
    }
  } catch (err) {
    console.warn('Weight load failed:', err);
  }
}

async function generateNomogram() {
  const altVal = document.getElementById('nomo-altitude')?.value;
  const gwVal = document.getElementById('nomo-gross-weight')?.value;
  if (!altVal || !gwVal) {
    showToast('Lutfen tum parametreleri secin', 'warning');
    return;
  }

  const body = {
    model: document.getElementById('nomo-model').value,
    engine_type: document.getElementById('nomo-engine').value,
    altitude: parseFloat(altVal),
    gross_weight: parseFloat(gwVal),
  };

  const preview = document.getElementById('nomo-preview');
  preview.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Nomogram üretiliyor...</span></div>';

  try {
    const result = await api('/api/nomogram', { method: 'POST', body: JSON.stringify(body) });
    preview.innerHTML = `<img src="${result.plot_url}" alt="Nomogram" 
      style="width:100%;border-radius:var(--radius-md);cursor:zoom-in" 
      onclick="window.open(this.src, '_blank')">
      <p class="text-muted mt-sm text-center" style="font-size:0.8rem">Resme tıkla: tam boyut aç</p>`;
    showToast('Nomogram oluşturuldu', 'success');
  } catch (err) {
    preview.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${err.message}</div></div>`;
    showToast('Nomogram hatası: ' + err.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Tab 5: Setup
// ---------------------------------------------------------------------------

async function loadSetupTab() {
  window._setupLoaded = true;
  try {
    const commands = await api('/api/setup/commands');
    const container = document.getElementById('setup-steps');
    if (!container) return;

    container.innerHTML = commands.map((cmd, i) => `
      <div class="step" id="step-${cmd.id}">
        <div class="step__indicator">${i + 1}</div>
        <div class="step__content">
          <div class="step__title">${cmd.label}</div>
          <div class="step__meta">Tahmini süre: ${cmd.eta}</div>
          <div class="step__command">${cmd.command}</div>
          <div class="step__meta mt-sm">Üretecekleri: ${cmd.artifacts.join(', ')}</div>
        </div>
        <button class="btn btn--secondary btn--sm" onclick="runSetupCommand('${cmd.id}', '${cmd.command.replace(/'/g, "\\'")}', '${cmd.label}')">
          Çalıştır
        </button>
      </div>
    `).join('');

    window._setupCommands = commands;
  } catch (err) {
    showToast('Setup yüklenemedi: ' + err.message, 'error');
  }
}

function runSetupCommand(id, command, label) {
  const stepEl = document.getElementById(`step-${id}`);
  if (stepEl) {
    stepEl.className = 'step step--running';
    stepEl.querySelector('.step__indicator').textContent = '⟳';
  }

  const logArea = document.getElementById('setup-log');
  logArea.textContent += `\n$ ${command}\n`;

  fetch('/api/setup/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command }),
  }).then(async resp => {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split('\n').filter(l => l.startsWith('data: '));

      for (const line of lines) {
        try {
          const data = JSON.parse(line.replace('data: ', ''));
          if (data.type === 'output') {
            logArea.textContent += data.text + '\n';
            logArea.scrollTop = logArea.scrollHeight;
          } else if (data.type === 'done') {
            if (stepEl) {
              stepEl.className = data.exit_code === 0 ? 'step step--done' : 'step step--error';
              stepEl.querySelector('.step__indicator').textContent = data.exit_code === 0 ? '✓' : '✗';
            }
            const status = data.exit_code === 0 ? 'success' : 'error';
            showToast(`${label}: ${data.exit_code === 0 ? 'tamamlandı' : 'başarısız'}`, status);
            loadStatus();
          } else if (data.type === 'error') {
            logArea.textContent += `ERROR: ${data.text}\n`;
            if (stepEl) {
              stepEl.className = 'step step--error';
              stepEl.querySelector('.step__indicator').textContent = '✗';
            }
          }
        } catch (e) { /* skip unparseable */ }
      }
    }
  }).catch(err => {
    logArea.textContent += `\nFATAL: ${err.message}\n`;
    if (stepEl) {
      stepEl.className = 'step step--error';
      stepEl.querySelector('.step__indicator').textContent = '✗';
    }
  });
}

async function runQuickstart() {
  if (!window._setupCommands) return;
  const logArea = document.getElementById('setup-log');
  logArea.textContent = '';

  for (const cmd of window._setupCommands) {
    const stepEl = document.getElementById(`step-${cmd.id}`);
    if (stepEl) {
      stepEl.className = 'step step--running';
      stepEl.querySelector('.step__indicator').textContent = '⟳';
    }

    logArea.textContent += `\n$ ${cmd.command}\n`;

    try {
      const resp = await fetch('/api/setup/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd.command }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let exitCode = -1;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const lines = text.split('\n').filter(l => l.startsWith('data: '));
        for (const line of lines) {
          try {
            const data = JSON.parse(line.replace('data: ', ''));
            if (data.type === 'output') {
              logArea.textContent += data.text + '\n';
              logArea.scrollTop = logArea.scrollHeight;
            } else if (data.type === 'done') {
              exitCode = data.exit_code;
            }
          } catch (e) { /* skip */ }
        }
      }

      if (stepEl) {
        stepEl.className = exitCode === 0 ? 'step step--done' : 'step step--error';
        stepEl.querySelector('.step__indicator').textContent = exitCode === 0 ? '✓' : '✗';
      }

      if (exitCode !== 0) {
        showToast(`${cmd.label} başarısız oldu, quickstart durduruluyor`, 'error');
        break;
      }
    } catch (err) {
      showToast(`${cmd.label} hatası: ${err.message}`, 'error');
      break;
    }
  }

  showToast('Quickstart tamamlandı', 'success');
  loadStatus();
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatFloat(val, decimals = 6) {
  if (val == null || isNaN(val)) return '-';
  return Number(val).toFixed(decimals);
}

function formatInt(val) {
  if (val == null || isNaN(val)) return '-';
  return Math.round(val).toLocaleString('tr-TR');
}

function metricCardHTML(label, value, delta = '', colorClass = '', desc = '') {
  const valClass = colorClass ? ` metric-card__value--${colorClass}` : '';
  const deltaHtml = delta
    ? `<div class="metric-card__delta">${delta}</div>`
    : '';
  const descHtml = desc
    ? `<div class="metric-card__desc">${desc}</div>`
    : '';
  return `<div class="metric-card" ${desc ? `title="${desc}"` : ''}>
    <div class="metric-card__label">${label}</div>
    <div class="metric-card__value${valClass}">${value}</div>
    ${deltaHtml}
    ${descHtml}
  </div>`;
}

const METRIC_DESC = {
  rows: 'Rapordaki toplam veri satiri sayisi',
  mae: 'Mean Absolute Error: Tahmin ile gercek deger arasindaki mutlak farklarin ortalamasi. Dusuk = iyi.',
  rmse: 'Root Mean Squared Error: Buyuk hatalara karsi hassas. MAE\'den buyukse ucta hatalar var demektir.',
  mape: 'Mean Absolute Percentage Error: Yuzde cinsinden ortalama hata. Olcek bagimsiz karsilastirma saglar.',
  r2: 'R-Squared (Belirtme Katsayisi): 1.0 = mukemmel uyum. Modelin verideki varyansin yuzde kacini acikladigini gosterir.',
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadStatus();

  // Load default tab
  loadOverview();
});
