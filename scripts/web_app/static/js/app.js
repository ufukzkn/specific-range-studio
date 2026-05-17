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
    const error = new Error(err.error || `API error ${resp.status}`);
    Object.assign(error, err);
    throw error;
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
// SVG Plane Loader
// ---------------------------------------------------------------------------

function planeLoaderSVG() {
  return `
    <svg class="spinner svg-calLoader" xmlns="http://www.w3.org/2000/svg" viewBox="-18 -18 266 266" role="img" aria-label="Yükleniyor">
      <path class="cal-loader__path"
        d="M86.429 40c63.616-20.04 101.511 25.08 107.265 61.93 6.487 41.54-18.593 76.99-50.6 87.643-59.46 19.791-101.262-23.577-107.142-62.616C29.398 83.441 59.945 48.343 86.43 40z"
        fill="none" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"
        stroke-dasharray="10 10 10 10 10 10 10 432" stroke-dashoffset="77" />
      <path class="cal-loader__plane"
        d="M141.493 37.93c-1.087-.927-2.942-2.002-4.32-2.501-2.259-.824-3.252-.955-9.293-1.172-4.017-.146-5.197-.23-5.47-.37-.766-.407-1.526-1.448-7.114-9.773-4.8-7.145-5.344-7.914-6.327-8.976-1.214-1.306-1.396-1.378-3.79-1.473-1.036-.04-2-.043-2.153-.002-.353.1-.87.586-1 .952-.139.399-.076.71.431 2.22.241.72 1.029 3.386 1.742 5.918 1.644 5.844 2.378 8.343 2.863 9.705.206.601.33 1.1.275 1.125-.24.097-10.56 1.066-11.014 1.032a3.532 3.532 0 0 1-1.002-.276l-.487-.246-2.044-2.613c-2.234-2.87-2.228-2.864-3.35-3.309-.717-.287-2.82-.386-3.276-.163-.457.237-.727.644-.737 1.152-.018.39.167.805 1.916 4.373 1.06 2.166 1.964 4.083 1.998 4.27.04.179.004.521-.076.75-.093.228-1.109 2.064-2.269 4.088-1.921 3.34-2.11 3.711-2.123 4.107-.008.25.061.557.168.725.328.512.72.644 1.966.676 1.32.029 2.352-.236 3.05-.762.222-.171 1.275-1.313 2.412-2.611 1.918-2.185 2.048-2.32 2.45-2.505.241-.111.601-.232.82-.271.267-.058 2.213.201 5.912.8 3.036.48 5.525.894 5.518.914 0 .026-.121.306-.27.638-.54 1.198-1.515 3.842-3.35 9.021-1.029 2.913-2.107 5.897-2.4 6.62-.703 1.748-.725 1.833-.594 2.286.137.46.45.833.872 1.012.41.177 3.823.24 4.37.085.852-.25 1.44-.688 2.312-1.724 1.166-1.39 3.169-3.948 6.771-8.661 5.8-7.583 6.561-8.49 7.387-8.702.233-.065 2.828-.056 5.784.011 5.827.138 6.64.09 8.62-.5 2.24-.67 4.035-1.65 5.517-3.016 1.136-1.054 1.135-1.014.207-1.962-.357-.38-.767-.777-.902-.893z" />
    </svg>
  `.trim();
}

function upgradePlaneLoaders(root = document) {
  const targets = [];
  if (root instanceof Element && root.matches('.spinner:not(.svg-calLoader)')) {
    targets.push(root);
  }
  if (root.querySelectorAll) {
    targets.push(...root.querySelectorAll('.spinner:not(.svg-calLoader)'));
  }

  targets.forEach((el) => {
    const template = document.createElement('template');
    template.innerHTML = planeLoaderSVG();
    el.replaceWith(template.content.firstElementChild);
  });
}

function initPlaneLoaderObserver() {
  upgradePlaneLoaders(document);
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) upgradePlaneLoaders(node);
      });
    });
  });
  observer.observe(document.body, { childList: true, subtree: true });
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
      if (target === 'cost' && !window._costLoaded) loadCostTab();
      if (target === 'predict' && !window._predictLoaded) loadPredictionTab();
      if (target === 'nomogram' && !window._nomogramLoaded) loadNomogramTab();
      if (target === 'dataset' && !window._datasetToolsLoaded) loadDatasetToolsTab();
      if (target === 'setup' && !window._setupLoaded) loadSetupTab();
      if (target === 'info' && !window._infoLoaded) loadInfoTab();
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
      { label: 'Interp', ok: status.interpolation_ready },
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
      renderMetricComparisonChart('compare-metric-chart', metrics);
    }
    await loadToleranceCurveChart();

    // Load rows
    await loadCompareRows();

  } catch (err) {
    showToast('Karsilastirma yuklenemedi: ' + err.message, 'error');
  }
}

async function loadCostTab() {
  window._costLoaded = true;
  updateCostControlLabels();
  await loadCostSimulation();
  await loadLatestPsoSummary();
}

function getCostInputs() {
  return {
    accuracy_weight: parseFloat(document.getElementById('cost-accuracy')?.value || '45'),
    latency_weight: parseFloat(document.getElementById('cost-latency')?.value || '25'),
    memory_weight: parseFloat(document.getElementById('cost-memory')?.value || '20'),
    cpu_weight: parseFloat(document.getElementById('cost-cpu')?.value || '10'),
    cpu_speed_factor: parseFloat(document.getElementById('cost-cpu-speed')?.value || '1'),
    ram_budget_gb: parseFloat(document.getElementById('cost-ram-budget')?.value || '1'),
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
  setText('cost-cpu-value', `${Math.round(inputs.cpu_weight)}%`);
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
    renderBenchmarkStatus(data.benchmark);
    renderCostSummary(data);
    renderCostSimulatorChart('compare-cost-chart', data.models);
  } catch (err) {
    renderBenchmarkStatus(null);
    if (summary) {
      const command = err.command || 'python scripts/run_pi_benchmark.py --models interpolation,xgboost,ft_transformer --sample-size 200 --warmup 20 --repetitions 200 --device cpu';
      summary.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <div class="empty-state__text text-warning">${escapeHTML(err.message || 'Gerçek benchmark ölçümü bulunamadı.')}</div>
          <div class="step__command mt-md">${escapeHTML(command)}</div>
          <button class="btn btn--primary mt-md" onclick="runPiBenchmarkFromCostTab()">Bu cihazda benchmark çalıştır</button>
        </div>
      `;
    }
    const chart = document.getElementById('compare-cost-chart');
    if (chart) chart.innerHTML = '';
  }
}

async function loadLatestPsoSummary() {
  const container = document.getElementById('pso-latest-summary');
  if (!container) return;
  container.innerHTML = '<div class="metric-card"><div class="loading-overlay"><div class="spinner"></div></div></div>';
  try {
    const data = await api('/api/pso/latest');
    if (!data.available) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <div class="empty-state__text text-warning">${escapeHTML(data.message || 'Henüz PSO sonucu yok.')}</div>
          <div class="step__command mt-md">${escapeHTML(data.command || 'python scripts/run_pso.py --dataset data/processed/combined_specific_range.csv --device cpu --population 4 --iterations 3')}</div>
        </div>
      `;
      return;
    }
    const best = data.best_row || {};
    const params = data.best_params || {};
    const objective = data.objective || {};
    const weights = objective.weights || {};
    const references = objective.references || {};
    const fmt = (value, digits = 4) => value === null || value === undefined || Number.isNaN(Number(value)) ? '-' : Number(value).toFixed(digits);
    container.innerHTML = `
      <div class="result-card result-card--ft">
        <div class="result-card__model">En iyi PSO FT-Transformer adayı</div>
        <div class="sim-score">${fmt(data.best_score, 3)}</div>
        <div class="text-muted" style="font-size:0.78rem;margin-top:4px">Toplam maliyet skoru; düşük daha iyi</div>
        <div class="sim-meta">
          <div class="sim-meta__item"><div class="sim-meta__label">RMSE</div><div class="sim-meta__value">${fmt(best.rmse, 6)}</div></div>
          <div class="sim-meta__item"><div class="sim-meta__label">MAE</div><div class="sim-meta__value">${fmt(best.mae, 6)}</div></div>
          <div class="sim-meta__item"><div class="sim-meta__label">p95 Gecikme</div><div class="sim-meta__value">${fmt(best.latency_ms, 2)} ms</div></div>
          <div class="sim-meta__item"><div class="sim-meta__label">Model Boyutu</div><div class="sim-meta__value">${fmt(best.model_size_mb, 3)} MB</div></div>
          <div class="sim-meta__item"><div class="sim-meta__label">Aday Sayısı</div><div class="sim-meta__value">${escapeHTML(data.history_count || '-')}</div></div>
          <div class="sim-meta__item"><div class="sim-meta__label">Pareto Benzeri Aday</div><div class="sim-meta__value">${escapeHTML((data.pareto_front || []).length || '-')}</div></div>
        </div>
      </div>
      <div class="result-card">
        <div class="result-card__model">Seçilen hiperparametreler</div>
        <div class="step__command">
          layers=${escapeHTML(params.n_layers ?? '-')} · heads=${escapeHTML(params.n_heads ?? '-')} · d_model=${escapeHTML(params.d_model ?? '-')} · d_ff=${escapeHTML(params.d_ff ?? '-')} · dropout=${fmt(params.dropout, 3)} · lr=${fmt(params.learning_rate, 6)}
        </div>
        <div class="simulator-help" style="margin-top:var(--space-md)">
          Ağırlıklar: RMSE ${fmt(weights.rmse, 2)}, latency ${fmt(weights.latency, 2)}, size ${fmt(weights.size, 2)}.
          Referanslar: RMSE ${fmt(references.rmse_ref, 4)}, latency ${fmt(references.latency_ref_ms, 1)} ms, size ${fmt(references.size_ref_mb, 2)} MB.
        </div>
        <div class="simulator-help" style="margin-top:var(--space-sm)">Çıktı: ${escapeHTML(data.path || '')}</div>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div class="empty-state__text text-danger">PSO sonucu okunamadı: ${escapeHTML(err.message)}</div></div>`;
  }
}

async function runPiBenchmarkFromCostTab() {
  const summary = document.getElementById('compare-cost-summary');
  if (summary) {
    summary.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1">
        <div class="loading-overlay"><div class="spinner"></div><span>Gerçek benchmark çalışıyor…</span></div>
        <pre class="readme-body" id="pi-benchmark-run-log" style="margin-top:var(--space-lg);text-align:left;max-height:260px"></pre>
      </div>
    `;
    upgradePlaneLoaders(summary);
  }
  const log = document.getElementById('pi-benchmark-run-log');
  try {
    const resp = await fetch('/api/benchmark/pi/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ models: 'interpolation,xgboost,ft_transformer', sample_size: 200, warmup: 20, repetitions: 200, device: 'cpu' }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `API error ${resp.status}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      const lines = text.split('\n').filter(l => l.startsWith('data: '));
      for (const line of lines) {
        const data = JSON.parse(line.replace('data: ', ''));
        if (data.type === 'output' && log) {
          log.textContent += data.text + '\n';
          log.scrollTop = log.scrollHeight;
        } else if (data.type === 'done') {
          showToast(`Pi benchmark ${data.exit_code === 0 ? 'tamamlandı' : 'başarısız'}`, data.exit_code === 0 ? 'success' : 'error');
        }
      }
    }
    await loadCostSimulation();
  } catch (err) {
    if (log) log.textContent += `\nFATAL: ${err.message}\n`;
    showToast('Pi benchmark çalıştırılamadı: ' + err.message, 'error');
  }
}

function renderBenchmarkStatus(benchmark) {
  const container = document.getElementById('pi-benchmark-status');
  if (!container) return;
  if (!benchmark) {
    container.innerHTML = `
      <div class="metric-card">
        <div class="metric-card__label">Benchmark Durumu</div>
        <div class="metric-card__value metric-card__value--warning">Ölçüm yok</div>
        <div class="metric-card__desc metric-card__desc--visible">Pi üzerinde benchmark komutunu çalıştırınca burada gerçek cihaz bilgileri görünür.</div>
      </div>
    `;
    return;
  }
  const system = benchmark.system || {};
  const config = benchmark.config || {};
  const created = benchmark.created_at ? new Date(benchmark.created_at).toLocaleString('tr-TR') : '-';
  container.innerHTML = `
    <div class="metric-card">
      <div class="metric-card__label">Son Ölçüm</div>
      <div class="metric-card__value">${escapeHTML(created)}</div>
      <div class="metric-card__desc metric-card__desc--visible">Run ID: ${escapeHTML(benchmark.run_id || '-')}</div>
    </div>
    <div class="metric-card">
      <div class="metric-card__label">Cihaz</div>
      <div class="metric-card__value">${escapeHTML(system.hostname || '-')}</div>
      <div class="metric-card__desc metric-card__desc--visible">${escapeHTML(system.machine || system.platform || '-')}</div>
    </div>
    <div class="metric-card">
      <div class="metric-card__label">Python / RAM</div>
      <div class="metric-card__value">${escapeHTML(system.python || '-')}</div>
      <div class="metric-card__desc metric-card__desc--visible">${system.total_ram_mb ? `${Number(system.total_ram_mb).toFixed(0)} MB toplam RAM` : 'RAM bilgisi yok'}</div>
    </div>
    <div class="metric-card">
      <div class="metric-card__label">Örnek / Tekrar</div>
      <div class="metric-card__value">${escapeHTML(config.sample_size || '-')} / ${escapeHTML(config.repetitions || '-')}</div>
      <div class="metric-card__desc metric-card__desc--visible">${escapeHTML(benchmark.path || '')}</div>
    </div>
  `;
}

function renderCostSummary(data) {
  const container = document.getElementById('compare-cost-summary');
  if (!container || !data?.models) return;

  const modelStyles = {
    interpolation: { cls: 'result-card--interp', color: '#06b6d4' },
    xgboost: { cls: 'result-card--xgboost', color: '#fbbf24' },
    ft_transformer: { cls: 'result-card--ft', color: '#a78bfa' },
  };

  const renderModelCard = (modelKey, model) => {
    const style = modelStyles[modelKey] || { cls: '', color: 'var(--accent)' };
    const isRuntimeWinner = data.runtime_winner === modelKey;
    const isLearnedWinner = data.learned_winner === modelKey;
    const ramPct = model.ram_budget_utilization * 100;
    const latencyLabel = Math.abs((data.hardware?.cpu_speed_factor || 1) - 1) > 0.001
      ? 'Senaryo p95 Gecikme'
      : 'Ölçülen p95 Gecikme';
    const latencyValue = Math.abs((data.hardware?.cpu_speed_factor || 1) - 1) > 0.001
      ? model.scenario_latency_p95_ms
      : model.measured_latency_p95_ms;
    return `
      <div class="result-card ${style.cls}">
        <div class="result-card__model">
          ${model.display_name}
          ${isRuntimeWinner ? ' • runtime lideri' : ''}
          ${isLearnedWinner ? ' • ML doğruluk/maliyet lideri' : ''}
        </div>
        <div class="sim-score" style="color:${style.color}">${model.fit_score.toFixed(1)}</div>
        <div class="text-muted" style="font-size:0.78rem;margin-top:4px">Uyum skoru / 100</div>
        <div class="sim-meta">
          <div class="sim-meta__item">
            <div class="sim-meta__label">${latencyLabel}</div>
            <div class="sim-meta__value">${latencyValue.toFixed(2)} ms</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Ölçülen p50 Gecikme</div>
            <div class="sim-meta__value">${(model.measured_latency_p50_ms || 0).toFixed(2)} ms</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Ölçülen Peak RSS</div>
            <div class="sim-meta__value">${model.measured_peak_rss_mb.toFixed(0)} MB</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Model Boyutu</div>
            <div class="sim-meta__value">${model.model_size_mb === null || model.model_size_mb === undefined ? '-' : model.model_size_mb.toFixed(2) + ' MB'}</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">RAM Kullanımı</div>
            <div class="sim-meta__value">${ramPct.toFixed(1)}%</div>
          </div>
          <div class="sim-meta__item">
            <div class="sim-meta__label">Ortalama CPU</div>
            <div class="sim-meta__value">${model.measured_cpu_avg_percent.toFixed(1)}%</div>
          </div>
        </div>
        <div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Accuracy Cost</span>
            <span class="text-mono">${model.accuracy_included && model.accuracy_component != null ? model.accuracy_component.toFixed(3) : 'Referans dışı'}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Latency Cost</span>
            <span class="text-mono">${model.latency_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">Memory Cost</span>
            <span class="text-mono">${model.memory_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:0.78rem;color:var(--text-muted)">CPU Cost</span>
            <span class="text-mono">${model.cpu_component.toFixed(3)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:0.78rem;color:var(--text-muted)">${model.accuracy_included ? 'Composite Cost' : 'Runtime Cost'}</span>
            <span class="text-mono">${model.combined_cost.toFixed(3)}</span>
          </div>
        </div>
        <div class="simulator-help" style="margin-top:var(--space-md)">${model.note}</div>
      </div>
    `;
  };

  const order = ['interpolation', 'xgboost', 'ft_transformer'];
  container.innerHTML = order
    .filter(key => data.models[key])
    .map(key => renderModelCard(key, data.models[key]))
    .join('');
}

function renderCompareMetrics(metrics) {
  const container = document.getElementById('compare-metrics');
  if (!container) return;

  const xgb = metrics.xgboost;
  const ft = metrics.ft_transformer;
  const interp = metrics.interpolation;

  container.innerHTML = `
    <div class="metric-card" style="border-top:3px solid #06b6d4">
      <div class="metric-card__label">Interpolasyon MAE</div>
      <div class="metric-card__value" style="color:#06b6d4">${formatFloat(interp?.mae, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #fbbf24">
      <div class="metric-card__label">XGBoost MAE</div>
      <div class="metric-card__value" style="color:#fbbf24">${formatFloat(xgb.mae, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #a78bfa">
      <div class="metric-card__label">FT-Transformer MAE</div>
      <div class="metric-card__value" style="color:#a78bfa">${formatFloat(ft.mae, 6)}</div>
    </div>
    <div class="metric-card" style="border-top:3px solid #06b6d4">
      <div class="metric-card__label">Interpolasyon R2</div>
      <div class="metric-card__value" style="color:#06b6d4">${formatFloat(interp?.r2, 6)}</div>
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

async function loadToleranceCurveChart() {
  const container = document.getElementById('compare-tolerance-curve');
  if (!container) return;
  container.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Eğri hazırlanıyor…</span></div>';

  try {
    const data = await api('/api/compare/tolerance-curve');
    renderToleranceSummary(data);
    if (window.Plotly && typeof renderToleranceCurve === 'function') {
      renderToleranceCurve('compare-tolerance-curve', data);
    } else {
      container.innerHTML = `
        <img
          src="/api/compare/tolerance-curve.svg?t=${Date.now()}"
          alt="Hata toleransı başarı eğrisi"
          style="width:100%;height:100%;min-height:340px;border-radius:var(--radius-md);display:block"
          onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'><div class=\\'empty-state__text text-warning\\'>Tolerans eğrisi görseli üretilemedi.</div></div>'"
        >
      `;
    }
  } catch (err) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${err.message}</div></div>`;
  }
}

function renderToleranceSummary(data) {
  const container = document.getElementById('compare-tolerance-summary');
  if (!container || !data?.summary) return;
  const s = data.summary;
  container.innerHTML = [
    metricCardHTML('XGB Median AE', formatFloat(s.xgboost_median_error, 6), '', 'warning', 'XGBoost mutlak hata medyanı. Düşük = iyi.'),
    metricCardHTML('FT Median AE', formatFloat(s.ft_transformer_median_error, 6), '', 'accent', 'FT-Transformer mutlak hata medyanı. Düşük = iyi.'),
    metricCardHTML('XGB P95 AE', formatFloat(s.xgboost_p95_error, 6), '', 'warning', 'Satırların %95’i bu mutlak hata eşiğinin altında.'),
    metricCardHTML('FT P95 AE', formatFloat(s.ft_transformer_p95_error, 6), '', 'accent', 'Satırların %95’i bu mutlak hata eşiğinin altında.'),
  ].join('');
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
    { key: 'interpolation_predicted', label: 'Interp Pred', format: 'float6' },
    { key: 'xgboost_predicted', label: 'XGB Pred', format: 'float6' },
    { key: 'ft_transformer_predicted', label: 'FT Pred', format: 'float6' },
    { key: 'interpolation_absolute_error', label: 'Interp Error', format: 'float6' },
    { key: 'xgboost_absolute_error', label: 'XGB Error', format: 'float6' },
    { key: 'ft_transformer_absolute_error', label: 'FT Error', format: 'float6' },
  ];

  const maxErr = resp.rows.length > 0
    ? Math.max(...resp.rows.map(r => Math.max(r.interpolation_absolute_error || 0, r.xgboost_absolute_error || 0, r.ft_transformer_absolute_error || 0)))
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

function predictionResultCardHTML({ key, label, value, reference, referenceLabel }) {
  if (value == null || isNaN(value)) return '';
  const classMap = {
    interpolation: 'result-card--interp',
    xgboost: 'result-card--xgboost',
    ft_transformer: 'result-card--ft',
  };
  let detailHtml = '';
  if (reference != null && !isNaN(reference)) {
    const err = Math.abs(value - reference);
    const pct = reference !== 0 ? ((err / Math.abs(reference)) * 100).toFixed(4) : '-';
    const signed = value - reference;
    detailHtml = `<div style="margin-top:var(--space-md);padding-top:var(--space-md);border-top:1px solid var(--glass-border)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:0.8rem;color:var(--text-muted)">${referenceLabel || 'Referans'} Farkı (AE)</span>
        <span style="font-family:var(--font-mono);font-weight:600;color:${err < 0.001 ? 'var(--success)' : err < 0.005 ? 'var(--warning)' : 'var(--danger)'}">${err.toFixed(6)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:0.8rem;color:var(--text-muted)">Yüzde Hata (APE)</span>
        <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${pct}%</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:0.8rem;color:var(--text-muted)">Fark (Signed)</span>
        <span style="font-family:var(--font-mono);font-weight:600;color:var(--text-secondary)">${signed >= 0 ? '+' : ''}${signed.toFixed(6)}</span>
      </div>
    </div>`;
  }
  return `<div class="result-card ${classMap[key] || ''}">
    <div class="result-card__model">${label}</div>
    <div class="result-card__value">${value.toFixed(6)}</div>
    ${detailHtml}
  </div>`;
}

function bestPredictionLabel(result, reference) {
  const candidates = [
    ['XGBoost', result.xgboost, '#fbbf24'],
    ['FT-Transformer', result.ft_transformer, '#a78bfa'],
  ].filter(([, value]) => value != null && !isNaN(value));
  if (reference == null || candidates.length < 2) return '';
  const ranked = candidates
    .map(([label, value, color]) => ({ label, value, color, error: Math.abs(value - reference) }))
    .sort((a, b) => a.error - b.error);
  return `<div class="card mb-lg" style="text-align:center;padding:var(--space-md)">
    <span style="font-size:0.8rem;color:var(--text-muted)">Bu referansa en yakın ML yöntemi:</span>
    <span style="font-weight:700;color:${ranked[0].color};margin-left:8px;font-size:1rem">${ranked[0].label}</span>
    <span style="font-size:0.8rem;color:var(--text-dim);margin-left:8px">(hata: ${ranked[0].error.toFixed(6)})</span>
  </div>`;
}

async function runPrediction() {
  const body = {
    altitude: parseFloat(document.getElementById('input-altitude').value),
    gross_weight: parseFloat(document.getElementById('input-gross-weight').value),
    drag_index: parseFloat(document.getElementById('input-drag-index').value),
    mach: parseFloat(document.getElementById('input-mach').value),
    fuel_flow: parseFloat(document.getElementById('input-fuel-flow').value),
    engine_type: document.getElementById('input-engine-type').value,
    methods: [document.getElementById('predict-method')?.value || 'all'],
    interpolation_method: document.getElementById('predict-interp-method')?.value || 'spline',
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
      html += '<div class="result-grid mb-lg">';

      if (result.interpolation != null) {
        html += predictionResultCardHTML({
          key: 'interpolation',
          label: `Interpolasyon (${result.interpolation_method || 'Spline'})`,
          value: result.interpolation,
          reference: actual,
          referenceLabel: 'Gerçek',
        });
      }

      if (result.xgboost != null) {
        html += predictionResultCardHTML({ key: 'xgboost', label: 'XGBoost Tahmini', value: result.xgboost, reference: actual, referenceLabel: 'Gerçek' });
      }

      if (result.ft_transformer != null) {
        html += predictionResultCardHTML({ key: 'ft_transformer', label: 'FT-Transformer Tahmini', value: result.ft_transformer, reference: actual, referenceLabel: 'Gerçek' });
      }

      html += '</div>';

      html += bestPredictionLabel(result, actual);

    } else {
      // === NO EXACT MATCH ===
      const wAvg = result.weighted_avg_sr;
      const interpolatedReference = result.interpolation != null ? result.interpolation : wAvg;

      html += `<div class="card mb-lg" style="border:1px solid rgba(245,158,11,0.3);background:rgba(245,158,11,0.05)">
        <div class="card__header">
          <div class="card__title" style="color:var(--warning)">Bire Bir Eslesen Satir Bulunamadi</div>
        </div>
        <p style="font-size:0.85rem;color:var(--text-muted);margin:0">Bu girdi kombinasyonu veri setinde yok. Asagida en yakin satirlarin mesafe-agirlikli ortalamasiyla karsilastirma yapiliyor.</p>
      </div>`;

      // Interpolation is the preferred estimated real value for custom inputs.
      if (interpolatedReference != null) {
        html += `<div class="card mb-lg" style="border:1px solid rgba(6,182,212,0.3);background:rgba(6,182,212,0.05)">
          <div class="card__header">
            <div class="card__title" style="color:var(--info)">Tahmini Gerçek Değer (Interpolasyon Referansı)</div>
          </div>
          <div style="text-align:center;margin-bottom:var(--space-sm)">
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">
              ${result.interpolation != null ? `${result.interpolation_method || 'Cubic Spline'} ile hesaplanan specific range` : 'Interpolasyon yoksa en yakın satırların mesafe ağırlıklı SR ortalaması'}
            </div>
            <div style="font-family:var(--font-mono);font-size:2rem;font-weight:700;color:var(--info)">${interpolatedReference.toFixed(6)}</div>
          </div>
        </div>`;
      }

      html += '<div class="result-grid mb-lg">';

      if (result.interpolation != null) {
        html += predictionResultCardHTML({
          key: 'interpolation',
          label: `Interpolasyon (${result.interpolation_method || 'Spline'})`,
          value: result.interpolation,
          reference: null,
          referenceLabel: 'Referans',
        });
      }

      if (result.xgboost != null) {
        html += predictionResultCardHTML({ key: 'xgboost', label: 'XGBoost Tahmini', value: result.xgboost, reference: interpolatedReference, referenceLabel: 'Interpolasyon Ref.' });
      }

      if (result.ft_transformer != null) {
        html += predictionResultCardHTML({ key: 'ft_transformer', label: 'FT-Transformer Tahmini', value: result.ft_transformer, reference: interpolatedReference, referenceLabel: 'Interpolasyon Ref.' });
      }

      html += '</div>';
      html += bestPredictionLabel(
        {
          xgboost: result.xgboost,
          ft_transformer: result.ft_transformer,
        },
        interpolatedReference
      );

      // Show nearest rows table
      if (result.nearest_rows && result.nearest_rows.length > 0) {
        html += `<div class="card mb-lg">
          <div class="card__header"><div class="card__title">En Yakin Gercek Satirlar</div></div>
          <p style="font-size:0.82rem;color:var(--text-muted);margin-bottom:var(--space-md)">
            Bu satırlar sadece bağlam için gösterilir. Custom input hata kıyasının ana referansı interpolasyon değeridir.
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
// Tab 5: Dataset Tools
// ---------------------------------------------------------------------------

async function loadDatasetToolsTab(force = false) {
  if (!force) window._datasetToolsLoaded = true;
  try {
    const [status, commands] = await Promise.all([
      api('/api/dataset-tools/status'),
      api('/api/dataset-tools/commands'),
    ]);
    renderDatasetToolStatus(status);
    renderDatasetToolCommands(commands);
    window._datasetToolCommands = commands;
  } catch (err) {
    const statusContainer = document.getElementById('dataset-status-cards');
    const stepsContainer = document.getElementById('dataset-tool-steps');
    if (statusContainer) statusContainer.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${err.message}</div></div>`;
    if (stepsContainer) stepsContainer.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">Dataset komutları yüklenemedi.</div></div>`;
    showToast('Veri üretimi sekmesi yüklenemedi: ' + err.message, 'error');
  }
}

function datasetStatusCard(label, ok, detail = '', installCommand = '') {
  const color = ok ? 'success' : 'warning';
  const value = ok ? 'Hazır' : 'Eksik';
  const actionHtml = (!ok && installCommand)
    ? `<button class="btn btn--secondary btn--sm metric-card__action" onclick="runDatasetToolCommand('${installCommand}')">Kur</button>`
    : '';
  const descHtml = detail
    ? `<div class="metric-card__desc metric-card__desc--visible">${escapeHTML(detail)}</div>`
    : '';
  return `<div class="metric-card" ${detail ? `title="${escapeHTML(detail)}"` : ''}>
    <div class="metric-card__label">${escapeHTML(label)}</div>
    <div class="metric-card__value metric-card__value--${color}">${escapeHTML(value)}</div>
    ${descHtml}
    ${actionHtml}
  </div>`;
}

function renderDatasetToolStatus(status) {
  const container = document.getElementById('dataset-status-cards');
  if (!container) return;

  const packageCards = (status.python_packages || []).map(pkg =>
    datasetStatusCard(pkg.label, pkg.ok, pkg.ok ? `${pkg.module} hazır` : (pkg.hint || `pip install ${pkg.package}`), pkg.install_command)
  );
  const systemCards = (status.system_tools || []).map(tool =>
    datasetStatusCard(tool.label, tool.ok, tool.ok ? (tool.hint || tool.path) : tool.hint, tool.install_command)
  );

  container.innerHTML = [
    datasetStatusCard('Dataset Tool', status.app_dir_exists, status.app_dir || ''),
    datasetStatusCard('Python Env', true, `Ana proje ortamı: ${status.python || 'sys.executable'}`),
    datasetStatusCard('Dataset GUI', status.gui_script, 'synthetic_data_gui.py'),
    datasetStatusCard('Sentetik Üretim', status.synthetic_script, '3-synthetic_production.py'),
    datasetStatusCard('U-Net Eğitim', status.train_script, 'train_unet.py'),
    datasetStatusCard('Segmentasyon', status.segment_script, '5-segment_curves.py'),
    datasetStatusCard('Model Dosyası', status.model_file, 'Model/best_unet_model (3).pth'),
    datasetStatusCard('Örnek Grafikler', status.grafikler_dir, 'Grafikler/'),
    datasetStatusCard('Demo Grafikler', status.demo_graphs_dir, 'demo_graphs/'),
    datasetStatusCard('Dataset Klasörü', status.dataset_dir, 'dataset_production/'),
    ...packageCards,
    ...systemCards,
  ].join('');
}

function renderDatasetToolCommands(commands) {
  const container = document.getElementById('dataset-tool-steps');
  if (!container) return;

  const pipelineLabels = {
    env_check: '0. Ortam Kontrolü',
    install_deps: '1. Python Bağımlılıkları',
    install_poppler: '2. Poppler Kurulumu',
    install_tesseract: '3. Tesseract Kurulumu',
    open_gui: '4. Dataset GUI',
    synthetic_demo: '5. Sentetik Veri',
    train_demo: '6. U-Net Eğitim',
    segment_export_demo: '7. Segmentasyon / Excel',
  };

  container.innerHTML = commands.map((cmd, i) => `
    <div class="step" id="dataset-step-${cmd.id}">
      <div class="step__indicator">${i + 1}</div>
      <div class="step__content">
        <div class="step__title">${pipelineLabels[cmd.id] || cmd.label}</div>
        <div class="step__meta">${cmd.description}</div>
        <div class="step__meta mt-sm">Tahmini süre: ${cmd.eta}${cmd.id === 'open_gui' ? ' • aynı sekmede açılır' : (cmd.detached ? ' • ayrı pencere' : '')}</div>
        <div class="step__command">${cmd.display_command}</div>
        <div class="step__meta mt-sm">Çıktılar: ${cmd.artifacts?.length ? cmd.artifacts.join(', ') : 'dosya üretmeyebilir'}</div>
      </div>
      <button class="btn btn--secondary btn--sm" onclick="${cmd.id === 'open_gui' ? 'showDatasetWebGui()' : `runDatasetToolCommand('${cmd.id}')`}">
        ${cmd.id === 'open_gui' ? 'Sekmede Aç' : 'Çalıştır'}
      </button>
    </div>
  `).join('');
}

function clearDatasetToolLog() {
  const logArea = document.getElementById('dataset-tool-log');
  if (logArea) logArea.textContent = 'Log temizlendi.';
}

async function showDatasetWebGui() {
  const panel = document.getElementById('dataset-web-gui');
  if (!panel) return;
  panel.hidden = false;
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  await loadDatasetGuiDefaults(false);
  showToast('Dataset GUI artık aynı sekmede hazır.', 'success');
}

function setNestedValue(target, path, value) {
  const parts = path.split('.');
  let cursor = target;
  for (let i = 0; i < parts.length - 1; i += 1) {
    cursor[parts[i]] = cursor[parts[i]] || {};
    cursor = cursor[parts[i]];
  }
  cursor[parts[parts.length - 1]] = value;
}

function getNestedValue(source, path) {
  return path.split('.').reduce((obj, key) => (obj && obj[key] !== undefined ? obj[key] : undefined), source);
}

function applyDatasetGuiConfig(config) {
  document.querySelectorAll('[data-dgui]').forEach((input) => {
    const value = getNestedValue(config, input.dataset.dgui);
    if (value === undefined) return;
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = value;
  });
}

function collectDatasetGuiConfig() {
  const config = {};
  document.querySelectorAll('[data-dgui]').forEach((input) => {
    const value = input.type === 'checkbox' ? input.checked : input.value;
    setNestedValue(config, input.dataset.dgui, value);
  });
  return config;
}

async function loadDatasetGuiDefaults(force = true) {
  if (!force && window._datasetGuiDefaultsLoaded) return;
  try {
    const defaults = await api('/api/dataset-gui/defaults');
    const saved = localStorage.getItem('datasetGuiPreset');
    applyDatasetGuiConfig(saved ? { ...defaults, ...JSON.parse(saved) } : defaults);
    window._datasetGuiDefaultsLoaded = true;
  } catch (err) {
    showToast('Dataset GUI varsayılanları yüklenemedi: ' + err.message, 'error');
  }
}

function saveDatasetGuiPreset() {
  localStorage.setItem('datasetGuiPreset', JSON.stringify(collectDatasetGuiConfig()));
  showToast('Dataset GUI ayarları tarayıcıya kaydedildi.', 'success');
}

async function loadDatasetGuiPreset() {
  const saved = localStorage.getItem('datasetGuiPreset');
  if (!saved) {
    await loadDatasetGuiDefaults(true);
    showToast('Kayıtlı ayar yok; varsayılanlar yüklendi.', 'info');
    return;
  }
  applyDatasetGuiConfig(JSON.parse(saved));
  showToast('Dataset GUI ayarları yüklendi.', 'success');
}

async function runDatasetGuiPipeline() {
  const logArea = document.getElementById('dataset-tool-log');
  const config = collectDatasetGuiConfig();
  if (logArea) {
    logArea.textContent += '\n\n### Dataset GUI Web Pipeline\n';
    logArea.scrollTop = logArea.scrollHeight;
  }

  try {
    const resp = await fetch('/api/dataset-gui/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `API error ${resp.status}`);
    }

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
          if (data.type === 'output' && logArea) {
            logArea.textContent += data.text + '\n';
            logArea.scrollTop = logArea.scrollHeight;
          } else if (data.type === 'done') {
            showToast(`Dataset GUI pipeline ${data.exit_code === 0 ? 'tamamlandı' : 'başarısız'}`, data.exit_code === 0 ? 'success' : 'error');
            loadDatasetToolsTab(true);
          } else if (data.type === 'error') {
            if (logArea) logArea.textContent += `ERROR: ${data.text}\n`;
            showToast('Dataset GUI pipeline hatası: ' + data.text, 'error');
          }
        } catch (e) { /* skip malformed SSE chunks */ }
      }
    }
  } catch (err) {
    if (logArea) logArea.textContent += `\nFATAL: ${err.message}\n`;
    showToast('Dataset GUI pipeline hatası: ' + err.message, 'error');
  }
}

async function stopDatasetGuiPipeline() {
  try {
    const result = await api('/api/dataset-gui/stop', { method: 'POST', body: JSON.stringify({}) });
    showToast(result.message || 'Durdurma isteği gönderildi.', result.stopped ? 'warning' : 'info');
  } catch (err) {
    showToast('Durdurma hatası: ' + err.message, 'error');
  }
}

async function loadDatasetGuiPreview(kind) {
  const config = collectDatasetGuiConfig();
  const params = new URLSearchParams({ kind });
  if (kind === 'training') {
    params.set('dataset_path', config.train?.dataset_path || 'dataset_production');
  } else {
    params.set('output_dir', config.inference?.output_dir || 'segmentation_results');
  }
  const panel = document.getElementById('dataset-preview-panel');
  if (panel) panel.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Önizleme yükleniyor…</span></div>';
  upgradePlaneLoaders(panel || document);
  try {
    const data = await api(`/api/dataset-gui/preview?${params.toString()}`);
    window._currentDatasetPreview = data;
    renderDatasetGuiPreview(data);
  } catch (err) {
    if (panel) panel.innerHTML = `<div class="empty-state"><div class="empty-state__text text-warning">${escapeHTML(err.message)}</div></div>`;
    showToast('Önizleme yüklenemedi: ' + err.message, 'error');
  }
}

function renderDatasetGuiPreview(data) {
  const panel = document.getElementById('dataset-preview-panel');
  if (!panel) return;
  const maskHtml = data.mask_url
    ? `<figure><img src="${escapeHTML(data.mask_url)}" alt="Maske önizleme"><figcaption>Maske</figcaption></figure>`
    : '';
  const editHtml = data.mask_url
    ? `<button class="btn btn--primary btn--sm" onclick="openDatasetMaskEditor()">Maskeyi Düzenle</button>`
    : '<span class="form-help">Bu önizlemede düzenlenebilir maske bulunamadı.</span>';
  panel.innerHTML = `
    <div class="dataset-preview__meta">${escapeHTML(data.kind === 'training' ? 'Eğitim verisi' : 'Segmentasyon sonucu')} • ${data.count} dosya içinden seçildi ${editHtml}</div>
    <div class="dataset-preview__grid">
      <figure><img src="${escapeHTML(data.image_url)}" alt="Görsel önizleme"><figcaption>${escapeHTML(data.image || 'Görsel')}</figcaption></figure>
      ${maskHtml}
    </div>
    <div id="dataset-mask-editor"></div>
  `;
}

function openCurrentDatasetPreview() {
  const data = window._currentDatasetPreview;
  if (!data?.image_url) {
    showToast('Açılacak önizleme yok.', 'info');
    return;
  }
  window.open(data.image_url, '_blank', 'noopener');
}

function openDatasetMaskEditor() {
  const data = window._currentDatasetPreview;
  const editor = document.getElementById('dataset-mask-editor');
  if (!editor || !data?.image_url || !data?.mask_url || !data?.mask) {
    showToast('Düzenlenebilir maske yok.', 'info');
    return;
  }

  editor.innerHTML = `
    <div class="mask-editor">
      <div class="mask-editor__toolbar">
        <label>Fırça <input id="mask-brush-size" type="range" min="1" max="50" value="5"></label>
        <label><input type="radio" name="mask-mode" value="add" checked> Ekle</label>
        <label><input type="radio" name="mask-mode" value="remove"> Sil</label>
        <button class="btn btn--success btn--sm" onclick="saveDatasetMask()">Kaydet</button>
        <button class="btn btn--secondary btn--sm" onclick="closeDatasetMaskEditor()">Kapat</button>
      </div>
      <div class="mask-editor__canvas-wrap">
        <canvas id="mask-edit-canvas"></canvas>
      </div>
      <div class="form-help">Sol tıkla çiz. Ekle modu maskeyi mavi overlay olarak ekler; Sil modu maskeyi kaldırır.</div>
    </div>
  `;
  initDatasetMaskCanvas(data);
}

function closeDatasetMaskEditor() {
  const editor = document.getElementById('dataset-mask-editor');
  if (editor) editor.innerHTML = '';
}

async function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'same-origin';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = `${src}${src.includes('?') ? '&' : '?'}t=${Date.now()}`;
  });
}

async function initDatasetMaskCanvas(data) {
  try {
    const [baseImg, maskImg] = await Promise.all([
      loadImageElement(data.image_url),
      loadImageElement(data.mask_url),
    ]);

    const canvas = document.getElementById('mask-edit-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d');
    const overlayCanvas = document.createElement('canvas');
    const overlayCtx = overlayCanvas.getContext('2d');

    canvas.width = baseImg.naturalWidth;
    canvas.height = baseImg.naturalHeight;
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;

    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
    const maskData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < maskData.data.length; i += 4) {
      const value = maskData.data[i] > 20 || maskData.data[i + 1] > 20 || maskData.data[i + 2] > 20 ? 255 : 0;
      maskData.data[i] = 255;
      maskData.data[i + 1] = 255;
      maskData.data[i + 2] = 255;
      maskData.data[i + 3] = value;
    }
    maskCtx.putImageData(maskData, 0, 0);

    window._maskEditorState = { canvas, ctx, baseImg, maskCanvas, maskCtx, overlayCanvas, overlayCtx, maskPath: data.mask, drawing: false };

    const redraw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);
      overlayCtx.clearRect(0, 0, canvas.width, canvas.height);
      overlayCtx.fillStyle = 'rgba(37, 99, 235, 0.48)';
      overlayCtx.fillRect(0, 0, canvas.width, canvas.height);
      overlayCtx.globalCompositeOperation = 'destination-in';
      overlayCtx.drawImage(maskCanvas, 0, 0);
      overlayCtx.globalCompositeOperation = 'source-over';
      ctx.drawImage(overlayCanvas, 0, 0);
    };
    window._maskEditorState.redraw = redraw;
    redraw();

    const paint = (event) => {
      const state = window._maskEditorState;
      if (!state) return;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * (canvas.width / rect.width);
      const y = (event.clientY - rect.top) * (canvas.height / rect.height);
      const displayScale = canvas.width / rect.width;
      const radius = Number(document.getElementById('mask-brush-size')?.value || 5) * displayScale;
      const mode = document.querySelector('input[name="mask-mode"]:checked')?.value || 'add';
      maskCtx.save();
      maskCtx.globalCompositeOperation = mode === 'add' ? 'source-over' : 'destination-out';
      maskCtx.fillStyle = 'rgba(255,255,255,1)';
      maskCtx.beginPath();
      maskCtx.arc(x, y, radius, 0, Math.PI * 2);
      maskCtx.fill();
      maskCtx.restore();
      redraw();
    };

    canvas.onpointerdown = (event) => {
      window._maskEditorState.drawing = true;
      canvas.setPointerCapture(event.pointerId);
      paint(event);
    };
    canvas.onpointermove = (event) => {
      if (window._maskEditorState?.drawing) paint(event);
    };
    canvas.onpointerup = (event) => {
      window._maskEditorState.drawing = false;
      try { canvas.releasePointerCapture(event.pointerId); } catch (e) { /* ignore */ }
    };
    canvas.onpointerleave = () => {
      if (window._maskEditorState) window._maskEditorState.drawing = false;
    };
  } catch (err) {
    showToast('Maske editörü yüklenemedi: ' + err.message, 'error');
  }
}

async function saveDatasetMask() {
  const state = window._maskEditorState;
  if (!state) {
    showToast('Kaydedilecek maske yok.', 'info');
    return;
  }
  try {
    const dataUrl = state.maskCanvas.toDataURL('image/png');
    const result = await api('/api/dataset-gui/mask', {
      method: 'POST',
      body: JSON.stringify({ mask: state.maskPath, data_url: dataUrl }),
    });
    showToast('Maske kaydedildi.', 'success');
    if (window._currentDatasetPreview) {
      window._currentDatasetPreview.mask_url = `${result.mask_url}?t=${Date.now()}`;
      renderDatasetGuiPreview(window._currentDatasetPreview);
    }
  } catch (err) {
    showToast('Maske kaydetme hatası: ' + err.message, 'error');
  }
}

async function runDatasetToolCommand(id) {
  const commands = window._datasetToolCommands || [];
  const cmd = commands.find(item => item.id === id);
  const label = cmd?.label || id;
  const stepEl = document.getElementById(`dataset-step-${id}`);
  const logArea = document.getElementById('dataset-tool-log');

  if (stepEl) {
    stepEl.className = 'step step--running';
    stepEl.querySelector('.step__indicator').textContent = '⟳';
  }
  if (logArea) {
    logArea.textContent += `\n\n### ${label}\n`;
    logArea.scrollTop = logArea.scrollHeight;
  }

  try {
    const resp = await fetch('/api/dataset-tools/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `API error ${resp.status}`);
    }

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
            if (logArea) {
              logArea.textContent += data.text + '\n';
              logArea.scrollTop = logArea.scrollHeight;
            }
          } else if (data.type === 'done') {
            if (stepEl) {
              stepEl.className = data.exit_code === 0 ? 'step step--done' : 'step step--error';
              stepEl.querySelector('.step__indicator').textContent = data.exit_code === 0 ? '✓' : '✗';
            }
            const status = data.exit_code === 0 ? 'success' : 'error';
            showToast(`${label}: ${data.exit_code === 0 ? 'tamamlandı' : 'başarısız'}`, status);
            loadDatasetToolsTab(true);
          } else if (data.type === 'error') {
            if (logArea) logArea.textContent += `ERROR: ${data.text}\n`;
            if (stepEl) {
              stepEl.className = 'step step--error';
              stepEl.querySelector('.step__indicator').textContent = '✗';
            }
          }
        } catch (e) { /* skip malformed SSE chunks */ }
      }
    }
  } catch (err) {
    if (logArea) logArea.textContent += `\nFATAL: ${err.message}\n`;
    if (stepEl) {
      stepEl.className = 'step step--error';
      stepEl.querySelector('.step__indicator').textContent = '✗';
    }
    showToast(`${label} hatası: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Tab 6: Info
// ---------------------------------------------------------------------------

async function loadInfoTab() {
  window._infoLoaded = true;
  try {
    const data = await api('/api/info');
    renderInfoCards('info-tabs', data.tabs || [], 'name', 'description');
    renderInfoCards('info-methods', data.methods || [], 'name', 'role');
    renderCostSimulatorInfo(data.cost_simulator || {});
    renderInfoReadmes(data.readmes || []);
  } catch (err) {
    showToast('Bilgi sekmesi yüklenemedi: ' + err.message, 'error');
  }
}

function escapeHTML(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderInfoCards(containerId, items, titleKey, descKey) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!items.length) {
    container.innerHTML = '<div class="empty-state"><div class="empty-state__text">Bilgi bulunamadı.</div></div>';
    return;
  }
  container.innerHTML = `<div class="info-list">${items.map(item => `
    <div class="info-list__item">
      <div class="info-list__title">${escapeHTML(item[titleKey])}</div>
      <div class="info-list__desc">${escapeHTML(item[descKey])}</div>
    </div>
  `).join('')}</div>`;
}

function renderCostSimulatorInfo(info) {
  const container = document.getElementById('info-cost-simulator');
  if (!container) return;

  if (!info || !Object.keys(info).length) {
    container.innerHTML = '<div class="empty-state"><div class="empty-state__text">Maliyet hesabı açıklaması bulunamadı.</div></div>';
    return;
  }

  const renderItems = (items = []) => items.map(item => `
    <div class="info-list__item">
      <div class="info-list__title">${escapeHTML(item.name || item.label)}</div>
      <div class="info-list__desc">${escapeHTML(item.detail || item.value)}</div>
    </div>
  `).join('');

  const renderFormulas = (items = []) => items.map(item => `
    <div class="cost-info__formula">
      <div class="cost-info__formula-title">${escapeHTML(item.name)}</div>
      <pre>${escapeHTML(item.formula)}</pre>
      ${item.detail ? `<div class="info-list__desc">${escapeHTML(item.detail)}</div>` : ''}
    </div>
  `).join('');

  container.innerHTML = `
    <div class="cost-info">
      <div class="cost-info__intro">
        <div class="info-list__title">${escapeHTML(info.title || 'Maliyet simülatörü')}</div>
        <div class="info-list__desc">${escapeHTML(info.summary || '')}</div>
      </div>

      <div class="cost-info__grid">
        <div>
          <div class="cost-info__section-title">Veri Kaynakları</div>
          <div class="info-list">${renderItems(info.sources || [])}</div>
        </div>
        <div>
          <div class="cost-info__section-title">Normalize Hedefler</div>
          <div class="cost-targets">
            ${(info.targets || []).map(target => `
              <div class="cost-target">
                <span>${escapeHTML(target.label)}</span>
                <strong>${escapeHTML(target.value)}</strong>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <div class="cost-info__section-title">Maliyet Formülleri</div>
      <div class="cost-info__formula-grid">${renderFormulas(info.formulas || [])}</div>

      ${(info.runtime_formulas || []).length ? `
        <details class="readme-card cost-info__details">
          <summary>
            <span>Ölçümden türetilmiş senaryo formülleri</span>
            <small>benchmark tabanlı</small>
          </summary>
          <div class="cost-info__formula-list">${renderFormulas(info.runtime_formulas || [])}</div>
        </details>
      ` : ''}

      <div class="simulator-help">
        ${(info.notes || []).map(note => `<div>${escapeHTML(note)}</div>`).join('')}
      </div>
    </div>
  `;
}

function renderInfoReadmes(readmes) {
  const container = document.getElementById('info-readmes');
  if (!container) return;
  container.innerHTML = readmes.map(readme => `
    <details class="readme-card" ${readme.title === 'Ana README' ? 'open' : ''}>
      <summary>
        <span>${escapeHTML(readme.title)}</span>
        <small>${readme.exists ? escapeHTML(readme.path) : 'bulunamadı'}</small>
      </summary>
      <pre class="readme-body">${escapeHTML(readme.exists ? readme.content : 'Bu README dosyası bulunamadı.')}</pre>
    </details>
  `).join('');
}

// ---------------------------------------------------------------------------
// Tab 7: Setup
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

function renderDataTable(containerId, rows, cols, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) {
    container.innerHTML = '<div class="empty-state"><div class="empty-state__text">Gösterilecek satır yok.</div></div>';
    return;
  }
  const formatCell = (value, col) => {
    if (value == null || Number.isNaN(value)) return '-';
    if (col.format === 'int') return formatInt(value);
    if (col.format === 'float4') return formatFloat(value, 4);
    if (col.format === 'float6') return formatFloat(value, 6);
    return escapeHTML(value);
  };
  container.innerHTML = `
    <div class="table-container">
      <table class="data-table">
        <thead><tr>${cols.map(col => `<th>${escapeHTML(col.label)}</th>`).join('')}</tr></thead>
        <tbody>
          ${safeRows.map((row, idx) => `<tr ${options.onRowClick ? `onclick="${options.onRowClick}(${idx})" style="cursor:pointer"` : ''}>
            ${cols.map(col => `<td>${formatCell(row[col.key], col)}</td>`).join('')}
          </tr>`).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderPagination(containerId, pageData, onPageClick) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const page = Number(pageData.page || 1);
  const totalPages = Number(pageData.total_pages || 1);
  if (totalPages <= 1) {
    container.innerHTML = '';
    return;
  }
  const pages = [1, page - 1, page, page + 1, totalPages].filter((value, index, arr) => value >= 1 && value <= totalPages && arr.indexOf(value) === index);
  container.innerHTML = `
    <div class="pagination">
      <button class="pagination__btn" ${page <= 1 ? 'disabled' : ''} data-page="${page - 1}">Önceki</button>
      ${pages.map(p => `<button class="pagination__btn ${p === page ? 'pagination__btn--active' : ''}" data-page="${p}">${p}</button>`).join('')}
      <button class="pagination__btn" ${page >= totalPages ? 'disabled' : ''} data-page="${page + 1}">Sonraki</button>
    </div>
  `;
  container.querySelectorAll('[data-page]').forEach(button => {
    button.addEventListener('click', () => onPageClick(Number(button.dataset.page)));
  });
}

function renderRowDetail(containerId, row, cols) {
  const container = document.getElementById(containerId);
  if (!container || !row) return;
  container.innerHTML = `<div class="result-grid">${cols.map(col => metricCardHTML(col.label, row[col.key] == null ? '-' : (col.format === 'float6' ? formatFloat(row[col.key], 6) : col.format === 'float4' ? formatFloat(row[col.key], 4) : col.format === 'int' ? formatInt(row[col.key]) : escapeHTML(row[col.key])))).join('')}</div>`;
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
  initPlaneLoaderObserver();
  initTabs();
  loadStatus();

  // Load default tab
  loadOverview();
});
