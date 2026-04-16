/* ======================================================================
   Specific Range Studio — Table Rendering & Filtering
   ====================================================================== */

/**
 * Render a data table inside a container element.
 *
 * @param {string} containerId   - ID of the wrapper element.
 * @param {Array}  rows          - Array of row objects.
 * @param {Array}  columns       - Array of {key, label, format?} descriptors.
 * @param {Object} options       - {errorColumn?, onRowClick?, maxErrorValue?}
 */
function renderDataTable(containerId, rows, columns, options = {}) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const { errorColumn, onRowClick, maxErrorValue } = options;

  if (!rows || rows.length === 0) {
    container.innerHTML =
      '<div class="empty-state"><div class="empty-state__icon">📋</div>' +
      '<div class="empty-state__text">Gösterilecek veri bulunamadı</div></div>';
    return;
  }

  let html = '<div class="table-container"><table class="data-table"><thead><tr>';
  columns.forEach(col => {
    html += `<th data-key="${col.key}">${col.label}</th>`;
  });
  html += '</tr></thead><tbody>';

  rows.forEach((row, rowIdx) => {
    const clickAttr = onRowClick ? ` onclick="${onRowClick}(${rowIdx})" style="cursor:pointer"` : '';
    html += `<tr data-idx="${rowIdx}"${clickAttr}>`;
    columns.forEach(col => {
      let value = row[col.key];
      let cellClass = '';

      // Format numbers
      if (col.format === 'float6' && typeof value === 'number') {
        value = value.toFixed(6);
      } else if (col.format === 'float4' && typeof value === 'number') {
        value = value.toFixed(4);
      } else if (col.format === 'int' && typeof value === 'number') {
        value = Math.round(value);
      }

      // Error coloring
      if (errorColumn && col.key === errorColumn && typeof row[col.key] === 'number') {
        const errVal = row[col.key];
        const maxErr = maxErrorValue || 0.01;
        const ratio = Math.min(errVal / maxErr, 1);
        if (ratio < 0.33) cellClass = 'cell-error-low';
        else if (ratio < 0.66) cellClass = 'cell-error-mid';
        else cellClass = 'cell-error-high';
      }

      html += `<td class="${cellClass}">${value != null ? value : '-'}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table></div>';
  container.innerHTML = html;
}

/**
 * Render pagination controls.
 *
 * @param {string}   containerId  - ID of the pagination wrapper.
 * @param {Object}   pageInfo     - {page, total_pages, total, per_page}
 * @param {Function} onPageChange - Callback(pageNumber)
 */
function renderPagination(containerId, pageInfo, onPageChange) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const { page, total_pages, total, per_page } = pageInfo;
  const start = (page - 1) * per_page + 1;
  const end = Math.min(page * per_page, total);

  let html = '<div class="pagination">';
  html += `<span>${start}–${end} / ${total} satır</span>`;
  html += '<div class="pagination__controls">';

  // Previous
  html += `<button class="pagination__btn" ${page <= 1 ? 'disabled' : ''} 
           onclick="(${onPageChange.toString()})(${page - 1})">◀</button>`;

  // Page numbers (show max 7)
  const maxButtons = 7;
  let startPage = Math.max(1, page - Math.floor(maxButtons / 2));
  let endPage = Math.min(total_pages, startPage + maxButtons - 1);
  if (endPage - startPage < maxButtons - 1) {
    startPage = Math.max(1, endPage - maxButtons + 1);
  }

  if (startPage > 1) {
    html += `<button class="pagination__btn" onclick="(${onPageChange.toString()})(1)">1</button>`;
    if (startPage > 2) html += '<span style="color:var(--text-dim);padding:0 4px">…</span>';
  }

  for (let i = startPage; i <= endPage; i++) {
    const active = i === page ? ' pagination__btn--active' : '';
    html += `<button class="pagination__btn${active}" onclick="(${onPageChange.toString()})(${i})">${i}</button>`;
  }

  if (endPage < total_pages) {
    if (endPage < total_pages - 1) html += '<span style="color:var(--text-dim);padding:0 4px">…</span>';
    html += `<button class="pagination__btn" onclick="(${onPageChange.toString()})(${total_pages})">${total_pages}</button>`;
  }

  // Next
  html += `<button class="pagination__btn" ${page >= total_pages ? 'disabled' : ''} 
           onclick="(${onPageChange.toString()})(${page + 1})">▶</button>`;

  html += '</div></div>';
  container.innerHTML = html;
}

/**
 * Render a compact detail card for a selected row.
 */
function renderRowDetail(containerId, row, columns) {
  const container = document.getElementById(containerId);
  if (!container || !row) {
    if (container) container.innerHTML = '<p class="text-muted">Detay görmek için bir satır seçin.</p>';
    return;
  }

  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px">';
  columns.forEach(col => {
    let value = row[col.key];
    if (col.format === 'float6' && typeof value === 'number') value = value.toFixed(6);
    else if (col.format === 'float4' && typeof value === 'number') value = value.toFixed(4);
    else if (col.format === 'int' && typeof value === 'number') value = Math.round(value);

    html += `<div style="padding:8px 12px;background:var(--bg-base);border-radius:var(--radius-sm);border:1px solid var(--glass-border)">
      <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px">${col.label}</div>
      <div style="font-family:var(--font-mono);font-size:0.9rem;color:var(--text-primary)">${value != null ? value : '-'}</div>
    </div>`;
  });
  html += '</div>';
  container.innerHTML = html;
}
