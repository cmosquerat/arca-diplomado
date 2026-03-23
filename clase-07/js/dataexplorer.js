/*  dataexplorer.js – Clase 07
 *  Interactive DataFrame Explorer + GroupBy Demo
 *  Pure JS — no Pyodide needed. Teaches pandas concepts visually.
 *
 *  Components:
 *    DataExplorer  — interactive HTML table with search/sort/column-toggle
 *                    + live pandas code generator
 *    GroupByDemo   — dropdown-driven groupby + instant results + pandas code
 */

// ─────────────────────────────────────────────────────────────────────────────
//  Embedded 20-row Titanic dataset
// ─────────────────────────────────────────────────────────────────────────────
const TITANIC_DATA = [
  {PassengerId:1,  Survived:0, Pclass:3, Sex:"male",   Age:22,   Fare:7.25,   Embarked:"S"},
  {PassengerId:2,  Survived:1, Pclass:1, Sex:"female", Age:38,   Fare:71.28,  Embarked:"C"},
  {PassengerId:3,  Survived:1, Pclass:3, Sex:"female", Age:26,   Fare:7.92,   Embarked:"S"},
  {PassengerId:4,  Survived:1, Pclass:1, Sex:"female", Age:35,   Fare:53.10,  Embarked:"S"},
  {PassengerId:5,  Survived:0, Pclass:3, Sex:"male",   Age:35,   Fare:8.05,   Embarked:"S"},
  {PassengerId:6,  Survived:0, Pclass:3, Sex:"male",   Age:null, Fare:8.46,   Embarked:"Q"},
  {PassengerId:7,  Survived:0, Pclass:1, Sex:"male",   Age:54,   Fare:51.86,  Embarked:"S"},
  {PassengerId:8,  Survived:0, Pclass:3, Sex:"male",   Age:2,    Fare:21.07,  Embarked:"S"},
  {PassengerId:9,  Survived:1, Pclass:3, Sex:"female", Age:27,   Fare:11.13,  Embarked:"S"},
  {PassengerId:10, Survived:1, Pclass:2, Sex:"female", Age:14,   Fare:30.07,  Embarked:"C"},
  {PassengerId:11, Survived:1, Pclass:3, Sex:"female", Age:4,    Fare:16.70,  Embarked:"S"},
  {PassengerId:12, Survived:1, Pclass:1, Sex:"female", Age:58,   Fare:26.55,  Embarked:"S"},
  {PassengerId:13, Survived:0, Pclass:3, Sex:"male",   Age:20,   Fare:8.05,   Embarked:"S"},
  {PassengerId:14, Survived:0, Pclass:3, Sex:"male",   Age:39,   Fare:31.27,  Embarked:"S"},
  {PassengerId:15, Survived:0, Pclass:3, Sex:"female", Age:14,   Fare:7.85,   Embarked:"S"},
  {PassengerId:16, Survived:1, Pclass:2, Sex:"female", Age:55,   Fare:16.00,  Embarked:"S"},
  {PassengerId:17, Survived:0, Pclass:3, Sex:"male",   Age:2,    Fare:29.12,  Embarked:"Q"},
  {PassengerId:18, Survived:1, Pclass:2, Sex:"male",   Age:null, Fare:13.00,  Embarked:"S"},
  {PassengerId:19, Survived:0, Pclass:3, Sex:"female", Age:31,   Fare:18.00,  Embarked:"S"},
  {PassengerId:20, Survived:1, Pclass:3, Sex:"female", Age:null, Fare:7.22,   Embarked:"C"},
];

const TITANIC_COLS = ['PassengerId','Survived','Pclass','Sex','Age','Fare','Embarked'];


// ─────────────────────────────────────────────────────────────────────────────
//  DataExplorer
// ─────────────────────────────────────────────────────────────────────────────
const DataExplorer = (() => {

  let _state = {
    search: '',
    sortCol: null,
    sortDir: 1,
    visibleCols: new Set(['Survived','Pclass','Sex','Age','Fare','Embarked']),
    containerId: null
  };

  // ── Filter + sort ──────────────────────────────────────────────────────────

  function _getFiltered() {
    let rows = [...TITANIC_DATA];

    if (_state.search) {
      const q = _state.search.toLowerCase();
      rows = rows.filter(r =>
        TITANIC_COLS.some(c => {
          const v = r[c];
          return v !== null && String(v).toLowerCase().includes(q);
        })
      );
    }

    if (_state.sortCol) {
      rows.sort((a, b) => {
        const av = a[_state.sortCol], bv = b[_state.sortCol];
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        return av < bv ? -_state.sortDir : av > bv ? _state.sortDir : 0;
      });
    }

    return rows;
  }

  // ── Pandas code generator ──────────────────────────────────────────────────

  function _buildCode() {
    const parts = ['df'];
    const hiddenCols = TITANIC_COLS.filter(c => !_state.visibleCols.has(c));

    if (_state.search) {
      parts.push(`.query("… '${_state.search}' …")  # busca en columnas`);
    }

    if (hiddenCols.length) {
      const visible = [..._state.visibleCols];
      parts.push(`[${visible.map(c => `"${c}"`).join(', ')}]`);
    }

    if (_state.sortCol) {
      const asc = _state.sortDir === 1 ? 'True' : 'False';
      parts.push(`.sort_values("${_state.sortCol}", ascending=${asc})`);
    }

    return parts.join('');
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  function _render() {
    const c = document.getElementById(_state.containerId);
    if (!c) return;

    const rows = _getFiltered();
    const cols = TITANIC_COLS.filter(col => _state.visibleCols.has(col));

    // ── Table ──────────────────────────────────────────────────────────────
    const tableWrap = c.querySelector('.df-table-wrap');
    if (!tableWrap) return;

    let html = '<table class="df-table"><thead><tr>';
    cols.forEach(col => {
      const arrow = _state.sortCol === col
        ? (_state.sortDir === 1 ? ' ↑' : ' ↓') : '';
      html += `<th class="df-th${_state.sortCol === col ? ' df-sorted' : ''}"
                   onclick="DataExplorer.sortBy('${col}')">${col}${arrow}</th>`;
    });
    html += '</tr></thead><tbody>';

    rows.forEach(row => {
      html += '<tr>';
      cols.forEach(col => {
        const val = row[col];
        let cls = '';
        if (val === null) cls = 'df-nan';
        else if (col === 'Survived') cls = val === 1 ? 'df-survived' : 'df-died';
        const display = val === null ? 'NaN' : val;
        html += `<td class="${cls}">${display}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    tableWrap.innerHTML = html;

    // ── Stats ──────────────────────────────────────────────────────────────
    const stats = c.querySelector('.df-stats');
    if (stats) {
      stats.innerHTML = `<span><strong>${rows.length}</strong> de ${TITANIC_DATA.length} filas</span>
        <span style="color:var(--code-orange)"><strong>${TITANIC_DATA.filter(r=>r.Age===null).length}</strong> valores NaN en Age</span>`;
    }

    // ── Code gen ───────────────────────────────────────────────────────────
    const codeEl = c.querySelector('.df-code-val');
    if (codeEl) codeEl.textContent = _buildCode();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  function init(containerId) {
    _state.containerId = containerId;
    const c = document.getElementById(containerId);
    if (!c) return;

    // Search
    const searchEl = c.querySelector('.df-search');
    if (searchEl) {
      searchEl.addEventListener('input', e => {
        _state.search = e.target.value;
        _render();
      });
    }

    // Column toggles
    c.querySelectorAll('.df-col-toggle').forEach(chk => {
      chk.addEventListener('change', e => {
        const col = e.target.dataset.col;
        if (e.target.checked) _state.visibleCols.add(col);
        else _state.visibleCols.delete(col);
        _render();
      });
    });

    // Reset
    const resetBtn = c.querySelector('.df-reset-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        _state.search = '';
        _state.sortCol = null;
        _state.sortDir = 1;
        _state.visibleCols = new Set(['Survived','Pclass','Sex','Age','Fare','Embarked']);
        c.querySelectorAll('.df-col-toggle').forEach(chk => { chk.checked = _state.visibleCols.has(chk.dataset.col); });
        const searchEl = c.querySelector('.df-search');
        if (searchEl) searchEl.value = '';
        _render();
      });
    }

    _render();
  }

  function sortBy(col) {
    if (_state.sortCol === col) {
      _state.sortDir *= -1;
    } else {
      _state.sortCol = col;
      _state.sortDir = 1;
    }
    _render();
  }

  return { init, sortBy };
})();


// ─────────────────────────────────────────────────────────────────────────────
//  GroupByDemo
// ─────────────────────────────────────────────────────────────────────────────
const GroupByDemo = (() => {

  const COL_LABELS = { Sex:'Sex', Pclass:'Pclass', Embarked:'Embarked' };

  const AGG_DEFS = {
    count: {
      label: 'Cantidad de pasajeros',
      compute: rows => rows.length,
      fmt: v => v,
      unit: 'pasajeros',
      code: col => `df.groupby("${col}").size().reset_index(name="count")`
    },
    survived: {
      label: 'Tasa de supervivencia (%)',
      compute: rows => (rows.filter(r => r.Survived === 1).length / rows.length * 100).toFixed(1),
      fmt: v => v + '%',
      unit: '',
      code: col => `df.groupby("${col}")["Survived"].mean().mul(100).round(1)`
    },
    mean_age: {
      label: 'Promedio de edad',
      compute: rows => {
        const ages = rows.filter(r => r.Age !== null).map(r => r.Age);
        return ages.length ? (ages.reduce((a,b)=>a+b,0)/ages.length).toFixed(1) : 'N/A';
      },
      fmt: v => v === 'N/A' ? v : v + ' años',
      unit: '',
      code: col => `df.groupby("${col}")["Age"].mean().round(1)`
    },
    sum_fare: {
      label: 'Suma de tarifas',
      compute: rows => rows.reduce((a,r)=>a+r.Fare,0).toFixed(2),
      fmt: v => '$' + v,
      unit: '',
      code: col => `df.groupby("${col}")["Fare"].sum().round(2)`
    }
  };

  function render(containerId) {
    const c = document.getElementById(containerId);
    if (!c) return;

    const colSel = c.querySelector('.gb-col-sel');
    const aggSel = c.querySelector('.gb-agg-sel');
    if (!colSel || !aggSel) return;

    const col = colSel.value;
    const aggKey = aggSel.value;
    const aggDef = AGG_DEFS[aggKey];

    // Group data
    const groups = {};
    TITANIC_DATA.forEach(row => {
      const key = row[col] !== null ? String(row[col]) : 'N/A';
      if (!groups[key]) groups[key] = [];
      groups[key].push(row);
    });

    // Compute results
    const results = Object.entries(groups)
      .map(([key, rows]) => ({ key, value: aggDef.compute(rows), count: rows.length }))
      .sort((a,b) => (isNaN(a.value) || isNaN(b.value)) ? 0 : b.value - a.value);

    // ── Result table ────────────────────────────────────────────────────────
    const maxVal = Math.max(...results.map(r => parseFloat(r.value) || 0));

    let html = `<table class="df-table gb-result-table"><thead><tr>
      <th>${col}</th>
      <th>${aggDef.label}</th>
      <th>n</th>
      <th style="width:100px">Visual</th>
    </tr></thead><tbody>`;

    results.forEach(({ key, value, count }) => {
      const num = parseFloat(value) || 0;
      const pct = maxVal > 0 ? (num / maxVal * 100).toFixed(1) : 0;
      const colClass = key === 'female' ? 'df-survived' : key === 'male' ? 'df-died' : '';
      html += `<tr>
        <td class="${colClass}" style="font-weight:700">${key}</td>
        <td style="font-weight:600;color:var(--arca-red)">${aggDef.fmt(value)}</td>
        <td style="color:var(--text-muted)">${count}</td>
        <td><div class="gb-bar" style="width:${pct}%"></div></td>
      </tr>`;
    });
    html += '</tbody></table>';

    const resultEl = c.querySelector('.gb-result');
    if (resultEl) resultEl.innerHTML = html;

    // ── Pandas code ─────────────────────────────────────────────────────────
    const codeEl = c.querySelector('.gb-code-val');
    if (codeEl) codeEl.textContent = aggDef.code(col);
  }

  return { render };
})();
