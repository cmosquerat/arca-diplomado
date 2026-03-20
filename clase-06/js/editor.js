/*  editor.js – Reusable Python editor with Pyodide execution
 *  Uses CodeMirror 5 for editing, Pyodide for in-browser Python.
 *  
 *  Usage:  <div class="py-editor" data-id="uniqueId" data-height="350px">
 *            print("hello")
 *          </div>
 *
 *  Attributes:
 *    data-id        unique identifier
 *    data-height    editor height (default 300px)
 *    data-readonly  "true" for read-only demos
 *    data-preload   comma-separated preload IDs (e.g. "vendedores,csv")
 */

const PyEditor = (() => {

  let pyodide = null;
  let pyodideReady = false;
  let pyodideLoading = false;
  const editors = {};
  const pendingRuns = [];

  const PRELOADS = {
    vendedores: `
vendedores = [
    {"nombre":"Ana", "ventas":250, "meta":150},
    {"nombre":"Luis","ventas":90,  "meta":150},
    {"nombre":"Sol", "ventas":180, "meta":150},
    {"nombre":"Paco","ventas":310, "meta":200}
]
`,
    csv: `
_CSV_DATA = """nombre,ventas,meta,zona
Ana,18500,15000,Norte
Luis,9200,15000,Sur
Sol,22000,15000,Norte
Paco,15000,15000,Sur
Mia,31000,20000,Norte"""
with open("datos_ventas.csv","w") as _f:
    _f.write(_CSV_DATA)
del _CSV_DATA, _f
`,
    csv_dirty: `
_CSV_DIRTY = """nombre,ventas,meta,zona
Ana,18500,15000,Norte
Luis,nueve mil,15000,Sur
Sol,22000,15000,Norte
,15000,15000,Sur
Mia,31000,20000,Norte"""
with open("datos_sucios.csv","w") as _f:
    _f.write(_CSV_DIRTY)
del _CSV_DIRTY, _f
`
  };

  async function initPyodide() {
    if (pyodideReady || pyodideLoading) return;
    pyodideLoading = true;
    _setAllStatus('Cargando Python...');
    try {
      pyodide = await loadPyodide();
      _setAllStatus('Cargando pandas...');
      await pyodide.loadPackage(['pandas', 'micropip']);
      pyodideReady = true;
      pyodideLoading = false;
      _setAllStatus('');
      for (const fn of pendingRuns) fn();
      pendingRuns.length = 0;
    } catch (e) {
      pyodideLoading = false;
      _setAllStatus('Error cargando Python');
      console.error('Pyodide init failed', e);
    }
  }

  function _setAllStatus(msg) {
    document.querySelectorAll('.py-editor-status').forEach(el => {
      el.textContent = msg;
      el.style.display = msg ? 'block' : 'none';
    });
  }

  function _buildUI(container) {
    const id = container.dataset.id || 'ed_' + Math.random().toString(36).slice(2, 8);
    const height = container.dataset.height || '300px';
    const readonly = container.dataset.readonly === 'true';
    const initialCode = container.textContent.trim();
    container.textContent = '';
    container.classList.add('py-editor-container');

    const statusBar = document.createElement('div');
    statusBar.className = 'py-editor-status';
    statusBar.style.display = 'none';
    container.appendChild(statusBar);

    const editorWrapper = document.createElement('div');
    editorWrapper.className = 'py-editor-code';
    editorWrapper.style.height = height;
    container.appendChild(editorWrapper);

    const toolbar = document.createElement('div');
    toolbar.className = 'py-editor-toolbar';

    const runBtn = document.createElement('button');
    runBtn.className = 'py-run-btn';
    runBtn.innerHTML = '<i class="fas fa-play"></i> Ejecutar';
    runBtn.onclick = () => _run(id);
    toolbar.appendChild(runBtn);

    if (!readonly) {
      const resetBtn = document.createElement('button');
      resetBtn.className = 'py-reset-btn';
      resetBtn.innerHTML = '<i class="fas fa-undo"></i> Reset';
      resetBtn.onclick = () => {
        editors[id].cm.setValue(initialCode);
        _clearOutput(id);
      };
      toolbar.appendChild(resetBtn);
    }

    const expandBtn = document.createElement('button');
    expandBtn.className = 'py-reset-btn';
    expandBtn.style.marginLeft = 'auto';
    expandBtn.innerHTML = '<i class="fas fa-expand"></i> Expandir';
    expandBtn.onclick = () => _toggleExpand(id);
    toolbar.appendChild(expandBtn);

    container.appendChild(toolbar);

    const outputEl = document.createElement('div');
    outputEl.className = 'py-editor-output';
    outputEl.id = 'output_' + id;
    container.appendChild(outputEl);

    const cm = CodeMirror(editorWrapper, {
      value: initialCode,
      mode: 'python',
      theme: 'material-darker',
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      lineWrapping: true,
      readOnly: readonly,
      inputStyle: 'contenteditable',
      extraKeys: {
        'Shift-Enter': () => _run(id),
        'Tab': (cm) => cm.replaceSelection('    ', 'end')
      }
    });

    editors[id] = { cm, container, outputEl, initialCode, preloads: container.dataset.preload || '' };
    setTimeout(() => cm.refresh(), 1);
    return id;
  }

  function _toggleExpand(id) {
    const ed = editors[id];
    if (!ed) return;
    const c = ed.container;
    const wrapper = c.querySelector('.py-editor-code');
    const btn = c.querySelector('.py-reset-btn[style*="margin-left"]');
    const isExpanded = c.classList.toggle('py-expanded');

    if (isExpanded) {
      ed._prevHeight = wrapper.style.height;
      wrapper.style.height = '';
      if (btn) btn.innerHTML = '<i class="fas fa-compress"></i> Colapsar';
    } else {
      wrapper.style.height = ed._prevHeight || (c.dataset.height || '300px');
      if (btn) btn.innerHTML = '<i class="fas fa-expand"></i> Expandir';
    }
    setTimeout(() => ed.cm.refresh(), 50);
  }

  function _clearOutput(id) {
    const ed = editors[id];
    if (ed) {
      ed.outputEl.textContent = '';
      ed.outputEl.classList.remove('has-error', 'has-output');
    }
  }

  async function _run(id) {
    const ed = editors[id];
    if (!ed) return;

    if (!pyodideReady) {
      initPyodide();
      pendingRuns.push(() => _run(id));
      return;
    }

    const btn = ed.container.querySelector('.py-run-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Ejecutando...';
    ed.outputEl.textContent = '';
    ed.outputEl.classList.remove('has-error', 'has-output');

    try {
      if (ed.preloads) {
        for (const key of ed.preloads.split(',')) {
          const k = key.trim();
          if (PRELOADS[k]) {
            await pyodide.runPythonAsync(PRELOADS[k]);
          }
        }
      }

      pyodide.runPython(`
import sys, io as _io
_capture = _io.StringIO()
sys.stdout = _capture
sys.stderr = _capture
`);

      await pyodide.runPythonAsync(ed.cm.getValue());

      const output = pyodide.runPython(`
_result = _capture.getvalue()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_result
`);

      if (output) {
        ed.outputEl.textContent = output;
        ed.outputEl.classList.add('has-output');
      } else {
        ed.outputEl.textContent = '(sin salida)';
        ed.outputEl.classList.add('has-output');
      }
    } catch (err) {
      pyodide.runPython('import sys; sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__');
      let msg = err.message || String(err);
      const lines = msg.split('\n');
      const meaningful = lines.filter(l => !l.startsWith('  File "<exec>"') && l.trim());
      ed.outputEl.textContent = meaningful.join('\n') || msg;
      ed.outputEl.classList.add('has-error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-play"></i> Ejecutar';
    }
  }

  function initAll() {
    document.querySelectorAll('.py-editor:not(.py-initialized)').forEach(el => {
      _buildUI(el);
      el.classList.add('py-initialized');
    });
  }

  function refreshAll() {
    Object.values(editors).forEach(ed => {
      ed.cm.refresh();
    });
  }

  function runEditor(id) { _run(id); }

  return { initAll, initPyodide, runEditor, refreshAll, editors };

})();
