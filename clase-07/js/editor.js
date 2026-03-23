/*  editor.js – Clase 07: Python editor with Pyodide + matplotlib support
 *  Extended from clase-06: adds matplotlib rendering + Titanic preloads
 *
 *  Usage:  <div class="py-editor" data-id="uniqueId" data-height="350px">
 *            print("hello")
 *          </div>
 *
 *  Attributes:
 *    data-id        unique identifier
 *    data-height    editor height (default 300px)
 *    data-readonly  "true" for read-only demos
 *    data-preload   comma-separated preload IDs (e.g. "titanic,mpl")
 */

const PyEditor = (() => {

  let pyodide = null;
  let pyodideReady = false;
  let pyodideLoading = false;
  const editors = {};
  const pendingRuns = [];

  // ── Embedded 30-row Titanic subset (avoids network dependency in practice) ──
  const _TITANIC_CSV = `PassengerId,Survived,Pclass,Sex,Age,Fare,Embarked
1,0,3,male,22.0,7.25,S
2,1,1,female,38.0,71.28,C
3,1,3,female,26.0,7.92,S
4,1,1,female,35.0,53.10,S
5,0,3,male,35.0,8.05,S
6,0,3,male,,8.46,Q
7,0,1,male,54.0,51.86,S
8,0,3,male,2.0,21.07,S
9,1,3,female,27.0,11.13,S
10,1,2,female,14.0,30.07,C
11,1,3,female,4.0,16.70,S
12,1,1,female,58.0,26.55,S
13,0,3,male,20.0,8.05,S
14,0,3,male,39.0,31.27,S
15,0,3,female,14.0,7.85,S
16,1,2,female,55.0,16.00,S
17,0,3,male,2.0,29.12,Q
18,1,2,male,,13.00,S
19,0,3,female,31.0,18.00,S
20,1,3,female,,7.22,C
21,0,2,male,35.0,26.00,S
22,1,2,male,34.0,13.00,S
23,1,3,female,15.0,8.03,Q
24,1,1,male,28.0,35.50,S
25,0,3,female,8.0,21.07,S
26,1,3,female,38.0,31.39,S
27,0,3,male,,7.22,C
28,0,1,male,19.0,263.00,S
29,1,3,female,,7.87,Q
30,0,3,male,,7.90,S`;

  const PRELOADS = {
    // ── Titanic CSV → virtual filesystem (for practice slides) ──
    titanic: `
_TC = """${_TITANIC_CSV}"""
with open("titanic.csv", "w") as _f:
    _f.write(_TC)
del _TC, _f
`,
    // ── matplotlib setup + helper ──
    mpl: `
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io as _mpl_io, base64 as _mpl_b64

def mostrar_grafica(fig=None):
    """Renderiza la figura activa como imagen en el output del editor."""
    if fig is None:
        fig = plt.gcf()
    buf = _mpl_io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = _mpl_b64.b64encode(buf.read()).decode()
    plt.close(fig)
    print(f'<img class="py-chart" src="data:image/png;base64,{b64}">')
`,
    // ── Titanic + matplotlib combo ──
    titanic_mpl: `
_TC = """${_TITANIC_CSV}"""
with open("titanic.csv", "w") as _f:
    _f.write(_TC)
del _TC, _f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io as _mpl_io, base64 as _mpl_b64

def mostrar_grafica(fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = _mpl_io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = _mpl_b64.b64encode(buf.read()).decode()
    plt.close(fig)
    print(f'<img class="py-chart" src="data:image/png;base64,{b64}">')
`,
    // ── Ventas de clase-06 (compatibilidad) ──
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
`
  };

  // ── Pyodide init ──────────────────────────────────────────────────────────

  async function initPyodide() {
    if (pyodideReady || pyodideLoading) return;
    pyodideLoading = true;
    _setAllStatus('Cargando Python…');
    try {
      pyodide = await loadPyodide();
      _setAllStatus('Cargando pandas + matplotlib…');
      await pyodide.loadPackage(['pandas', 'micropip', 'matplotlib']);
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

  // ── Build UI ──────────────────────────────────────────────────────────────

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
      resetBtn.onclick = () => { editors[id].cm.setValue(initialCode); _clearOutput(id); };
      toolbar.appendChild(resetBtn);

      const copyBtn = document.createElement('button');
      copyBtn.className = 'py-reset-btn';
      copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copiar';
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(editors[id].cm.getValue()).then(() => {
          copyBtn.innerHTML = '<i class="fas fa-check"></i> Copiado';
          setTimeout(() => { copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copiar'; }, 1200);
        });
      };
      toolbar.appendChild(copyBtn);
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
      ed.outputEl.innerHTML = '';
      ed.outputEl.className = 'py-editor-output';
    }
  }

  // ── Run ───────────────────────────────────────────────────────────────────

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
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Ejecutando…';
    ed.outputEl.innerHTML = '';
    ed.outputEl.className = 'py-editor-output';

    try {
      if (ed.preloads) {
        for (const key of ed.preloads.split(',')) {
          const k = key.trim();
          if (PRELOADS[k]) await pyodide.runPythonAsync(PRELOADS[k]);
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
        // ── Chart / HTML output ──────────────────────────────────────────
        if (/<img[\s>]/i.test(output)) {
          ed.outputEl.innerHTML = output;
          ed.outputEl.classList.add('has-output', 'has-chart');
        } else {
          ed.outputEl.textContent = output;
          ed.outputEl.classList.add('has-output');
        }
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

  // ── Public API ────────────────────────────────────────────────────────────

  function initAll() {
    document.querySelectorAll('.py-editor:not(.py-initialized)').forEach(el => {
      _buildUI(el);
      el.classList.add('py-initialized');
    });
  }

  function refreshAll() {
    Object.values(editors).forEach(ed => ed.cm.refresh());
  }

  function runEditor(id) { _run(id); }

  return { initAll, initPyodide, runEditor, refreshAll, editors };

})();
