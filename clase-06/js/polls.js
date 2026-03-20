/*  polls.js – Firebase Realtime polls for clase-06
 *  Shared by presentacion.html (display results) and votar.html (submit votes).
 *
 *  Firebase path:  clase06/{sessionId}/polls/{pollId}/responses
 *  Active poll:    clase06/{sessionId}/activePoll
 */

const Polls = (() => {
  let db = null;
  let sessionId = null;
  let initialized = false;
  const charts = {};

  const POLL_DEFS = {
    comfort: {
      question: '¿Qué imprime este código?',
      code: 'resultado = clasificar_vendedor(\n  {"nombre":"Sol", "ventas":180, "meta":150}\n)\nprint(resultado)',
      options: ['"Estrella"', '"Cumple"', '"Bajo"', 'Error'],
      type: 'bar',
      colors: ['#C82B40', '#22c55e', '#2563EB', '#EA580C']
    },
    error_quiz: {
      question: '¿Qué tipo de error tiene este código?',
      options: ['Sintaxis', 'Runtime', 'Lógico'],
      type: 'bar',
      colors: ['#C82B40', '#EA580C', '#16A34A']
    },
    prediction: {
      question: '¿Qué imprime este código?',
      code: 'from math import sqrt\nprint(sqrt(16))',
      options: ['4.0', '4', '16', 'Error'],
      type: 'bar',
      colors: ['#22c55e', '#C82B40', '#2563EB', '#EA580C']
    },
    useful: {
      question: '¿Qué número imprime?',
      code: 'import pandas as pd\n# datos_ventas.csv: 10 vendedores\ndf = pd.read_csv("datos_ventas.csv")\nprint(df.shape[0])',
      options: ['10', '11', '4', '(10, 4)'],
      type: 'bar',
      colors: ['#22c55e', '#C82B40', '#2563EB', '#EA580C']
    }
  };

  function init() {
    if (initialized) return;
    sessionId = 'live';
    firebase.initializeApp(FIREBASE_CONFIG);
    db = firebase.database();
    initialized = true;
  }

  function getSessionId() { return sessionId; }
  function getPollDef(pollId) { return POLL_DEFS[pollId]; }
  function getDb() { return db; }

  // --- PRESENTER SIDE: listen for results ---

  function _recount(snap, counts, def) {
    def.options.forEach(o => counts[o] = 0);
    const data = snap.val();
    if (data) {
      Object.values(data).forEach(v => {
        if (v && v.answer && counts.hasOwnProperty(v.answer)) {
          counts[v.answer]++;
        }
      });
    }
  }

  function listenPoll(pollId, canvasOrCallback) {
    if (!db) return;
    const def = POLL_DEFS[pollId];
    if (!def) return;

    const counts = {};
    def.options.forEach(o => counts[o] = 0);

    const ref = db.ref(`clase06/${sessionId}/polls/${pollId}/responses`);

    if (typeof canvasOrCallback === 'function') {
      ref.on('value', snap => {
        _recount(snap, counts, def);
        canvasOrCallback(counts, def);
      });
      return;
    }

    const canvas = typeof canvasOrCallback === 'string'
      ? document.getElementById(canvasOrCallback)
      : canvasOrCallback;
    if (!canvas) return;

    const chartConfig = _makeChartConfig(def, counts);
    const chart = new Chart(canvas.getContext('2d'), chartConfig);
    charts[pollId] = chart;

    ref.on('value', snap => {
      _recount(snap, counts, def);
      chart.data.datasets[0].data = def.options.map(o => counts[o]);
      chart.update();

      const totalEl = canvas.closest('.poll-container')?.querySelector('.poll-total');
      if (totalEl) {
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        totalEl.textContent = total + ' votos';
      }
    });
  }

  function _makeChartConfig(def, counts) {
    const data = def.options.map(o => counts[o] || 0);
    const base = {
      data: {
        labels: def.options,
        datasets: [{
          data,
          backgroundColor: def.colors,
          borderWidth: 0,
          borderRadius: def.type === 'bar' ? 6 : 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: def.type === 'doughnut', position: 'bottom',
            labels: { color: '#6B7280', font: { family: 'Fira Sans', size: 13 } }
          }
        }
      }
    };

    if (def.type === 'bar') {
      base.type = 'bar';
      base.options.scales = {
        y: { beginAtZero: true, ticks: { stepSize: 1, color: '#6B7280', font: { family: 'Fira Sans' } },
             grid: { color: 'rgba(0,0,0,0.06)' } },
        x: { ticks: { color: '#2D2D2D', font: { family: 'Fira Sans', weight: 600, size: 14 } },
             grid: { display: false } }
      };
    } else {
      base.type = 'doughnut';
      base.options.cutout = '55%';
    }

    return base;
  }

  // --- VOTER SIDE: submit vote ---

  function vote(pollId, answer) {
    if (!db) return Promise.reject('No DB');
    return db.ref(`clase06/${sessionId}/polls/${pollId}/responses`).push({
      answer,
      ts: Date.now()
    });
  }

  function hasVoted(pollId) {
    return localStorage.getItem(`voted_${pollId}`) === '1';
  }

  function markVoted(pollId) {
    localStorage.setItem(`voted_${pollId}`, '1');
  }

  // --- ACTIVE POLL CONTROL (presenter sets, voter listens) ---

  function resetPoll(pollId) {
    if (!db) return Promise.reject('No DB');
    return db.ref(`clase06/${sessionId}/polls/${pollId}/responses`).remove().then(() => {
      localStorage.removeItem(`voted_${pollId}`);
    });
  }

  function setActivePoll(pollId) {
    if (!db) return;
    db.ref(`clase06/${sessionId}/activePoll`).set(pollId || null);
  }

  function onActivePollChange(callback) {
    if (!db) return;
    db.ref(`clase06/${sessionId}/activePoll`).on('value', snap => {
      callback(snap.val());
    });
  }

  return {
    init, getSessionId, getPollDef, getDb,
    listenPoll, vote, hasVoted, markVoted,
    setActivePoll, onActivePollChange, resetPoll,
    POLL_DEFS
  };
})();
