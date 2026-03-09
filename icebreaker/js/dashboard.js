/* ===== PALETTE ===== */
const C = {
  red:     '#C82B40',
  dark:    '#6B1525',
  crimson: '#A31545',
  coral:   '#E85D75',
  rose:    '#F0A0B0',
  wine:    '#8B2040',
  blush:   '#D4707F',
  muted:   '#7a6b71',
  text:    '#f5f0f2',
  card:    '#1a1215',
  bg:      '#0d0709'
};

const PALETTE = [C.red, C.coral, C.crimson, C.rose, C.dark, C.wine, C.blush, C.muted];

/* ===== QUESTION METADATA ===== */
const Q_META = {
  q1_role: [
    'Planificación de demanda', 'Gestión comercial y ventas',
    'Operaciones y logística', 'Análisis financiero',
    'Gestión de personas', 'Soporte tecnológico', 'Dirección y estrategia'
  ],
  q2_transformations: [
    'Esta es mi primera', 'He vivido 1–2 cambios importantes',
    'He visto 3 o más transformaciones', 'Llevo tantas que ya perdí la cuenta'
  ],
  q3_hours: [
    'Menos de 1 hora', '1–3 horas', '3–5 horas',
    'Más de 5 horas', 'Prefiero no pensarlo'
  ],
  q4_tool: [
    'Excel', 'Google Sheets', 'SAP / ERP',
    'Power BI / Tableau', 'Correo y WhatsApp', 'Otra'
  ],
  q5_urgent: [
    'Lo tengo al alcance en minutos', 'Debo pedirlo y esperar horas o días',
    'Lo calculo manualmente con lo que tengo', 'Decido con intuición y experiencia'
  ],
  q6_expectations: [
    'Automatizar un reporte que hoy toma horas',
    'Auditar lo que la IA me entrega',
    'Construir análisis sin depender de TI',
    'Hablar el idioma de los equipos técnicos',
    'Liderar proyectos de datos en mi área'
  ],
  q7_ai: [
    'No las he usado', 'Las he probado por curiosidad',
    'Las uso ocasionalmente en el trabajo', 'Son parte de mi flujo diario'
  ]
};

const HOURS_MAP = {
  'Menos de 1 hora': 0.5,
  '1–3 horas': 2,
  '3–5 horas': 4,
  'Más de 5 horas': 7,
  'Prefiero no pensarlo': 5
};

/* ===== STATE ===== */
let responses = [];
let charts = {};
let wordList = [];
let sessionId = '';
let counterEl, hoursEl;

/* ===== CHART.JS DEFAULTS ===== */
Chart.defaults.color = C.text;
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 12;
Chart.defaults.animation.duration = 600;
Chart.defaults.animation.easing = 'easeOutQuart';

/* ===== SESSION ===== */
function getDefaultSession() {
  const d = new Date();
  return d.getFullYear().toString() +
    String(d.getMonth() + 1).padStart(2, '0') +
    String(d.getDate()).padStart(2, '0');
}

function startSession() {
  const input = document.getElementById('sessionInput');
  sessionId = (input.value.trim() || getDefaultSession());
  input.value = sessionId;
  generateQR();
  showDashboard();
  listenForResponses();
}

/* ===== QR ===== */
function generateQR() {
  const surveyUrl = SURVEY_BASE_URL + 'index.html?s=' + sessionId;
  document.getElementById('surveyUrl').textContent = surveyUrl;

  const qrContainer = document.getElementById('qrCode');
  qrContainer.innerHTML = '';

  try {
    const qr = qrcode(0, 'M');
    qr.addData(surveyUrl);
    qr.make();

    const size = Math.min(280, window.innerWidth - 80);
    const moduleCount = qr.getModuleCount();
    const cellSize = Math.floor(size / moduleCount);
    const totalSize = cellSize * moduleCount;

    const canvas = document.createElement('canvas');
    canvas.width = totalSize;
    canvas.height = totalSize;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, totalSize, totalSize);

    ctx.fillStyle = C.red;
    for (let row = 0; row < moduleCount; row++) {
      for (let col = 0; col < moduleCount; col++) {
        if (qr.isDark(row, col)) {
          ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
        }
      }
    }

    qrContainer.appendChild(canvas);
  } catch (e) {
    console.error('QR generation failed:', e);
    qrContainer.innerHTML = '<p style="color:var(--text-muted)">QR no disponible</p>';
  }
}

/* ===== DASHBOARD VISIBILITY ===== */
function showDashboard() {
  document.getElementById('metricHero').style.display = 'block';
  document.getElementById('dashboardGrid').style.display = 'grid';
}

/* ===== CHART CREATION ===== */
function initCharts() {
  /* Q1 — Doughnut */
  charts.roles = new Chart(document.getElementById('chartRoles'), {
    type: 'doughnut',
    data: {
      labels: Q_META.q1_role.map(l => l.length > 22 ? l.slice(0, 20) + '…' : l),
      datasets: [{ data: new Array(7).fill(0), backgroundColor: PALETTE, borderWidth: 0 }]
    },
    options: {
      responsive: true,
      cutout: '55%',
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 } } }
      }
    }
  });

  /* Q2 — Polar Area */
  charts.transformations = new Chart(document.getElementById('chartTransformations'), {
    type: 'polarArea',
    data: {
      labels: ['Primera', '1–2 cambios', '3+ cambios', 'Incontables'],
      datasets: [{
        data: new Array(4).fill(0),
        backgroundColor: [
          'rgba(240,160,176,0.55)',
          'rgba(232,93,117,0.55)',
          'rgba(200,43,64,0.55)',
          'rgba(107,21,37,0.55)'
        ],
        borderColor: [C.rose, C.coral, C.red, C.dark],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          beginAtZero: true,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      },
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 }, padding: 10 } }
      }
    }
  });

  /* Q3 — Stacked percentage bar (single horizontal bar, segmented) */
  charts.hours = new Chart(document.getElementById('chartHours'), {
    type: 'bar',
    data: {
      labels: [''],
      datasets: Q_META.q3_hours.map((label, i) => ({
        label: label,
        data: [0],
        backgroundColor: ['#4ade80', '#facc15', C.coral, C.red, C.dark][i],
        borderWidth: 0,
        borderSkipped: false
      }))
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: {
          stacked: true,
          display: false
        },
        y: {
          stacked: true,
          display: false
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { font: { size: 10 }, boxWidth: 10, padding: 8 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ctx.dataset.label + ': ' + ctx.raw + ' personas'
          }
        }
      },
      barPercentage: 1,
      categoryPercentage: 1
    }
  });

  /* Q4 — Horizontal bars (the ONE bar chart we keep) */
  charts.tools = new Chart(document.getElementById('chartTools'), {
    type: 'bar',
    data: {
      labels: Q_META.q4_tool.map(l => l.length > 25 ? l.slice(0, 23) + '…' : l),
      datasets: [{
        data: new Array(6).fill(0),
        backgroundColor: [C.red, C.coral, C.crimson, C.wine, C.blush, C.muted],
        borderWidth: 0,
        borderRadius: 4,
        barThickness: 28
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      scales: {
        x: {
          beginAtZero: true,
          ticks: { stepSize: 1, font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          ticks: { font: { size: 11 }, color: '#b8a8af' },
          grid: { display: false }
        }
      },
      plugins: { legend: { display: false } }
    }
  });

  /* Q5 — Doughnut (thicker ring, distinct from Q1) */
  charts.urgent = new Chart(document.getElementById('chartUrgent'), {
    type: 'doughnut',
    data: {
      labels: [
        'En minutos',
        'Esperar horas/días',
        'Cálculo manual',
        'Intuición'
      ],
      datasets: [{
        data: new Array(4).fill(0),
        backgroundColor: ['#4ade80', C.coral, '#facc15', C.muted],
        borderWidth: 0,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      cutout: '40%',
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 }, padding: 10 } }
      }
    }
  });

  /* Q6 — Radar */
  charts.expectations = new Chart(document.getElementById('chartExpectations'), {
    type: 'radar',
    data: {
      labels: ['Automatizar\nreportes', 'Auditar\nIA', 'Independencia\nde TI', 'Idioma\ntécnico', 'Liderar\nproyectos'],
      datasets: [{
        data: new Array(5).fill(0),
        backgroundColor: 'rgba(200, 43, 64, 0.2)',
        borderColor: C.red,
        borderWidth: 2,
        pointBackgroundColor: C.red,
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          beginAtZero: true,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.08)' },
          pointLabels: { font: { size: 10 }, color: C.text }
        }
      },
      plugins: { legend: { display: false } }
    }
  });

  /* Q7 — Vertical bars (progression from "never" to "daily") */
  charts.ai = new Chart(document.getElementById('chartAI'), {
    type: 'bar',
    data: {
      labels: ['No las\nhe usado', 'Por\ncuriosidad', 'Ocasional\nen trabajo', 'Flujo\ndiario'],
      datasets: [{
        data: new Array(4).fill(0),
        backgroundColor: [
          'rgba(122,107,113,0.7)',
          'rgba(240,160,176,0.7)',
          'rgba(232,93,117,0.7)',
          'rgba(200,43,64,0.9)'
        ],
        borderColor: [C.muted, C.rose, C.coral, C.red],
        borderWidth: 2,
        borderRadius: 6,
        barPercentage: 0.7
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          ticks: { font: { size: 10 }, color: '#b8a8af', maxRotation: 0 },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1, font: { size: 11 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

/* ===== UPDATE CHARTS ===== */
function tally(key, labels) {
  const counts = new Array(labels.length).fill(0);
  responses.forEach(r => {
    const val = r[key];
    if (Array.isArray(val)) {
      val.forEach(v => {
        const idx = labels.indexOf(v);
        if (idx >= 0) counts[idx]++;
      });
    } else {
      const idx = labels.indexOf(val);
      if (idx >= 0) counts[idx]++;
    }
  });
  return counts;
}

function updateAll() {
  const n = responses.length;
  animateCounter(counterEl, n);

  charts.roles.data.datasets[0].data = tally('q1_role', Q_META.q1_role);
  charts.roles.update();

  charts.transformations.data.datasets[0].data = tally('q2_transformations', Q_META.q2_transformations);
  charts.transformations.update();

  const hourCounts = tally('q3_hours', Q_META.q3_hours);
  hourCounts.forEach((val, i) => { charts.hours.data.datasets[i].data = [val]; });
  charts.hours.update();

  charts.tools.data.datasets[0].data = tally('q4_tool', Q_META.q4_tool);
  charts.tools.update();

  charts.urgent.data.datasets[0].data = tally('q5_urgent', Q_META.q5_urgent);
  charts.urgent.update();

  charts.expectations.data.datasets[0].data = tally('q6_expectations', Q_META.q6_expectations);
  charts.expectations.update();

  charts.ai.data.datasets[0].data = tally('q7_ai', Q_META.q7_ai);
  charts.ai.update();

  updateHoursMetric();
  updateWordCloud();
  pulseCards();
}

function updateHoursMetric() {
  let total = 0;
  responses.forEach(r => {
    const h = HOURS_MAP[r.q3_hours];
    if (h !== undefined) total += h;
  });
  animateCounter(hoursEl, Math.round(total));
}

function updateWordCloud() {
  const freq = {};
  responses.forEach(r => {
    const text = (r.q8_automate || '').toLowerCase().trim();
    if (!text) return;
    const words = text.split(/\s+/);
    words.forEach(w => {
      w = w.replace(/[^a-záéíóúñü]/gi, '');
      if (w.length < 3) return;
      freq[w] = (freq[w] || 0) + 1;
    });
  });

  wordList = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 60)
    .map(([word, count]) => [word, count * 8 + 12]);

  const canvas = document.getElementById('wordcloudCanvas');
  if (wordList.length === 0) return;

  try {
    WordCloud(canvas, {
      list: wordList,
      gridSize: 10,
      weightFactor: 1,
      fontFamily: 'Inter, sans-serif',
      fontWeight: 700,
      color: () => PALETTE[Math.floor(Math.random() * PALETTE.length)],
      backgroundColor: 'transparent',
      rotateRatio: 0.3,
      minRotation: -0.3,
      maxRotation: 0.3,
      shuffle: true,
      drawOutOfBound: false
    });
  } catch (e) {
    console.warn('WordCloud render skipped:', e);
  }
}

/* ===== ANIMATIONS ===== */
function animateCounter(el, target) {
  if (!el) return;
  const current = parseInt(el.textContent) || 0;
  if (current === target) return;

  const duration = 600;
  const start = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(current + (target - current) * ease);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function pulseCards() {
  document.querySelectorAll('.chart-card').forEach(card => {
    card.classList.add('pulse');
    setTimeout(() => card.classList.remove('pulse'), 1000);
  });
}

/* ===== FIREBASE LISTENER ===== */
function listenForResponses() {
  if (typeof DEMO_MODE !== 'undefined' && DEMO_MODE) {
    listenDemo();
    return;
  }

  firebase.initializeApp(FIREBASE_CONFIG);
  const db = firebase.database();
  const refPath = 'sessions/' + sessionId + '/responses';

  db.ref(refPath).on('child_added', snapshot => {
    const data = snapshot.val();
    if (data && data.timestamp) {
      responses.push(data);
      updateAll();
    }
  });
}

/* ===== DEMO MODE ===== */
function listenDemo() {
  const demoData = generateDemoResponses(8);
  let i = 0;

  function addNext() {
    if (i >= demoData.length) return;
    responses.push(demoData[i]);
    i++;
    updateAll();
    setTimeout(addNext, 800 + Math.random() * 1200);
  }

  setTimeout(addNext, 1500);
}

function generateDemoResponses(n) {
  const pick = arr => arr[Math.floor(Math.random() * arr.length)];
  const pickN = (arr, max) => {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, 1 + Math.floor(Math.random() * max));
  };

  const automateIdeas = [
    'consolidar reportes de ventas',
    'actualizar inventario diario',
    'cruce de datos entre SAP y Excel',
    'reporte de cierre mensual',
    'seguimiento de rutas',
    'conciliación bancaria',
    'proyecciones de demanda',
    'control de devoluciones',
    'dashboard de KPIs',
    'análisis de cartera vencida',
    'forecast semanal',
    'reportes de productividad'
  ];

  return Array.from({ length: n }, () => ({
    q1_role: pick(Q_META.q1_role),
    q2_transformations: pick(Q_META.q2_transformations),
    q3_hours: pick(Q_META.q3_hours),
    q4_tool: pick(Q_META.q4_tool),
    q5_urgent: pick(Q_META.q5_urgent),
    q6_expectations: pickN(Q_META.q6_expectations, 3),
    q7_ai: pick(Q_META.q7_ai),
    q8_automate: pick(automateIdeas),
    timestamp: Date.now()
  }));
}

/* ===== INIT ===== */
counterEl = document.getElementById('counter');
hoursEl = document.getElementById('hoursNumber');

document.getElementById('sessionInput').value = getDefaultSession();
initCharts();
