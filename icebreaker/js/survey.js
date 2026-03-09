const QUESTIONS = [
  {
    id: 'q1_role',
    number: 1,
    title: '¿Cuál es tu responsabilidad principal en la cadena de valor?',
    type: 'single',
    options: [
      'Planificación de demanda',
      'Gestión comercial y ventas',
      'Operaciones y logística',
      'Análisis financiero',
      'Gestión de personas',
      'Soporte tecnológico',
      'Dirección y estrategia'
    ]
  },
  {
    id: 'q2_transformations',
    number: 2,
    title: '¿Cuántas transformaciones tecnológicas has vivido en la empresa?',
    type: 'single',
    options: [
      'Esta es mi primera',
      'He vivido 1–2 cambios importantes',
      'He visto 3 o más transformaciones',
      'Llevo tantas que ya perdí la cuenta'
    ]
  },
  {
    id: 'q3_hours',
    number: 3,
    title: '¿Cuántas horas a la semana inviertes en tareas repetitivas con datos?',
    subtitle: 'Copiar, pegar, formatear, consolidar manualmente…',
    type: 'single',
    options: [
      'Menos de 1 hora',
      '1–3 horas',
      '3–5 horas',
      'Más de 5 horas'
    ]
  },
  {
    id: 'q4_tool',
    number: 4,
    title: '¿Con qué herramienta tomas decisiones basadas en datos hoy?',
    type: 'single',
    options: [
      'Excel',
      'Google Sheets',
      'SAP / ERP',
      'Power BI / Tableau',
      'Correo y WhatsApp',
      'Otra'
    ]
  },
  {
    id: 'q5_urgent',
    number: 5,
    title: '¿Qué tan rápido accedes al dato que necesitas para decidir?',
    type: 'single',
    options: [
      'En minutos, lo tengo disponible',
      'En horas, debo consolidarlo',
      'Depende del tipo de dato',
      'Generalmente uso mi criterio y experiencia'
    ]
  },
  {
    id: 'q6_expectations',
    number: 6,
    title: '¿Qué resultado concreto haría que este diplomado valga cada hora invertida?',
    subtitle: 'Selecciona hasta 3 opciones.',
    type: 'multi',
    maxSelect: 3,
    options: [
      'Automatizar un reporte que hoy toma horas',
      'Auditar lo que la IA me entrega',
      'Construir análisis sin depender de TI',
      'Hablar el idioma de los equipos técnicos',
      'Liderar proyectos de datos en mi área'
    ]
  },
  {
    id: 'q7_ai',
    number: 7,
    title: '¿Cómo describes tu relación actual con herramientas de IA?',
    subtitle: 'ChatGPT, Copilot, Gemini, etc.',
    type: 'single',
    options: [
      'No las he usado',
      'Las he probado por curiosidad',
      'Las uso ocasionalmente en el trabajo',
      'Son parte de mi flujo diario'
    ]
  },
  {
    id: 'q8_automate',
    number: 8,
    title: 'Si pudieras automatizar UNA sola tarea de tu día a día, ¿cuál sería?',
    subtitle: 'Respuesta corta — lo primero que te venga a la mente.',
    type: 'text',
    placeholder: 'Ej: consolidar reportes de ventas semanales'
  }
];

const TOTAL = QUESTIONS.length;
let currentIndex = -1; // -1 = welcome screen
let answers = {};

const wrapper = document.getElementById('surveyWrapper');
const progressBar = document.getElementById('progress');

function getSessionId() {
  const params = new URLSearchParams(window.location.search);
  return params.get('s') || new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

const SESSION_ID = getSessionId();

function alreadySubmitted() {
  return localStorage.getItem('survey_done_' + SESSION_ID) === '1';
}

function markSubmitted() {
  localStorage.setItem('survey_done_' + SESSION_ID, '1');
}

function updateProgress() {
  const pct = currentIndex < 0 ? 0 : ((currentIndex + 1) / (TOTAL + 1)) * 100;
  progressBar.style.width = pct + '%';
}

function buildWelcome() {
  const div = document.createElement('div');
  div.className = 'question-slide';
  div.dataset.index = '-1';
  div.innerHTML = `
    <div class="screen-center">
      <div class="logos">
        <img src="img/udla_white.png" alt="UDLA">
        <img src="img/arca_white.png" alt="Arca Continental">
      </div>
      <h1>Diplomado en <span>Data Science</span> Aplicada con Python</h1>
      <p>Antes de empezar, queremos conocer al equipo.
         Son 8 preguntas rápidas — menos de 2 minutos.</p>
      <button class="btn btn-primary" onclick="goTo(0)">Comenzar</button>
    </div>
  `;
  return div;
}

function buildThankYou() {
  const div = document.createElement('div');
  div.className = 'question-slide';
  div.dataset.index = 'done';
  div.innerHTML = `
    <div class="screen-center">
      <div class="thank-you-icon">✓</div>
      <h1>Respuestas <span>registradas</span></h1>
      <p>Los resultados aparecen en tiempo real en la pantalla del instructor.
         Gracias por participar — empecemos.</p>
    </div>
  `;
  return div;
}

function buildQuestion(q) {
  const div = document.createElement('div');
  div.className = 'question-slide';
  div.dataset.index = q.number - 1;

  let body = '';

  if (q.type === 'single') {
    body = `<div class="options-grid">
      ${q.options.map((opt, i) => `
        <button class="option-btn" data-qid="${q.id}" data-value="${opt}" onclick="selectSingle(this)">
          <span class="option-indicator"></span>
          <span>${opt}</span>
        </button>
      `).join('')}
    </div>`;
  } else if (q.type === 'multi') {
    body = `
      <p class="multi-hint">Máximo ${q.maxSelect} opciones</p>
      <div class="options-grid">
        ${q.options.map(opt => `
          <button class="option-btn multi" data-qid="${q.id}" data-value="${opt}" onclick="selectMulti(this, ${q.maxSelect})">
            <span class="option-indicator"></span>
            <span>${opt}</span>
          </button>
        `).join('')}
      </div>`;
  } else if (q.type === 'text') {
    body = `
      <input type="text" class="text-input" id="input-${q.id}"
             placeholder="${q.placeholder || ''}" maxlength="80"
             onkeydown="if(event.key==='Enter') submitText('${q.id}')">
    `;
  }

  const isLast = q.number === TOTAL;
  const nextLabel = isLast ? 'Enviar' : 'Siguiente';
  const nextAction = isLast ? 'submitSurvey()' : `goTo(${q.number})`;

  div.innerHTML = `
    <span class="question-number">Pregunta ${q.number} de ${TOTAL}</span>
    <h2 class="question-text">${q.title}</h2>
    ${q.subtitle ? `<p class="question-subtitle">${q.subtitle}</p>` : ''}
    ${body}
    <div class="nav-row">
      <button class="btn btn-ghost" onclick="goTo(${q.number - 2})" ${q.number === 1 ? 'style="visibility:hidden"' : ''}>Atrás</button>
      <button class="btn btn-primary" id="next-${q.id}" onclick="${nextAction}" ${q.type !== 'text' ? 'disabled' : ''}>${nextLabel}</button>
    </div>
  `;
  return div;
}

function selectSingle(btn) {
  const qid = btn.dataset.qid;
  const container = btn.closest('.options-grid');
  container.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  answers[qid] = btn.dataset.value;

  const nextBtn = document.getElementById('next-' + qid);
  if (nextBtn) nextBtn.disabled = false;

  setTimeout(() => {
    if (nextBtn) nextBtn.click();
  }, 400);
}

function selectMulti(btn, max) {
  const qid = btn.dataset.qid;
  const container = btn.closest('.options-grid');
  const selected = container.querySelectorAll('.option-btn.selected');

  if (btn.classList.contains('selected')) {
    btn.classList.remove('selected');
  } else if (selected.length < max) {
    btn.classList.add('selected');
  }

  const values = Array.from(container.querySelectorAll('.option-btn.selected'))
    .map(b => b.dataset.value);
  answers[qid] = values;

  const nextBtn = document.getElementById('next-' + qid);
  if (nextBtn) nextBtn.disabled = values.length === 0;
}

function submitText(qid) {
  const input = document.getElementById('input-' + qid);
  if (input && input.value.trim()) {
    answers[qid] = input.value.trim();
    const q = QUESTIONS.find(q => q.id === qid);
    if (q.number === TOTAL) {
      submitSurvey();
    } else {
      goTo(q.number);
    }
  }
}

function goTo(index) {
  if (index >= 0) {
    const q = QUESTIONS[index];
    if (q && q.type === 'text') {
      const input = document.getElementById('input-' + q.id);
      if (input && input.value.trim()) {
        answers[q.id] = input.value.trim();
      }
    }
  }

  const slides = wrapper.querySelectorAll('.question-slide');
  slides.forEach(s => {
    if (s.classList.contains('active')) {
      s.classList.remove('active');
      s.classList.add('exit-left');
      setTimeout(() => s.classList.remove('exit-left'), 500);
    }
  });

  currentIndex = index;
  updateProgress();

  setTimeout(() => {
    let target;
    if (index === -1) {
      target = wrapper.querySelector('[data-index="-1"]');
    } else if (index >= TOTAL) {
      target = wrapper.querySelector('[data-index="done"]');
    } else {
      target = wrapper.querySelector(`[data-index="${index}"]`);
    }
    if (target) {
      target.classList.add('active');
      const input = target.querySelector('.text-input');
      if (input) setTimeout(() => input.focus(), 300);
    }
  }, 80);
}

async function submitSurvey() {
  const q8 = QUESTIONS[TOTAL - 1];
  if (q8.type === 'text' && !answers[q8.id]) {
    const input = document.getElementById('input-' + q8.id);
    if (input && input.value.trim()) {
      answers[q8.id] = input.value.trim();
    }
  }

  const unanswered = QUESTIONS.filter(q => !answers[q.id] || (Array.isArray(answers[q.id]) && answers[q.id].length === 0));
  if (unanswered.length > 0) {
    goTo(unanswered[0].number - 1);
    return;
  }

  answers.timestamp = Date.now();

  try {
    if (typeof DEMO_MODE !== 'undefined' && DEMO_MODE) {
      console.log('DEMO — response:', answers);
    } else {
      firebase.initializeApp(FIREBASE_CONFIG);
      const db = firebase.database();
      await db.ref('sessions/' + SESSION_ID + '/responses').push(answers);
    }
    markSubmitted();
  } catch (err) {
    console.error('Firebase write error:', err);
  }

  currentIndex = TOTAL;
  updateProgress();
  goTo(TOTAL);
}

function init() {
  if (alreadySubmitted()) {
    wrapper.innerHTML = '';
    const done = buildThankYou();
    done.classList.add('active');
    wrapper.appendChild(done);
    progressBar.style.width = '100%';
    return;
  }

  wrapper.innerHTML = '';
  wrapper.appendChild(buildWelcome());
  QUESTIONS.forEach(q => wrapper.appendChild(buildQuestion(q)));
  wrapper.appendChild(buildThankYou());

  goTo(-1);
}

init();
