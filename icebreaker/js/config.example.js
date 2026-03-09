// Copy this file to config.js and fill in your Firebase credentials
// config.js is gitignored — never commit real keys

const FIREBASE_CONFIG = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "000000000000",
  appId: "1:000000000000:web:0000000000000000000000"
};

const DEMO_MODE = true;

const SURVEY_BASE_URL = window.location.hostname === 'localhost'
  ? window.location.origin + window.location.pathname.replace(/\/[^/]*$/, '/')
  : 'https://YOUR_USERNAME.github.io/YOUR_REPO/icebreaker/';
