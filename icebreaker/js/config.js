const FIREBASE_CONFIG = {
  apiKey: "AIzaSyAIy7unRM1HUk_fiYrC2fimXF0PfyQXxtE",
  authDomain: "udla-c4a0a.firebaseapp.com",
  databaseURL: "https://udla-c4a0a-default-rtdb.firebaseio.com",
  projectId: "udla-c4a0a",
  storageBucket: "udla-c4a0a.firebasestorage.app",
  messagingSenderId: "420164855749",
  appId: "1:420164855749:web:feec180044a2a3801b4dac"
};

const DEMO_MODE = false;

const SURVEY_BASE_URL = window.location.hostname === 'localhost'
  ? window.location.origin + window.location.pathname.replace(/\/[^/]*$/, '/')
  : 'https://cmosquerat.github.io/arca-diplomado/icebreaker/';
