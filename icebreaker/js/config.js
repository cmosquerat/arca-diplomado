/*
 * Firebase Configuration
 *
 * 1. Go to https://console.firebase.google.com/
 * 2. Create a new project (or use an existing one)
 * 3. Go to Project Settings > General > Your apps > Add web app
 * 4. Copy the firebaseConfig object and paste it below
 * 5. Go to Realtime Database > Create Database > Start in test mode
 * 6. Rename this file to config.js (remove .example)
 */

const FIREBASE_CONFIG = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "000000000000",
  appId: "YOUR_APP_ID"
};

const DEMO_MODE = true;

const SURVEY_BASE_URL = window.location.hostname === 'localhost'
  ? window.location.origin + window.location.pathname.replace(/\/[^/]*$/, '/')
  : 'https://cmosquerat.github.io/arca-diplomado/icebreaker/';
