import { reactive } from 'vue';
import { apiFetch, setAuthTokenGetter, setUnauthorizedHandler } from './services/api';

const STORAGE_KEY = 'traffic-console-auth';

const authState = reactive({
  token: null,
  user: null,
  ready: false
});

function persistAuth() {
  if (!authState.token || !authState.user) {
    localStorage.removeItem(STORAGE_KEY);
    return;
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    token: authState.token,
    user: authState.user
  }));
}

function loadAuth() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    authState.token = parsed.token || null;
    authState.user = parsed.user || null;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export function initializeAuth() {
  if (authState.ready) {
    return;
  }

  loadAuth();
  setAuthTokenGetter(() => authState.token);
  setUnauthorizedHandler(() => {
    clearAuth();
  });
  authState.ready = true;
}

export function useAuth() {
  return authState;
}

export function isAuthenticated() {
  return Boolean(authState.token && authState.user);
}

export function hasRole(role) {
  return authState.user?.role === role;
}

export function clearAuth() {
  authState.token = null;
  authState.user = null;
  persistAuth();
}

export async function login({ username, password }) {
  const result = await apiFetch('/api/auth/login', {
    method: 'POST',
    skipAuth: true,
    body: JSON.stringify({ username, password })
  });

  authState.token = result.access_token;
  authState.user = result.user;
  persistAuth();

  return result.user;
}

export async function refreshProfile() {
  if (!authState.token) {
    return null;
  }

  try {
    const user = await apiFetch('/api/auth/me');
    authState.user = user;
    persistAuth();
    return user;
  } catch {
    clearAuth();
    return null;
  }
}