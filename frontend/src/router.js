import { createRouter, createWebHistory } from 'vue-router';
import { initializeAuth, isAuthenticated, refreshProfile, useAuth } from './auth';
import LoginView from './views/LoginView.vue';
import DashboardView from './views/DashboardView.vue';
import AdminLogsView from './views/AdminLogsView.vue';

function getHomeRouteForRole(role) {
  return role === 'admin' ? { name: 'admin-logs' } : { name: 'dashboard' };
}

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/login',
    name: 'login',
    component: LoginView,
    meta: {
      guestOnly: true
    }
  },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: DashboardView,
    meta: {
      requiresAuth: true
    }
  },
  {
    path: '/admin/logs',
    name: 'admin-logs',
    component: AdminLogsView,
    meta: {
      requiresAuth: true,
      roles: ['admin']
    }
  }
];

export const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach(async (to) => {
  initializeAuth();
  const auth = useAuth();

  if (auth.token) {
    const user = await refreshProfile();
    if (!user && to.meta.requiresAuth) {
      return {
        name: 'login',
        query: { redirect: to.fullPath }
      };
    }
  }

  if (to.meta.requiresAuth && !isAuthenticated()) {
    return {
      name: 'login',
      query: { redirect: to.fullPath }
    };
  }

  if (to.meta.guestOnly && isAuthenticated()) {
    return getHomeRouteForRole(auth.user?.role);
  }

  if (to.name === 'dashboard' && auth.user?.role === 'admin') {
    return { name: 'admin-logs' };
  }

  if (to.meta.roles?.length && !to.meta.roles.includes(auth.user?.role)) {
    return getHomeRouteForRole(auth.user?.role);
  }

  return true;
});