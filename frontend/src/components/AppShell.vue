<script setup>
import { computed } from 'vue';
import { RouterLink, useRouter } from 'vue-router';
import { clearAuth, useAuth } from '../auth';

const props = defineProps({
  eyebrow: {
    type: String,
    default: 'Edge Traffic Control'
  },
  title: {
    type: String,
    required: true
  },
  subtitle: {
    type: String,
    default: ''
  }
});

const auth = useAuth();
const router = useRouter();

const showDashboardLink = computed(() => auth.user?.role !== 'admin');
const showAdminLink = computed(() => auth.user?.role === 'admin');

async function handleLogout() {
  clearAuth();
  await router.push({ name: 'login' });
}
</script>

<template>
  <div class="app-shell">
    <div class="page-shell">
      <header class="topbar">
        <div class="brand-block">
          <div class="brand-mark">ET</div>
          <div>
            <p class="eyebrow">{{ eyebrow }}</p>
            <h1>{{ title }}</h1>
          </div>
        </div>

        <div class="topbar-actions">
          <RouterLink v-if="showDashboardLink" class="nav-link" :to="{ name: 'dashboard' }">实时监控</RouterLink>
          <RouterLink v-if="showAdminLink" class="nav-link" :to="{ name: 'admin-logs' }">管理员日志</RouterLink>
          <div class="user-pill">
            <strong>{{ auth.user?.display_name || 'Unknown' }}</strong>
            <span class="muted">{{ auth.user?.role || 'guest' }}</span>
          </div>
          <button class="ghost-btn" type="button" @click="handleLogout">退出登录</button>
        </div>
      </header>

      <section class="hero-header">
        <div>
          <p class="eyebrow">{{ eyebrow }}</p>
          <h2>{{ title }}</h2>
          <p v-if="subtitle">{{ subtitle }}</p>
        </div>

        <slot name="hero-actions" />
      </section>

      <slot />
    </div>
  </div>
</template>