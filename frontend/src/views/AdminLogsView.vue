<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import AppShell from '../components/AppShell.vue';
import { apiFetch } from '../services/api';

const logs = ref([]);
const isLoading = ref(true);
const isRefreshing = ref(false);
const errorMessage = ref('');

let refreshTimer = null;

const totalLogs = computed(() => logs.value.length);
const errorCount = computed(() => logs.value.filter((item) => item.level === 'error').length);
const warnCount = computed(() => logs.value.filter((item) => item.level === 'warn').length);

async function loadLogs(showLoader = false) {
  if (showLoader) {
    isLoading.value = true;
  } else {
    isRefreshing.value = true;
  }

  try {
    const response = await apiFetch('/api/admin/logs?limit=120');
    logs.value = response.logs || [];
    errorMessage.value = '';
  } catch (error) {
    errorMessage.value = error.message || '无法获取管理员日志。';
  } finally {
    isLoading.value = false;
    isRefreshing.value = false;
  }
}

onMounted(async () => {
  await loadLogs(true);
  refreshTimer = window.setInterval(() => {
    loadLogs(false);
  }, 5000);
});

onBeforeUnmount(() => {
  if (refreshTimer) {
    window.clearInterval(refreshTimer);
  }
});
</script>

<template>
  <AppShell
    eyebrow="Administrator Audit Console"
    title="管理员运行日志"
  >
    <template #hero-actions>
      <div class="panel-actions">
        <span v-if="isRefreshing" class="muted">正在刷新...</span>
        <button class="ghost-btn" type="button" @click="loadLogs(true)">立即刷新</button>
      </div>
    </template>

    <div v-if="errorMessage" class="error-banner" style="margin-bottom: 14px;">{{ errorMessage }}</div>

    <section class="logs-grid">
      <div class="stat-stack">
        <article class="stat-card">
          <p>当前日志数</p>
          <h3>{{ totalLogs }}</h3>
        </article>
        <article class="stat-card">
          <p>告警条数</p>
          <h3>{{ warnCount }}</h3>
        </article>
        <article class="stat-card">
          <p>错误条数</p>
          <h3>{{ errorCount }}</h3>
        </article>
      </div>

      <article class="panel panel-table">
        <div class="panel-title-row">
          <h3>系统与鉴权日志</h3>
          <span>最新 120 条</span>
        </div>

        <div v-if="isLoading" class="empty-state">正在加载日志...</div>

        <div v-else-if="!logs.length" class="empty-state">暂无日志记录。</div>

        <div v-else class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>time</th>
                <th>level</th>
                <th>actor</th>
                <th>action</th>
                <th>message</th>
                <th>ip</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in logs" :key="item.id">
                <td>{{ item.timestamp }}</td>
                <td>
                  <span class="log-level" :class="item.level">{{ item.level }}</span>
                </td>
                <td>
                  <div>{{ item.actor || '-' }}</div>
                  <div class="meta-kv" v-if="item.role">
                    <span>role</span>
                    <strong>{{ item.role }}</strong>
                  </div>
                </td>
                <td>{{ item.action }}</td>
                <td>{{ item.message }}</td>
                <td>{{ item.ip || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </AppShell>
</template>