<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import AppShell from '../components/AppShell.vue';
import { apiFetch } from '../services/api';

const logs = ref([]);
const isLoading = ref(true);
const isRefreshing = ref(false);
const errorMessage = ref('');
const appliedFilters = ref('');

const filters = reactive({
  level: '',
  actor: '',
  keyword: '',
  startTime: '',
  endTime: ''
});
const currentPage = ref(1);
const pageSize = ref(20);

let refreshTimer = null;

const filteredLogs = computed(() => {
  const keyword = filters.keyword.trim().toLowerCase();
  if (!keyword) {
    return logs.value;
  }
  return logs.value.filter((item) => {
    const haystacks = [item.action, item.message, item.actor, item.role, item.ip]
      .filter(Boolean)
      .map((value) => String(value).toLowerCase());
    return haystacks.some((value) => value.includes(keyword));
  });
});

const totalLogs = computed(() => filteredLogs.value.length);
const errorCount = computed(() => filteredLogs.value.filter((item) => item.level === 'error').length);
const warnCount = computed(() => filteredLogs.value.filter((item) => item.level === 'warn').length);
const hasActiveFilters = computed(() => Boolean(filters.level || filters.actor.trim() || filters.keyword.trim() || filters.startTime || filters.endTime));
const totalPages = computed(() => Math.max(1, Math.ceil(totalLogs.value / pageSize.value)));
const pageStart = computed(() => (totalLogs.value ? (currentPage.value - 1) * pageSize.value + 1 : 0));
const pageEnd = computed(() => Math.min(currentPage.value * pageSize.value, totalLogs.value));
const paginatedLogs = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value;
  const end = start + pageSize.value;
  return filteredLogs.value.slice(start, end);
});
const visiblePageNumbers = computed(() => {
  const windowSize = 5;
  const halfWindow = Math.floor(windowSize / 2);
  let start = Math.max(1, currentPage.value - halfWindow);
  let end = Math.min(totalPages.value, start + windowSize - 1);
  start = Math.max(1, end - windowSize + 1);
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
});

watch([filteredLogs, pageSize], () => {
  if (currentPage.value > totalPages.value) {
    currentPage.value = totalPages.value;
  }
  if (currentPage.value < 1) {
    currentPage.value = 1;
  }
}, { immediate: true });

watch(() => filters.keyword, () => {
  currentPage.value = 1;
});

function toUtcIso(localDateTime) {
  if (!localDateTime) {
    return '';
  }
  const parsed = new Date(localDateTime);
  if (Number.isNaN(parsed.getTime())) {
    return '';
  }
  return parsed.toISOString();
}

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return '-';
  }
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return String(timestamp);
  }
  return parsed.toLocaleString('zh-CN', {
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

function buildQueryString() {
  const params = new URLSearchParams();
  params.set('limit', '200');
  if (filters.level) {
    params.set('level', filters.level);
  }
  if (filters.actor.trim()) {
    params.set('actor', filters.actor.trim());
  }
  if (filters.startTime) {
    params.set('start_time', toUtcIso(filters.startTime));
  }
  if (filters.endTime) {
    params.set('end_time', toUtcIso(filters.endTime));
  }
  return params.toString();
}

function updateAppliedFilters() {
  const tags = [];
  if (filters.level) {
    tags.push(`级别: ${filters.level}`);
  }
  if (filters.actor.trim()) {
    tags.push(`操作者: ${filters.actor.trim()}`);
  }
  if (filters.startTime) {
    tags.push(`开始: ${formatTimestamp(toUtcIso(filters.startTime))}`);
  }
  if (filters.endTime) {
    tags.push(`结束: ${formatTimestamp(toUtcIso(filters.endTime))}`);
  }
  if (filters.keyword.trim()) {
    tags.push(`关键字: ${filters.keyword.trim()}`);
  }
  appliedFilters.value = tags.join(' · ');
}

async function loadLogs(showLoader = false) {
  if (showLoader) {
    isLoading.value = true;
  } else {
    isRefreshing.value = true;
  }

  try {
    const response = await apiFetch(`/api/admin/logs?${buildQueryString()}`);
    logs.value = response.logs || [];
    currentPage.value = 1;
    updateAppliedFilters();
    errorMessage.value = '';
  } catch (error) {
    errorMessage.value = error.message || '无法获取管理员日志。';
  } finally {
    isLoading.value = false;
    isRefreshing.value = false;
  }
}

function resetFilters() {
  filters.level = '';
  filters.actor = '';
  filters.keyword = '';
  filters.startTime = '';
  filters.endTime = '';
  loadLogs(true);
}

function goToPage(page) {
  if (page < 1 || page > totalPages.value || page === currentPage.value) {
    return;
  }
  currentPage.value = page;
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
          <p>当前筛选结果</p>
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
          <span>服务端最新 200 条</span>
        </div>

        <div class="log-filter-panel">
          <div class="log-filter-grid">
            <label class="log-filter-field">
              <span>级别</span>
              <select v-model="filters.level">
                <option value="">全部</option>
                <option value="info">info</option>
                <option value="warn">warn</option>
                <option value="error">error</option>
              </select>
            </label>

            <label class="log-filter-field">
              <span>操作者</span>
              <input v-model.trim="filters.actor" type="text" placeholder="如 admin / system" />
            </label>

            <label class="log-filter-field log-filter-field-wide">
              <span>关键字</span>
              <input v-model.trim="filters.keyword" type="text" placeholder="搜索 action、message、role、ip" />
            </label>

            <label class="log-filter-field">
              <span>开始时间</span>
              <input v-model="filters.startTime" type="datetime-local" />
            </label>

            <label class="log-filter-field">
              <span>结束时间</span>
              <input v-model="filters.endTime" type="datetime-local" />
            </label>
          </div>

          <div class="log-filter-actions">
            <button class="btn" type="button" @click="loadLogs(true)">应用筛选</button>
            <button class="ghost-btn" type="button" @click="resetFilters">重置</button>
            <label class="pagination-size-picker">
              <span>每页</span>
              <select v-model.number="pageSize">
                <option :value="10">10</option>
                <option :value="20">20</option>
                <option :value="50">50</option>
              </select>
            </label>
            <span class="muted" v-if="hasActiveFilters">{{ appliedFilters }}</span>
          </div>
        </div>

        <div v-if="isLoading" class="empty-state">正在加载日志...</div>

        <div v-else-if="!filteredLogs.length" class="empty-state">当前筛选条件下没有匹配日志。</div>

        <div v-else class="table-wrap">
          <table class="logs-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>级别</th>
                <th>操作者</th>
                <th>动作</th>
                <th>消息</th>
                <th>IP</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in paginatedLogs" :key="item.id">
                <td class="log-time-cell">
                  <div class="log-time-primary">{{ formatTimestamp(item.timestamp) }}</div>
                  <div class="log-time-secondary">{{ item.timestamp }}</div>
                </td>
                <td class="log-level-cell">
                  <span class="log-level" :class="item.level">{{ item.level }}</span>
                </td>
                <td class="log-actor-cell">
                  <div class="log-actor-primary">{{ item.actor || '-' }}</div>
                  <div class="meta-kv" v-if="item.role">
                    <span>角色</span>
                    <strong>{{ item.role }}</strong>
                  </div>
                </td>
                <td class="log-action-cell">{{ item.action }}</td>
                <td class="log-message-cell">{{ item.message }}</td>
                <td class="log-ip-cell">{{ item.ip || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="filteredLogs.length" class="pagination-bar">
          <div class="pagination-summary">
            显示第 {{ pageStart }} - {{ pageEnd }} 条，共 {{ totalLogs }} 条
          </div>
          <div class="pagination-controls">
            <button class="ghost-btn" type="button" :disabled="currentPage === 1" @click="goToPage(1)">首页</button>
            <button class="ghost-btn" type="button" :disabled="currentPage === 1" @click="goToPage(currentPage - 1)">上一页</button>
            <button
              v-for="page in visiblePageNumbers"
              :key="page"
              class="pagination-page-btn"
              :class="{ active: page === currentPage }"
              type="button"
              @click="goToPage(page)"
            >
              {{ page }}
            </button>
            <button class="ghost-btn" type="button" :disabled="currentPage === totalPages" @click="goToPage(currentPage + 1)">下一页</button>
            <button class="ghost-btn" type="button" :disabled="currentPage === totalPages" @click="goToPage(totalPages)">末页</button>
          </div>
        </div>
      </article>
    </section>
  </AppShell>
</template>