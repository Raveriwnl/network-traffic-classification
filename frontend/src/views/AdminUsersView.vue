<script setup>
import { computed, onMounted, reactive, ref } from 'vue';
import AppShell from '../components/AppShell.vue';
import { apiFetch } from '../services/api';

const users = ref([]);
const isLoading = ref(true);
const isRefreshing = ref(false);
const isCreating = ref(false);
const savingUserId = ref(null);
const editingUserId = ref(null);
const errorMessage = ref('');
const successMessage = ref('');

const createForm = reactive({
  username: '',
  password: '',
  display_name: '',
  role: 'analyst'
});

const editForm = reactive({
  display_name: '',
  role: 'analyst',
  status: 'active'
});

const totalUsers = computed(() => users.value.length);
const adminCount = computed(() => users.value.filter((item) => item.role === 'admin').length);
const disabledCount = computed(() => users.value.filter((item) => item.status === 'disabled').length);

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return '从未';
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

function clearMessages() {
  errorMessage.value = '';
  successMessage.value = '';
}

async function loadUsers(showLoader = false) {
  clearMessages();
  if (showLoader) {
    isLoading.value = true;
  } else {
    isRefreshing.value = true;
  }

  try {
    const response = await apiFetch('/api/admin/users');
    users.value = response.users || [];
  } catch (error) {
    errorMessage.value = error.message || '无法获取用户列表。';
  } finally {
    isLoading.value = false;
    isRefreshing.value = false;
  }
}

function resetCreateForm() {
  createForm.username = '';
  createForm.password = '';
  createForm.display_name = '';
  createForm.role = 'analyst';
}

function startEditing(user) {
  editingUserId.value = user.id;
  editForm.display_name = user.display_name;
  editForm.role = user.role;
  editForm.status = user.status;
  clearMessages();
}

function cancelEditing() {
  editingUserId.value = null;
}

async function createUser() {
  clearMessages();
  isCreating.value = true;
  try {
    await apiFetch('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify(createForm)
    });
    successMessage.value = '用户创建成功。';
    resetCreateForm();
    await loadUsers(false);
  } catch (error) {
    errorMessage.value = error.message || '创建用户失败。';
  } finally {
    isCreating.value = false;
  }
}

async function updateUser(userId) {
  clearMessages();
  savingUserId.value = userId;
  try {
    await apiFetch(`/api/admin/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify(editForm)
    });
    successMessage.value = '用户信息已更新。';
    editingUserId.value = null;
    await loadUsers(false);
  } catch (error) {
    errorMessage.value = error.message || '更新用户失败。';
  } finally {
    savingUserId.value = null;
  }
}

onMounted(async () => {
  await loadUsers(true);
});
</script>

<template>
  <AppShell
    eyebrow="Administrator User Console"
    title="管理员用户管理"
    subtitle="创建、启停和调整管理员/分析员账号。"
  >
    <template #hero-actions>
      <div class="panel-actions">
        <span v-if="isRefreshing" class="muted">正在刷新...</span>
        <button class="ghost-btn" type="button" @click="loadUsers(true)">刷新列表</button>
      </div>
    </template>

    <div v-if="errorMessage" class="error-banner" style="margin-bottom: 14px;">{{ errorMessage }}</div>
    <div v-if="successMessage" class="success-banner" style="margin-bottom: 14px;">{{ successMessage }}</div>

    <section class="logs-grid">
      <div class="stat-stack">
        <article class="stat-card">
          <p>用户总数</p>
          <h3>{{ totalUsers }}</h3>
        </article>
        <article class="stat-card">
          <p>管理员</p>
          <h3>{{ adminCount }}</h3>
        </article>
        <article class="stat-card">
          <p>已禁用</p>
          <h3>{{ disabledCount }}</h3>
        </article>
      </div>

      <div class="admin-users-layout">
        <article class="panel user-form-panel">
          <div class="panel-title-row">
            <h3>创建用户</h3>
            <span>最少 6 位密码</span>
          </div>

          <form class="user-form-grid" @submit.prevent="createUser">
            <label class="field">
              <span>用户名</span>
              <input v-model.trim="createForm.username" type="text" required maxlength="64" />
            </label>

            <label class="field">
              <span>显示名称</span>
              <input v-model.trim="createForm.display_name" type="text" required maxlength="128" />
            </label>

            <label class="field">
              <span>密码</span>
              <input v-model="createForm.password" type="password" required minlength="6" maxlength="128" />
            </label>

            <label class="field">
              <span>角色</span>
              <select v-model="createForm.role">
                <option value="analyst">analyst</option>
                <option value="admin">admin</option>
              </select>
            </label>

            <div class="inline-meta">
              <button class="btn" type="submit" :disabled="isCreating">
                {{ isCreating ? '创建中...' : '创建用户' }}
              </button>
              <button class="ghost-btn" type="button" @click="resetCreateForm">重置</button>
            </div>
          </form>
        </article>

        <article class="panel panel-table">
          <div class="panel-title-row">
            <h3>用户列表</h3>
            <span>管理员与分析员账号</span>
          </div>

          <div v-if="isLoading" class="empty-state">正在加载用户...</div>
          <div v-else-if="!users.length" class="empty-state">暂无用户记录。</div>

          <div v-else class="table-wrap">
            <table class="users-table">
              <thead>
                <tr>
                  <th>用户</th>
                  <th>角色</th>
                  <th>状态</th>
                  <th>最近登录</th>
                  <th>创建时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="user in users" :key="user.id">
                  <tr>
                    <td>
                      <div class="user-name-primary">{{ user.display_name }}</div>
                      <div class="log-time-secondary">{{ user.username }}</div>
                    </td>
                    <td>
                      <span class="user-role-pill" :class="user.role">{{ user.role }}</span>
                    </td>
                    <td>
                      <span class="user-status-pill" :class="user.status">{{ user.status }}</span>
                    </td>
                    <td class="log-time-cell">{{ formatTimestamp(user.last_login_at) }}</td>
                    <td class="log-time-cell">{{ formatTimestamp(user.created_at) }}</td>
                    <td>
                      <button class="ghost-btn" type="button" @click="startEditing(user)">
                        {{ editingUserId === user.id ? '编辑中' : '编辑' }}
                      </button>
                    </td>
                  </tr>
                  <tr v-if="editingUserId === user.id" class="user-edit-row">
                    <td colspan="6">
                      <form class="user-edit-grid" @submit.prevent="updateUser(user.id)">
                        <label class="field">
                          <span>显示名称</span>
                          <input v-model.trim="editForm.display_name" type="text" required maxlength="128" />
                        </label>

                        <label class="field">
                          <span>角色</span>
                          <select v-model="editForm.role">
                            <option value="analyst">analyst</option>
                            <option value="admin">admin</option>
                          </select>
                        </label>

                        <label class="field">
                          <span>状态</span>
                          <select v-model="editForm.status">
                            <option value="active">active</option>
                            <option value="disabled">disabled</option>
                          </select>
                        </label>

                        <div class="inline-meta">
                          <button class="btn" type="submit" :disabled="savingUserId === user.id">
                            {{ savingUserId === user.id ? '保存中...' : '保存修改' }}
                          </button>
                          <button class="ghost-btn" type="button" @click="cancelEditing">取消</button>
                        </div>
                      </form>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  </AppShell>
</template>