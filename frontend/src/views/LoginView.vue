<script setup>
import { reactive, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { login } from '../auth';

const router = useRouter();
const route = useRoute();

const form = reactive({
  username: '',
  password: ''
});

const errorMessage = ref('');
const isSubmitting = ref(false);

async function handleSubmit() {
  errorMessage.value = '';
  isSubmitting.value = true;

  try {
    const user = await login(form);
    const fallbackRoute = user.role === 'admin' ? '/admin/logs' : '/dashboard';
    const redirect = String(route.query.redirect || fallbackRoute);
    const targetRoute = user.role === 'admin' && redirect === '/dashboard' ? fallbackRoute : redirect;
    await router.push(targetRoute);
  } catch (error) {
    errorMessage.value = error.message || '登录失败，请检查用户名和密码。';
  } finally {
    isSubmitting.value = false;
  }
}
</script>

<template>
  <div class="auth-screen">
    <div class="auth-card auth-card-compact">
      <section class="auth-panel auth-panel-compact">
        <p class="eyebrow">Encrypted Traffic Operations</p>
        <h2>登录</h2>

        <form class="auth-form" @submit.prevent="handleSubmit">
          <div class="field">
            <label for="username">用户名</label>
            <input id="username" v-model.trim="form.username" autocomplete="username" required />
          </div>

          <div class="field">
            <label for="password">密码</label>
            <input id="password" v-model="form.password" type="password" autocomplete="current-password" required />
          </div>

          <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>

          <div class="inline-meta">
            <button class="btn" type="submit" :disabled="isSubmitting">
              {{ isSubmitting ? '登录中...' : '进入系统' }}
            </button>
          </div>
        </form>
      </section>
    </div>
  </div>
</template>