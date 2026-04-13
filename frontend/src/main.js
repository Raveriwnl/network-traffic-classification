import { createApp } from 'vue';
import App from './App.vue';
import { initializeAuth } from './auth';
import { router } from './router';
import './style.css';

initializeAuth();

createApp(App)
	.use(router)
	.mount('#app');
