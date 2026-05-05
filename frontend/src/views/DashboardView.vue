<script setup>
import * as echarts from 'echarts';
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { useRouter } from 'vue-router';
import AppShell from '../components/AppShell.vue';
import { clearAuth, useAuth } from '../auth';
import { apiFetch } from '../services/api';
import { createTelemetrySocket } from '../services/wsClient';

const MONITOR_STORAGE_KEY = 'dashboard.monitorEnabled';

const router = useRouter();
const auth = useAuth();

const classes = [
  { key: 'openlive', label: '直播主播侧', color: '#ff7a18' },
  { key: 'live', label: '直播观众侧', color: '#3dc2ff' },
  { key: 'message', label: '即时消息', color: '#7ce0c3' },
  { key: 'short_video', label: '短视频', color: '#ffd166' },
  { key: 'video', label: '视频点播', color: '#8de36f' },
  { key: 'meeting', label: '视频会议', color: '#f08bd2' },
  { key: 'phone_game', label: '手机游戏', color: '#9f84ff' },
  { key: 'cloud_game', label: '云游戏', color: '#ff5f7c' }
];

const classLabels = classes.reduce((acc, item) => {
  acc[item.key] = item.label;
  return acc;
}, {});

const state = reactive({
  monitorEnabled: true,
  socketStatus: 'connecting',
  latestPrediction: '-',
  latestConfidence: 0,
  metrics: {
    accuracy: 0,
    recall: 0,
    inference_latency_ms: 0,
    power_w: 0,
    flows_per_sec: 0
  },
  packetSeries: [],
  iatSeries: [],
  timeLabels: [],
  classDistribution: classes.reduce((acc, item) => {
    acc[item.key] = 0;
    return acc;
  }, {}),
  recentPackets: [],
  errorMessage: ''
});

const packetChartRef = ref(null);
const pieChartRef = ref(null);
let packetChart = null;
let pieChart = null;
let socketController = null;

function loadMonitorPreference() {
  const storedValue = window.localStorage.getItem(MONITOR_STORAGE_KEY);
  if (storedValue === null) {
    return true;
  }
  return storedValue === 'true';
}

function persistMonitorPreference(enabled) {
  window.localStorage.setItem(MONITOR_STORAGE_KEY, String(enabled));
}

const socketBadge = computed(() => {
  if (!state.monitorEnabled) return 'status-off';
  if (state.socketStatus === 'connected') return 'status-ok';
  if (state.socketStatus === 'connecting') return 'status-warn';
  return 'status-off';
});

const socketStatusText = computed(() => {
  if (!state.monitorEnabled) return '已关闭';
  if (state.socketStatus === 'connected') return '已连接';
  if (state.socketStatus === 'connecting') return '连接中';
  if (state.socketStatus === 'error') return '异常';
  return '已断开';
});

const topClass = computed(() => {
  const entries = Object.entries(state.classDistribution);
  if (!entries.length) return '-';
  const [key] = entries.sort((a, b) => b[1] - a[1])[0];
  return classLabels[key] || key;
});

function createPacketOption() {
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: {
      top: 8,
      textStyle: { color: '#f3f7fb' },
      itemGap: 24
    },
    grid: {
      left: 40,
      right: 24,
      top: 48,
      bottom: 28
    },
    xAxis: {
      type: 'category',
      data: state.timeLabels,
      boundaryGap: false,
      axisLine: { lineStyle: { color: 'rgba(243,247,251,0.22)' } },
      axisLabel: { color: '#9db5cb', fontSize: 11 }
    },
    yAxis: [
      {
        type: 'value',
        name: 'packet_size',
        axisLabel: { color: '#9db5cb' },
        splitLine: { lineStyle: { color: 'rgba(243,247,251,0.06)' } }
      },
      {
        type: 'value',
        name: 'iat(ms)',
        axisLabel: { color: '#9db5cb' },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: 'Packet Size',
        type: 'line',
        smooth: true,
        symbol: 'none',
        data: state.packetSeries,
        lineStyle: { width: 3, color: '#ff7a18' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(255,122,24,0.38)' },
            { offset: 1, color: 'rgba(255,122,24,0.02)' }
          ])
        }
      },
      {
        name: 'IAT',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        symbol: 'none',
        data: state.iatSeries,
        lineStyle: { width: 3, color: '#3dc2ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(61,194,255,0.28)' },
            { offset: 1, color: 'rgba(61,194,255,0.02)' }
          ])
        }
      }
    ]
  };
}

function createPieOption() {
  return {
    tooltip: { trigger: 'item' },
    legend: {
      orient: 'vertical',
      right: 0,
      top: 'middle',
      textStyle: { color: '#f3f7fb', fontSize: 12 }
    },
    series: [
      {
        type: 'pie',
        radius: ['34%', '68%'],
        center: ['34%', '50%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderColor: '#07111e',
          borderWidth: 2
        },
        label: {
          show: true,
          color: '#f3f7fb',
          formatter: '{b|{b}}\n{c|{d}%}',
          rich: {
            b: { fontSize: 11, fontWeight: 600 },
            c: { fontSize: 11, color: '#9db5cb' }
          }
        },
        labelLine: { length: 12, length2: 8 },
        data: classes.map((item) => ({
          name: item.label,
          value: state.classDistribution[item.key] || 0,
          itemStyle: { color: item.color }
        }))
      }
    ]
  };
}

function pushSeriesValue(collection, value, maxLen = 60) {
  collection.push(value);
  if (collection.length > maxLen) {
    collection.shift();
  }
}

function handleTelemetry(payload) {
  const now = new Date(payload.timestamp || Date.now());
  const timeStr = now.toLocaleTimeString('zh-CN', { hour12: false });

  const stream = payload.stream || {};
  const metrics = payload.metrics || {};
  const distribution = payload.distribution || {};
  const prediction = payload.prediction || {};

  pushSeriesValue(state.timeLabels, timeStr, 60);
  pushSeriesValue(state.packetSeries, Number(stream.packet_size || 0), 60);
  pushSeriesValue(state.iatSeries, Number(stream.iat || 0), 60);

  state.metrics.accuracy = Number(metrics.accuracy || 0);
  state.metrics.recall = Number(metrics.recall || 0);
  state.metrics.inference_latency_ms = Number(metrics.inference_latency_ms || 0);
  state.metrics.power_w = Number(metrics.power_w || 0);
  state.metrics.flows_per_sec = Number(metrics.flows_per_sec || 0);

  for (const item of classes) {
    state.classDistribution[item.key] = Number(distribution[item.key] || 0);
  }

  state.latestPrediction = classLabels[prediction.class_name] || prediction.class_name || '-';
  state.latestConfidence = Number(prediction.confidence || 0);

  state.recentPackets.unshift({
    time: timeStr,
    packet_size: Number(stream.packet_size || 0),
    iat: Number(stream.iat || 0),
    direction: String(stream.direction || 'downlink')
  });
  state.recentPackets = state.recentPackets.slice(0, 18);

  packetChart?.setOption(createPacketOption());
  pieChart?.setOption(createPieOption());
}

function handleResize() {
  packetChart?.resize();
  pieChart?.resize();
}

async function loadInitialTelemetry() {
  try {
    const payload = await apiFetch('/api/telemetry/latest');
    handleTelemetry(payload);
    state.errorMessage = '';
  } catch (error) {
    state.errorMessage = error.message || '无法加载初始遥测数据。';
  }
}

async function handleUnauthorized() {
  clearAuth();
  await router.push({ name: 'login' });
}

async function activateMonitoring() {
  if (state.monitorEnabled && state.socketStatus === 'connected') {
    return;
  }
  state.monitorEnabled = true;
  persistMonitorPreference(true);
  state.errorMessage = '';
  await startTelemetryMonitoring();
}

function stopTelemetryMonitoring() {
  socketController?.close();
  socketController = null;
  state.socketStatus = 'disconnected';
}

async function startTelemetryMonitoring() {
  stopTelemetryMonitoring();
  state.socketStatus = 'connecting';
  await loadInitialTelemetry();
  if (!state.monitorEnabled) {
    return;
  }
  socketController = createTelemetrySocket({
    token: auth.token,
    onMessage: handleTelemetry,
    onStatus: (status) => {
      state.socketStatus = status;
    },
    onUnauthorized: handleUnauthorized
  });
}

async function toggleMonitoring() {
  state.monitorEnabled = !state.monitorEnabled;
  persistMonitorPreference(state.monitorEnabled);
  state.errorMessage = '';
  if (state.monitorEnabled) {
    await startTelemetryMonitoring();
    return;
  }
  stopTelemetryMonitoring();
}

onMounted(async () => {
  state.monitorEnabled = loadMonitorPreference();
  packetChart = echarts.init(packetChartRef.value);
  pieChart = echarts.init(pieChartRef.value);
  packetChart.setOption(createPacketOption());
  pieChart.setOption(createPieOption());

  if (state.monitorEnabled) {
    await startTelemetryMonitoring();
  } else {
    state.socketStatus = 'disconnected';
  }

  window.addEventListener('dashboard:activate-monitoring', activateMonitoring);
  window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
  socketController?.close();
  window.removeEventListener('dashboard:activate-monitoring', activateMonitoring);
  window.removeEventListener('resize', handleResize);
  packetChart?.dispose();
  pieChart?.dispose();
});
</script>

<template>
  <AppShell
    eyebrow="Authenticated Telemetry Dashboard"
    title="加密流量分类实时看板"
  >
    <template #hero-actions>
      <div class="panel-actions telemetry-hero-actions">
        <div class="status-pill" :class="socketBadge">
          WebSocket: {{ socketStatusText }}
        </div>
        <button
          class="telemetry-toggle"
          :class="{ active: state.monitorEnabled }"
          type="button"
          @click="toggleMonitoring"
        >
          <span class="telemetry-toggle-track">
            <span class="telemetry-toggle-thumb"></span>
          </span>
          <span>{{ state.monitorEnabled ? '实时监控已开启' : '实时监控已关闭' }}</span>
        </button>
      </div>
    </template>

    <div v-if="state.errorMessage" class="error-banner">{{ state.errorMessage }}</div>

    <section class="kpi-grid" style="margin-top: 16px;">
      <article class="kpi-card">
        <p>推理延迟</p>
        <h2>{{ state.metrics.inference_latency_ms.toFixed(2) }} ms</h2>
      </article>
      <article class="kpi-card">
        <p>能耗监控</p>
        <h2>{{ state.metrics.power_w.toFixed(2) }} W</h2>
      </article>
      <article class="kpi-card">
        <p>吞吐速率</p>
        <h2>{{ state.metrics.flows_per_sec.toFixed(1) }} flows/s</h2>
      </article>
      <article class="kpi-card accent">
        <p>当前主导业务</p>
        <h2>{{ topClass }}</h2>
      </article>
    </section>

    <section class="visual-grid">
      <article class="panel panel-wide">
        <div class="panel-title-row">
          <h3>流量包大小与到达间隔</h3>
          <span>观测窗口: 60 秒</span>
        </div>
        <div ref="packetChartRef" class="chart chart-line"></div>
      </article>

      <article class="panel">
        <div class="panel-title-row">
          <h3>8类业务占比分布</h3>
          <span>实时统计</span>
        </div>
        <div ref="pieChartRef" class="chart chart-pie"></div>
      </article>
    </section>

    <section class="bottom-grid">
      <article class="panel">
        <div class="panel-title-row">
          <h3>最新分类输出</h3>
        </div>
        <div class="prediction-card">
          <p class="prediction-label">业务类别</p>
          <h2>{{ state.latestPrediction }}</h2>
          <p class="prediction-confidence">置信度: {{ (state.latestConfidence * 100).toFixed(2) }}%</p>
          <p class="panel-note">
            精度: {{ (state.metrics.accuracy * 100).toFixed(2) }}% · 召回率: {{ (state.metrics.recall * 100).toFixed(2) }}%
          </p>
        </div>
      </article>

      <article class="panel panel-table">
        <div class="panel-title-row">
          <h3>Input Metadata (最近18条)</h3>
          <span>packet_size / iat / direction</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>time</th>
                <th>packet_size</th>
                <th>iat(ms)</th>
                <th>direction</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, idx) in state.recentPackets" :key="`${item.time}-${idx}`">
                <td>{{ item.time }}</td>
                <td>{{ item.packet_size }}</td>
                <td>{{ item.iat.toFixed(2) }}</td>
                <td>
                  <span class="direction-badge" :class="item.direction === 'uplink' ? 'up' : 'down'">
                    {{ item.direction }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </AppShell>
</template>