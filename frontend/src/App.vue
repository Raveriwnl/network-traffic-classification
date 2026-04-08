<script setup>
import * as echarts from 'echarts';
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { createTelemetrySocket } from './services/wsClient';

const classes = [
  { key: 'openlive', label: '直播主播侧', color: '#ff6b35' },
  { key: 'live', label: '直播观众侧', color: '#0bb4ff' },
  { key: 'message', label: '即时消息', color: '#50e3c2' },
  { key: 'short_video', label: '短视频', color: '#ffd166' },
  { key: 'video', label: '视频点播', color: '#7bd389' },
  { key: 'meeting', label: '视频会议', color: '#f15bb5' },
  { key: 'phone_game', label: '手机游戏', color: '#9b5de5' },
  { key: 'cloud_game', label: '云游戏', color: '#ff006e' }
];

const classLabels = classes.reduce((acc, item) => {
  acc[item.key] = item.label;
  return acc;
}, {});

const state = reactive({
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
  recentPackets: []
});

const packetChartRef = ref(null);
const pieChartRef = ref(null);
let packetChart = null;
let pieChart = null;
let socketController = null;

const socketBadge = computed(() => {
  if (state.socketStatus === 'connected') return 'status-ok';
  if (state.socketStatus === 'connecting') return 'status-warn';
  return 'status-off';
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
      textStyle: { color: '#f5efe2' },
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
      axisLine: { lineStyle: { color: 'rgba(245,239,226,0.3)' } },
      axisLabel: { color: '#c9b89e', fontSize: 11 }
    },
    yAxis: [
      {
        type: 'value',
        name: 'packet_size',
        axisLabel: { color: '#c9b89e' },
        splitLine: { lineStyle: { color: 'rgba(245,239,226,0.08)' } }
      },
      {
        type: 'value',
        name: 'iat(ms)',
        axisLabel: { color: '#c9b89e' },
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
        lineStyle: { width: 3, color: '#ff6b35' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(255,107,53,0.45)' },
            { offset: 1, color: 'rgba(255,107,53,0.02)' }
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
        lineStyle: { width: 3, color: '#0bb4ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(11,180,255,0.35)' },
            { offset: 1, color: 'rgba(11,180,255,0.02)' }
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
      textStyle: { color: '#f5efe2', fontSize: 12 }
    },
    series: [
      {
        type: 'pie',
        radius: ['34%', '68%'],
        center: ['34%', '50%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderColor: '#1b1f2f',
          borderWidth: 2
        },
        label: {
          show: true,
          color: '#f5efe2',
          formatter: '{b|{b}}\n{c|{d}%}',
          rich: {
            b: { fontSize: 11, fontWeight: 600 },
            c: { fontSize: 11, color: '#c9b89e' }
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

onMounted(() => {
  packetChart = echarts.init(packetChartRef.value);
  pieChart = echarts.init(pieChartRef.value);
  packetChart.setOption(createPacketOption());
  pieChart.setOption(createPieOption());

  socketController = createTelemetrySocket({
    onMessage: handleTelemetry,
    onStatus: (status) => {
      state.socketStatus = status;
    }
  });

  window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
  socketController?.close();
  window.removeEventListener('resize', handleResize);
  packetChart?.dispose();
  pieChart?.dispose();
});
</script>

<template>
  <div class="page-shell">
    <header class="hero-header">
      <div>
        <p class="eyebrow">Edge Traffic Intelligence Console</p>
        <h1>面向边缘计算的加密网络流量精准分类系统</h1>
      </div>
      <div class="status-pill" :class="socketBadge">
        WebSocket: {{ state.socketStatus }}
      </div>
    </header>

    <section class="kpi-grid">
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
          <h3>流量包大小 & 到达时间间隔波动</h3>
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
          <p class="prediction-confidence">
            置信度: {{ (state.latestConfidence * 100).toFixed(2) }}%
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
                  <span
                    class="direction-badge"
                    :class="item.direction === 'uplink' ? 'up' : 'down'"
                  >
                    {{ item.direction }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </div>
</template>
