/**
 * SensorChart.jsx — Real-time Sensor Chart
 * ===========================================
 * Hiển thị 4 sensor dưới dạng line chart, cập nhật real-time từ BLE.
 * Dùng Chart.js qua react-chartjs-2.
 *
 * Props:
 *   sensorData  {{ s1[], s2[], s3[], s4[], timestamps[] }}
 *   isConnected {boolean}
 *   height      {number}  default 280
 */

import { useRef, useEffect, useMemo } from 'react'
import {
  Chart as ChartJS,
  LineElement, PointElement, LinearScale,
  CategoryScale, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  LineElement, PointElement, LinearScale,
  CategoryScale, Title, Tooltip, Legend, Filler,
)

// Màu từng sensor
const SENSOR_COLORS = {
  s1: { line: '#10B981', fill: '#10B98115' },   // teal    — ụ ngón cái
  s2: '#F59E0B',                                 // amber   — ngón cái
  s3: '#6366F1',                                 // indigo  — ụ ngón út
  s4: '#F43F5E',                                 // rose    — gót chân
}

const SENSOR_LABELS = {
  s1: 'Ụ ngón cái (S1)',
  s2: 'Ngón cái (S2)',
  s3: 'Ụ ngón út (S3)',
  s4: 'Gót chân (S4)',
}

export default function SensorChart({ sensorData, isConnected, height = 280 }) {
  const chartRef = useRef(null)

  // Tạo labels thời gian từ timestamps
  const labels = useMemo(() => {
    if (!sensorData?.timestamps?.length) return []
    return sensorData.timestamps.map((ts) => {
      const d = new Date(ts)
      return `${d.getMinutes().toString().padStart(2,'0')}:${d.getSeconds().toString().padStart(2,'0')}.${Math.floor(d.getMilliseconds()/100)}`
    })
  }, [sensorData?.timestamps])

  const chartData = useMemo(() => ({
    labels,
    datasets: [
      makeDataset('s1', sensorData?.s1 ?? []),
      makeDataset('s2', sensorData?.s2 ?? []),
      makeDataset('s3', sensorData?.s3 ?? []),
      makeDataset('s4', sensorData?.s4 ?? []),
    ],
  }), [sensorData, labels])

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },   // tắt animation để real-time mượt
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#94A3B8',
          font: { size: 11, family: "'DM Mono', monospace" },
          boxWidth: 12, boxHeight: 2, padding: 16,
          usePointStyle: true, pointStyle: 'line',
        },
      },
      tooltip: {
        backgroundColor: '#1E2433',
        borderColor: '#2D3548',
        borderWidth: 1,
        titleColor: '#94A3B8',
        bodyColor: '#E2E8F0',
        padding: 10,
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()}`,
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#475569',
          font: { size: 10, family: 'monospace' },
          maxTicksLimit: 8,
          maxRotation: 0,
        },
        grid: { color: '#1E2433' },
        border: { color: '#2D3548' },
      },
      y: {
        min: 0,
        ticks: {
          color: '#475569',
          font: { size: 10 },
          callback: (v) => v.toFixed(1),
        },
        grid: { color: '#2D354850' },
        border: { color: '#2D3548' },
      },
    },
  }), [])

  // Empty state
  const isEmpty = !sensorData?.s1?.length

  return (
    <div style={styles.wrap}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>Dữ liệu cảm biến</span>
        <div style={styles.badges}>
          {Object.entries(SENSOR_LABELS).map(([key, label]) => (
            <SensorBadge key={key} id={key} label={label} />
          ))}
        </div>
      </div>

      {/* Chart area */}
      <div style={{ ...styles.chartArea, height }}>
        {isEmpty ? (
          <EmptyState isConnected={isConnected} />
        ) : (
          <Line ref={chartRef} data={chartData} options={options} />
        )}
      </div>

      {/* Y-axis hint */}
      {!isEmpty && (
        <div style={styles.hint}>
          Normalized 0 – 1 &nbsp;·&nbsp; S1,S3,S4: ÷4095 &nbsp;·&nbsp; S2: ÷26400
        </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────

function SensorBadge({ id, label }) {
  const color = typeof SENSOR_COLORS[id] === 'object'
    ? SENSOR_COLORS[id].line
    : SENSOR_COLORS[id]
  return (
    <div style={{ ...styles.badge, borderColor: color + '60', color }}>
      <span style={{ ...styles.badgeDot, background: color }} />
      {label}
    </div>
  )
}

function EmptyState({ isConnected }) {
  return (
    <div style={styles.empty}>
      <div style={styles.emptyIcon}>{isConnected ? '📶' : '🦷'}</div>
      <div style={styles.emptyText}>
        {isConnected
          ? 'Đang chờ dữ liệu…'
          : 'Kết nối thiết bị để bắt đầu đo'}
      </div>
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────

const SENSOR_MAX = { s1: 4095, s2: 26400, s3: 4095, s4: 4095 }
function makeDataset(key, data) {
  const color = SENSOR_COLORS[key]
  const lineColor = typeof color === 'object' ? color.line : color
  const fillColor = typeof color === 'object' ? color.fill : color + '10'
  const normalized = data.map(v => parseFloat((v / SENSOR_MAX[key]).toFixed(4)))
  return {
    label:           SENSOR_LABELS[key],
    data:            normalized,
    borderColor:     lineColor,
    backgroundColor: fillColor,
    borderWidth:     1.5,
    pointRadius:     0,
    tension:         0.3,
    fill:            key === 's1',   // chỉ fill dưới S1 để tránh lộn xộn
  }
}

// ── Styles ────────────────────────────────────────────────────────────────
const styles = {
  wrap: {
    background: '#1E2433',
    border: '1px solid #2D3548',
    borderRadius: 12,
    padding: '16px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  header: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'flex-start', flexWrap: 'wrap', gap: 8,
  },
  title: { fontSize: 14, fontWeight: 600, color: '#E2E8F0' },
  badges: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  badge: {
    display: 'flex', alignItems: 'center', gap: 4,
    fontSize: 10, padding: '2px 8px',
    border: '1px solid', borderRadius: 999,
    fontFamily: 'monospace',
  },
  badgeDot: { width: 6, height: 6, borderRadius: '50%' },
  chartArea: { position: 'relative', width: '100%' },
  empty: {
    height: '100%', display: 'flex',
    flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: 8, opacity: 0.5,
  },
  emptyIcon: { fontSize: 32 },
  emptyText: { fontSize: 13, color: '#64748B' },
  hint: {
    fontSize: 10, color: '#475569',
    textAlign: 'right', fontFamily: 'monospace',
  },
}
