/**
 * BLEConnect.jsx — Kết nối Bluetooth
 * =====================================
 * Hiển thị nút kết nối/ngắt kết nối BLE và trạng thái thiết bị.
 * Nhận useBLE hook output từ parent — không tự gọi hook.
 *
 * Props:
 *   isConnected  {boolean}
 *   isScanning   {boolean}
 *   hasError     {boolean}
 *   status       {string}   BLEStatus enum
 *   statusMsg    {string}
 *   deviceName   {string|null}
 *   stats        {{ sampleCount, actualHz, durationSec }}
 *   onConnect    {function}
 *   onDisconnect {function}
 */

import { useState } from 'react'
import { BLEStatus } from '../services/bleServices'

export default function BLEConnect({
  isConnected,
  isScanning,
  hasError,
  status,
  statusMsg,
  deviceName,
  stats = {},
  onConnect,
  onDisconnect,
}) {
  const [isLoading, setIsLoading] = useState(false)

  async function handleConnect() {
    setIsLoading(true)
    try {
      await onConnect()
    } catch (_) {
      // lỗi đã được xử lý trong useBLE
    } finally {
      setIsLoading(false)
    }
  }

  // ── Màu + label theo trạng thái ──────────────────────────────────
  const statusConfig = {
    [BLEStatus.DISCONNECTED]: { color: '#6B7280', dot: '#6B7280', label: 'Chưa kết nối' },
    [BLEStatus.SCANNING]:     { color: '#F59E0B', dot: '#F59E0B', label: 'Đang tìm kiếm…' },
    [BLEStatus.CONNECTING]:   { color: '#3B82F6', dot: '#3B82F6', label: 'Đang kết nối…' },
    [BLEStatus.CONNECTED]:    { color: '#10B981', dot: '#10B981', label: deviceName ?? 'Đã kết nối' },
    [BLEStatus.ERROR]:        { color: '#EF4444', dot: '#EF4444', label: 'Lỗi kết nối' },
  }
  const cfg = statusConfig[status] ?? statusConfig[BLEStatus.DISCONNECTED]

  return (
    <div style={styles.card}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.titleRow}>
          <span style={styles.icon}>📡</span>
          <span style={styles.title}>Bluetooth</span>
        </div>

        {/* Status dot + label */}
        <div style={styles.statusRow}>
          <span style={{ ...styles.dot, background: cfg.color,
            boxShadow: isConnected ? `0 0 0 4px ${cfg.color}30` : 'none',
            animation: isScanning ? 'pulse 1.2s ease-in-out infinite' : 'none',
          }} />
          <span style={{ ...styles.statusLabel, color: cfg.color }}>{cfg.label}</span>
        </div>
      </div>

      {/* Device name nếu đã kết nối */}
      {deviceName && (
        <div style={styles.deviceRow}>
          <span style={styles.deviceIcon}>🦷</span>
          <span style={styles.deviceName}>{deviceName}</span>
        </div>
      )}

      {/* Error message */}
      {hasError && statusMsg && (
        <div style={styles.errorBox}>{statusMsg}</div>
      )}

      {/* Stats khi đang đo */}
      {isConnected && stats.sampleCount > 0 && (
        <div style={styles.statsRow}>
          <Stat label="Mẫu" value={stats.sampleCount.toLocaleString()} />
          <Stat label="Hz thực" value={`${stats.actualHz.toFixed(1)}`} />
          <Stat label="Thời gian" value={`${stats.durationSec}s`} />
        </div>
      )}

      {/* Button */}
      <button
        style={{
          ...styles.btn,
          ...(isConnected ? styles.btnDisconnect : styles.btnConnect),
          opacity: isLoading || isScanning ? 0.7 : 1,
          cursor: isLoading || isScanning ? 'not-allowed' : 'pointer',
        }}
        onClick={isConnected ? onDisconnect : handleConnect}
        disabled={isLoading || isScanning}
      >
        {isLoading || isScanning
          ? '⏳ Đang xử lý…'
          : isConnected
            ? '⏏ Ngắt kết nối'
            : '🔗 Kết nối thiết bị'}
      </button>

      {/* CSS animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.5; transform: scale(1.3); }
        }
      `}</style>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div style={styles.stat}>
      <span style={styles.statValue}>{value}</span>
      <span style={styles.statLabel}>{label}</span>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────
const styles = {
  card: {
    background: '#1E2433',
    border: '1px solid #2D3548',
    borderRadius: 12,
    padding: '16px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    minWidth: 240,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleRow: { display: 'flex', alignItems: 'center', gap: 8 },
  icon: { fontSize: 18 },
  title: { fontSize: 14, fontWeight: 600, color: '#E2E8F0', letterSpacing: '0.02em' },
  statusRow: { display: 'flex', alignItems: 'center', gap: 6 },
  dot: {
    width: 8, height: 8, borderRadius: '50%',
    display: 'inline-block', flexShrink: 0,
    transition: 'box-shadow 0.3s',
  },
  statusLabel: { fontSize: 12, fontWeight: 500 },
  deviceRow: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: '#0D1117', borderRadius: 6, padding: '6px 10px',
  },
  deviceIcon: { fontSize: 12 },
  deviceName: { fontSize: 12, color: '#94A3B8', fontFamily: 'monospace' },
  errorBox: {
    background: '#2D1515', border: '1px solid #EF444440',
    borderRadius: 6, padding: '8px 10px',
    fontSize: 12, color: '#FCA5A5', lineHeight: 1.5,
  },
  statsRow: {
    display: 'flex', gap: 0,
    background: '#0D1117', borderRadius: 8, overflow: 'hidden',
  },
  stat: {
    flex: 1, display: 'flex', flexDirection: 'column',
    alignItems: 'center', padding: '8px 4px',
    borderRight: '1px solid #2D3548',
  },
  statValue: { fontSize: 15, fontWeight: 700, color: '#10B981', fontVariantNumeric: 'tabular-nums' },
  statLabel: { fontSize: 10, color: '#64748B', marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.05em' },
  btn: {
    width: '100%', padding: '10px 0', borderRadius: 8,
    border: 'none', fontSize: 13, fontWeight: 600,
    cursor: 'pointer', transition: 'all 0.15s', letterSpacing: '0.02em',
  },
  btnConnect:    { background: '#10B981', color: '#fff' },
  btnDisconnect: { background: '#2D3548', color: '#94A3B8' },
}
