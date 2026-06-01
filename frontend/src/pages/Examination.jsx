/**
 * Examination.jsx — Trang khám bệnh
 * ====================================
 * Điều phối toàn bộ flow khám:
 *   1. Chọn / xác nhận bệnh nhân
 *   2. Tạo session (POST /sessions)
 *   3. Kết nối BLE → stream sensor
 *   4. WebSocket → backend nhận sample → predict real-time
 *   5. Kết thúc → nhận SessionDiagnosis → chuyển sang Report
 *
 * Props:
 *   patientId  {number|null}  — nếu có, bỏ qua bước chọn bệnh nhân
 *   onBack     {function}     — quay lại Dashboard
 *   onFinish   {function(sessionId)} — chuyển sang Report sau khi khám xong
 *
 * Dev tip: đổi MOCK_BLE = true để test UI mà không cần firmware.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import api from '../services/apiService'
import useBLE from '../hooks/useBLE'
import BLEConnect from '../components/BLEConnect'
import SensorChart from '../components/SensorChart'
import ResultDisplay from '../components/ResultDisplay'

// ── Dev flag: mock BLE data (set false khi có firmware thật) ──────────────
const MOCK_BLE = true

// ── Flow steps ─────────────────────────────────────────────────────────────
const STEP = {
  SELECT_PATIENT: 'select_patient',  // chọn bệnh nhân nếu chưa có
  READY:          'ready',           // đã có patient, chưa bắt đầu
  CONNECTING:     'connecting',      // đang kết nối BLE + tạo session
  MEASURING:      'measuring',       // đang đo
  FINISHING:      'finishing',       // đang flush + chờ diagnosis
  DONE:           'done',            // có SessionDiagnosis
}

// ── Helpers ────────────────────────────────────────────────────────────────
function calcAge(y) { return new Date().getFullYear() - y }

// ── Mock BLE generator (dùng khi MOCK_BLE = true) ─────────────────────────
function useMockBLE({ wsConnection, bufferSize = 200 }) {
  const [sensorData, setSensorData] = useState({ s1: [], s2: [], s3: [], s4: [], timestamps: [] })
  const [sampleCount, setSampleCount] = useState(0)
  const [isConnected, setIsConnected] = useState(false)
  const [isScanning,  setIsScanning]  = useState(false)
  const intervalRef = useRef(null)
  const bufRef = useRef({ s1: [], s2: [], s3: [], s4: [], timestamps: [] })
  const wsRef  = useRef(wsConnection)
  useEffect(() => { wsRef.current = wsConnection }, [wsConnection])

  const connect = useCallback(async () => {
    setIsScanning(true)
    await new Promise(r => setTimeout(r, 800))
    setIsScanning(false)
    setIsConnected(true)
    intervalRef.current = setInterval(() => {
      const now = Date.now()
      const t   = now / 1000
      const sample = {
        s1: Math.round(1800 + Math.sin(t * 2.1) * 900 + (Math.random() - 0.5) * 200),
        s2: Math.round(1200 + Math.sin(t * 1.7 + 1) * 700 + (Math.random() - 0.5) * 150),
        s3: Math.round(2200 + Math.sin(t * 2.4 + 2) * 600 + (Math.random() - 0.5) * 250),
        s4: Math.round(3000 + Math.sin(t * 1.3 + 3) * 800 + (Math.random() - 0.5) * 300),
        ts: now,
      }
      if (wsRef.current?.isOpen) wsRef.current.sendSample(sample)
      const buf = bufRef.current
      buf.s1.push(sample.s1); buf.s2.push(sample.s2)
      buf.s3.push(sample.s3); buf.s4.push(sample.s4)
      buf.timestamps.push(sample.ts)
      if (buf.s1.length > bufferSize) {
        buf.s1.shift(); buf.s2.shift(); buf.s3.shift(); buf.s4.shift(); buf.timestamps.shift()
      }
      setSampleCount(n => n + 1)
      if (buf.s1.length % 5 === 0) {
        setSensorData({ s1:[...buf.s1], s2:[...buf.s2], s3:[...buf.s3], s4:[...buf.s4], timestamps:[...buf.timestamps] })
      }
    }, 20)
  }, [bufferSize])

  const disconnect = useCallback(() => {
    clearInterval(intervalRef.current)
    setIsConnected(false)
  }, [])

  useEffect(() => () => clearInterval(intervalRef.current), [])

  return {
    status: isConnected ? 'connected' : isScanning ? 'scanning' : 'disconnected',
    statusMsg: isConnected ? 'Mock BLE — dữ liệu giả' : '',
    deviceName: isConnected ? 'SmartInsole (Mock)' : null,
    isConnected, isScanning, hasError: false,
    sensorData,
    stats: { sampleCount, actualHz: 50, durationSec: Math.round(sampleCount / 50) },
    connect, disconnect,
  }
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function Examination({ patientId: initPatientId, onBack, onFinish }) {
  const [step,        setStep]        = useState(initPatientId ? STEP.READY : STEP.SELECT_PATIENT)
  const [patientId,   setPatientId]   = useState(initPatientId ?? null)
  const [patient,     setPatient]     = useState(null)
  const [session,     setSession]     = useState(null)
  const [wsConn,      setWsConn]      = useState(null)
  const [windowResult,setWindowResult]= useState(null)
  const [diagnosis,   setDiagnosis]   = useState(null)
  const [statusMsg,   setStatusMsg]   = useState('')
  const [error,       setError]       = useState('')

  // Patient search state
  const [searchInput,  setSearchInput]  = useState('')
  const [searchResult, setSearchResult] = useState([])
  const [searching,    setSearching]    = useState(false)

  // BLE hook — real hoặc mock
  const bleHook = MOCK_BLE
    ? useMockBLE({ wsConnection: wsConn })
    : useBLE({ wsConnection: wsConn })

  const { isConnected, sensorData, stats, connect, disconnect } = bleHook

  // ── Load patient info ─────────────────────────────────────────────
  useEffect(() => {
    if (!patientId) return
    api.patients.get(patientId)
      .then(setPatient)
      .catch(() => setPatient(null))
  }, [patientId])

  // ── Patient search ────────────────────────────────────────────────
  useEffect(() => {
    if (!searchInput.trim()) { setSearchResult([]); return }
    const t = setTimeout(async () => {
      setSearching(true)
      try {
        const res = await api.patients.list({ search: searchInput, per_page: 6 })
        setSearchResult(res.items ?? [])
      } catch { setSearchResult([]) }
      finally { setSearching(false) }
    }, 300)
    return () => clearTimeout(t)
  }, [searchInput])

  // ── Start session + BLE ───────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (!patientId) return
    setError('')
    setStep(STEP.CONNECTING)
    setStatusMsg('Đang tạo phiên khám…')

    try {
      // 1. Tạo session
      const sess = await api.sessions.create({ patient_id: patientId })
      setSession(sess)
      setStatusMsg('Đang kết nối BLE…')

      // 2. Mở WebSocket
      const conn = api.ws.connect(sess.id, {
        onWindowResult: (pred) => setWindowResult(pred),
        onDiagnosis:    (diag) => {
          setDiagnosis(diag)
          setStep(STEP.DONE)
          setStatusMsg('')
        },
        onError: (msg) => setError(`WebSocket: ${msg}`),
        onClose: () => {},
      })
      setWsConn(conn)

      // 3. Kết nối BLE (sau khi WS sẵn sàng)
      await connect()
      setStep(STEP.MEASURING)
      setStatusMsg('')

    } catch (err) {
      setError(err.message)
      setStep(STEP.READY)
      setStatusMsg('')
    }
  }, [patientId, connect])

  // ── Stop measurement ──────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    if (!wsConn) return
    setStep(STEP.FINISHING)
    setStatusMsg('Đang tổng hợp kết quả…')
    disconnect()
    wsConn.endSession()
    // diagnosis sẽ đến qua onDiagnosis callback
    // fallback: nếu 10s không có → gọi REST
    setTimeout(async () => {
      if (!diagnosis && session) {
        try {
          const diag = await api.sessions.diagnosis(session.id)
          setDiagnosis(diag)
          setStep(STEP.DONE)
          setStatusMsg('')
        } catch {}
      }
    }, 10000)
  }, [wsConn, disconnect, diagnosis, session])

  // ── Cleanup khi unmount ───────────────────────────────────────────
  useEffect(() => () => { wsConn?.close(); disconnect() }, [])

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div style={s.root}>

      {/* ── Header ── */}
      <header style={s.header}>
        <button style={s.backBtn} onClick={onBack}>← Danh sách</button>

        <div style={s.headerCenter}>
          {patient ? (
            <div style={s.patientChip}>
              <div style={s.chipAvatar}>{patient.full_name.charAt(0)}</div>
              <div>
                <div style={s.chipName}>{patient.full_name}</div>
                <div style={s.chipMeta}>
                  {calcAge(patient.birth_year)} tuổi
                  {session && <> · Phiên #{session.id}</>}
                </div>
              </div>
            </div>
          ) : (
            <span style={s.headerTitle}>Phiên khám mới</span>
          )}
        </div>

        {/* Step indicator */}
        <StepIndicator step={step} />
      </header>

      {/* ── STEP: SELECT PATIENT ── */}
      {step === STEP.SELECT_PATIENT && (
        <div style={s.selectPatientPane}>
          <div style={s.selectCard}>
            <div style={s.selectTitle}>Chọn bệnh nhân</div>
            <div style={s.selectSearch}>
              <input
                style={s.searchInput}
                placeholder="Tìm tên bệnh nhân…"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                autoFocus
              />
            </div>
            <div style={s.selectResults}>
              {searching && <div style={s.selectHint}>Đang tìm…</div>}
              {!searching && searchInput && searchResult.length === 0 && (
                <div style={s.selectHint}>Không tìm thấy bệnh nhân</div>
              )}
              {!searchInput && (
                <div style={s.selectHint}>Nhập tên để tìm kiếm</div>
              )}
              {searchResult.map(p => (
                <button key={p.id} style={s.selectRow} onClick={() => {
                  setPatientId(p.id)
                  setPatient(p)
                  setStep(STEP.READY)
                }}>
                  <div style={s.selectAvatar}>{p.full_name.charAt(0)}</div>
                  <div>
                    <div style={s.selectName}>{p.full_name}</div>
                    <div style={s.selectMeta}>{calcAge(p.birth_year)} tuổi · ID #{p.id}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── STEP: READY / CONNECTING / MEASURING / FINISHING / DONE ── */}
      {step !== STEP.SELECT_PATIENT && (
        <div style={s.body}>

          {/* ── LEFT: BLE + Chart ── */}
          <div style={s.leftCol}>

            {/* BLE card */}
            <BLEConnect
              isConnected={bleHook.isConnected}
              isScanning={bleHook.isScanning}
              hasError={bleHook.hasError}
              status={bleHook.status}
              statusMsg={bleHook.statusMsg}
              deviceName={bleHook.deviceName}
              stats={stats}
              onConnect={handleStart}
              onDisconnect={handleStop}
            />

            {/* Status + error messages */}
            {statusMsg && <div style={s.infoMsg}>⏳ {statusMsg}</div>}
            {error     && <div style={s.errorMsg}>⚠ {error}</div>}
            {MOCK_BLE  && <div style={s.mockBadge}>DEV · Mock BLE data</div>}

            {/* Sensor chart */}
            <SensorChart
              sensorData={sensorData}
              isConnected={isConnected}
              height={260}
            />

            {/* Measurement controls */}
            {step === STEP.MEASURING && (
              <MeasurementBar stats={stats} onStop={handleStop} />
            )}

            {/* Done — go to report */}
            {step === STEP.DONE && diagnosis && (
              <button
                style={s.reportBtn}
                onClick={() => onFinish?.(session?.id)}
              >
                📋 Xem báo cáo đầy đủ →
              </button>
            )}
          </div>

          {/* ── RIGHT: ML Result ── */}
          <div style={s.rightCol}>
            <ResultDisplay
              windowResult={windowResult}
              diagnosis={step === STEP.DONE ? diagnosis : null}
              isActive={step === STEP.MEASURING}
            />

            {/* Guidance card */}
            {step === STEP.READY && (
              <GuideCard onStart={handleStart} hasPatient={!!patientId} />
            )}
          </div>

        </div>
      )}
    </div>
  )
}

// ── MeasurementBar ─────────────────────────────────────────────────────────

function MeasurementBar({ stats, onStop }) {
  return (
    <div style={s.measBar}>
      <div style={s.measInfo}>
        <span style={s.measDot} />
        <span style={s.measLabel}>Đang đo</span>
        <span style={s.measStat}>{stats.durationSec}s</span>
        <span style={s.measStat}>{stats.sampleCount.toLocaleString()} mẫu</span>
        <span style={s.measStat}>{stats.actualHz.toFixed(1)} Hz</span>
      </div>
      <button
        style={s.stopBtn}
        onClick={onStop}
        disabled={stats.durationSec < 5}
        title={stats.durationSec < 5 ? 'Cần đo ít nhất 5 giây' : ''}
      >
        ■ Dừng đo
      </button>
    </div>
  )
}

// ── StepIndicator ──────────────────────────────────────────────────────────

const STEP_ORDER = [STEP.READY, STEP.CONNECTING, STEP.MEASURING, STEP.FINISHING, STEP.DONE]
const STEP_LABEL = {
  [STEP.READY]:      'Sẵn sàng',
  [STEP.CONNECTING]: 'Kết nối',
  [STEP.MEASURING]:  'Đang đo',
  [STEP.FINISHING]:  'Xử lý',
  [STEP.DONE]:       'Hoàn thành',
}

function StepIndicator({ step }) {
  const cur = STEP_ORDER.indexOf(step)
  return (
    <div style={s.steps}>
      {STEP_ORDER.map((st, i) => (
        <div key={st} style={s.stepItem}>
          <div style={{
            ...s.stepDot,
            background: i < cur ? '#10B981' : i === cur ? '#3B82F6' : '#2D3548',
            border: i === cur ? '2px solid #3B82F680' : '2px solid transparent',
          }} />
          <span style={{ ...s.stepLabel, color: i === cur ? '#E2E8F0' : i < cur ? '#475569' : '#334155' }}>
            {STEP_LABEL[st]}
          </span>
          {i < STEP_ORDER.length - 1 && (
            <div style={{ ...s.stepLine, background: i < cur ? '#10B981' : '#2D3548' }} />
          )}
        </div>
      ))}
    </div>
  )
}

// ── GuideCard ──────────────────────────────────────────────────────────────

function GuideCard({ onStart, hasPatient }) {
  const steps = [
    { icon: '👟', text: 'Bệnh nhân mang lót giày và bật nguồn' },
    { icon: '📡', text: 'Nhấn "Kết nối thiết bị" để ghép BLE' },
    { icon: '🚶', text: 'Bệnh nhân đi bộ bình thường 5–10 phút' },
    { icon: '■',  text: 'Nhấn "Dừng đo" để nhận kết quả' },
  ]
  return (
    <div style={s.guideCard}>
      <div style={s.guideTitle}>Hướng dẫn</div>
      <div style={s.guideList}>
        {steps.map((st, i) => (
          <div key={i} style={s.guideRow}>
            <span style={s.guideNum}>{i + 1}</span>
            <span style={s.guideIcon}>{st.icon}</span>
            <span style={s.guideText}>{st.text}</span>
          </div>
        ))}
      </div>
      {hasPatient && (
        <button style={s.startBtn} onClick={onStart}>
          ▶ Bắt đầu khám
        </button>
      )}
    </div>
  )
}

// ── Styles ─────────────────────────────────────────────────────────────────

const s = {
  root: {
    minHeight: '100vh',
    background: '#0D1117',
    color: '#E2E8F0',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    display: 'flex', flexDirection: 'column',
  },

  // Header
  header: {
    display: 'flex', alignItems: 'center',
    padding: '12px 24px', gap: 16,
    borderBottom: '1px solid #1E2433',
    background: '#0D1117',
    position: 'sticky', top: 0, zIndex: 10,
  },
  headerTitle: { fontSize: 15, fontWeight: 600, color: '#F1F5F9' },
  backBtn: {
    background: 'transparent', border: '1px solid #2D3548',
    borderRadius: 7, padding: '7px 12px',
    color: '#94A3B8', fontSize: 13, cursor: 'pointer', flexShrink: 0,
  },
  headerCenter: { flex: 1 },

  // Patient chip
  patientChip: { display: 'flex', alignItems: 'center', gap: 10 },
  chipAvatar: {
    width: 32, height: 32, borderRadius: 8,
    background: '#1E2433', border: '1px solid #2D3548',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13, fontWeight: 600, color: '#94A3B8', flexShrink: 0,
  },
  chipName: { fontSize: 14, fontWeight: 500, color: '#F1F5F9' },
  chipMeta: { fontSize: 11, color: '#475569' },

  // Step indicator
  steps: { display: 'flex', alignItems: 'center', gap: 0, flexShrink: 0 },
  stepItem: { display: 'flex', alignItems: 'center', gap: 4 },
  stepDot: { width: 8, height: 8, borderRadius: '50%', flexShrink: 0, transition: 'all 0.3s' },
  stepLabel: { fontSize: 10, letterSpacing: '0.02em', whiteSpace: 'nowrap' },
  stepLine: { width: 16, height: 1, flexShrink: 0, transition: 'background 0.3s' },

  // Select patient
  selectPatientPane: {
    flex: 1, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    padding: 24,
  },
  selectCard: {
    width: '100%', maxWidth: 440,
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 14, padding: 24,
    display: 'flex', flexDirection: 'column', gap: 16,
  },
  selectTitle: { fontSize: 16, fontWeight: 600, color: '#F1F5F9' },
  selectSearch: {},
  searchInput: {
    width: '100%', boxSizing: 'border-box',
    background: '#0D1117', border: '1px solid #2D3548',
    borderRadius: 8, padding: '9px 14px',
    color: '#E2E8F0', fontSize: 13, outline: 'none',
  },
  selectResults: { display: 'flex', flexDirection: 'column', gap: 4, minHeight: 48 },
  selectHint: { fontSize: 13, color: '#475569', textAlign: 'center', padding: '16px 0' },
  selectRow: {
    display: 'flex', alignItems: 'center', gap: 12,
    background: 'transparent', border: '1px solid #2D3548',
    borderRadius: 9, padding: '10px 14px',
    cursor: 'pointer', textAlign: 'left', color: '#E2E8F0',
    transition: 'background 0.12s',
  },
  selectAvatar: {
    width: 34, height: 34, borderRadius: 8,
    background: '#2D3548',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 14, fontWeight: 600, color: '#94A3B8', flexShrink: 0,
  },
  selectName: { fontSize: 13, fontWeight: 500 },
  selectMeta: { fontSize: 11, color: '#475569', marginTop: 2 },

  // Body layout
  body: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: '1fr 380px',
    gap: 0,
    minHeight: 0,
  },

  // Left col
  leftCol: {
    display: 'flex', flexDirection: 'column', gap: 14,
    padding: '24px 20px 24px 24px',
    overflowY: 'auto',
    borderRight: '1px solid #1E2433',
  },

  // Right col
  rightCol: {
    display: 'flex', flexDirection: 'column', gap: 14,
    padding: '24px 24px 24px 20px',
    overflowY: 'auto',
  },

  // Status messages
  infoMsg: {
    background: '#1E2433', border: '1px solid #3B82F630',
    borderRadius: 8, padding: '8px 14px',
    fontSize: 12, color: '#93C5FD',
  },
  errorMsg: {
    background: '#2D1515', border: '1px solid #EF444440',
    borderRadius: 8, padding: '8px 14px',
    fontSize: 12, color: '#FCA5A5',
  },
  mockBadge: {
    background: '#2D1A00', border: '1px solid #F59E0B30',
    borderRadius: 6, padding: '5px 10px',
    fontSize: 10, color: '#F59E0B', letterSpacing: '0.05em',
    textAlign: 'center',
  },

  // Measurement bar
  measBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 10, padding: '10px 16px',
  },
  measInfo: { display: 'flex', alignItems: 'center', gap: 12 },
  measDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: '#EF4444',
    boxShadow: '0 0 0 3px #EF444430',
    animation: 'pulse 1.2s ease-in-out infinite',
    flexShrink: 0,
  },
  measLabel: { fontSize: 12, color: '#E2E8F0', fontWeight: 500 },
  measStat:  { fontSize: 12, color: '#475569', fontVariantNumeric: 'tabular-nums' },
  stopBtn: {
    background: '#EF4444', border: 'none', borderRadius: 7,
    padding: '7px 16px', color: '#fff',
    fontSize: 12, fontWeight: 600, cursor: 'pointer',
    transition: 'opacity 0.15s',
  },

  // Report button
  reportBtn: {
    width: '100%',
    background: '#10B981', border: 'none', borderRadius: 10,
    padding: '14px 0', color: '#fff',
    fontSize: 14, fontWeight: 600, cursor: 'pointer',
  },

  // Guide card
  guideCard: {
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 12, padding: '18px 20px',
    display: 'flex', flexDirection: 'column', gap: 14,
  },
  guideTitle: { fontSize: 13, fontWeight: 600, color: '#94A3B8' },
  guideList:  { display: 'flex', flexDirection: 'column', gap: 10 },
  guideRow:   { display: 'flex', alignItems: 'flex-start', gap: 10 },
  guideNum: {
    width: 20, height: 20, borderRadius: '50%',
    background: '#2D3548', flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 10, color: '#64748B', fontWeight: 600,
  },
  guideIcon: { fontSize: 14, flexShrink: 0, marginTop: 2 },
  guideText:  { fontSize: 13, color: '#94A3B8', lineHeight: 1.5 },
  startBtn: {
    width: '100%', background: '#10B981', border: 'none',
    borderRadius: 8, padding: '10px 0',
    color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer',
  },
}
