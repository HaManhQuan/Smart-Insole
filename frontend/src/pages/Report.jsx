/**
 * Report.jsx — Báo cáo phiên khám
 * ==================================
 * Hiển thị kết quả chẩn đoán đầy đủ + lịch sử window-by-window.
 * Hỗ trợ in PDF qua window.print() với @media print CSS.
 *
 * Props:
 *   sessionId  {number}    — ID phiên khám cần xem
 *   onBack     {function}  — quay lại Dashboard
 *
 * API calls (không có BLE):
 *   GET /sessions/{id}           → SessionResponse
 *   GET /sessions/{id}/diagnosis → SessionDiagnosis
 *   GET /patients/{id}           → PatientResponse
 *   GET /sessions/{id}/predictions → SessionPredictionHistory
 *   GET /model/info              → ModelInfo
 */

import { useState, useEffect, useRef } from 'react'
import api from '../services/apiService'

// ── Constants ──────────────────────────────────────────────────────────────

const DIAG = {
  Normal:    { color: '#10B981', bg: '#10B98118', border: '#10B98140', icon: '✓', label: 'Bình thường',        desc: 'Dáng đi trong giới hạn bình thường.' },
  Parkinson: { color: '#F59E0B', bg: '#F59E0B18', border: '#F59E0B40', icon: '⚠', label: 'Nghi ngờ Parkinson', desc: 'Phát hiện các đặc điểm dáng đi liên quan đến Parkinson. Cần đánh giá lâm sàng thêm.' },
  Abnormal:  { color: '#EF4444', bg: '#EF444418', border: '#EF444440', icon: '!', label: 'Dáng đi bất thường',  desc: 'Phát hiện bất thường trong dáng đi. Không đủ đặc trưng để phân loại Parkinson.' },
  Uncertain: { color: '#6B7280', bg: '#6B728018', border: '#6B728040', icon: '?', label: 'Chưa xác định',       desc: 'Dữ liệu không đủ để đưa ra kết luận.' },
}

const SENSOR_META = {
  s1: { label: 'Ụ ngón cái',  color: '#10B981' },
  s2: { label: 'Ngón cái',    color: '#F59E0B' },
  s3: { label: 'Ụ ngón út',   color: '#6366F1' },
  s4: { label: 'Gót chân',    color: '#F43F5E' },
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmt(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('vi-VN', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function fmtDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString('vi-VN', {
    day: '2-digit', month: '2-digit', year: 'numeric',
  })
}

function calcAge(y) { return new Date().getFullYear() - y }

function pct(v) { return `${(v * 100).toFixed(1)}%` }

// ── Main ───────────────────────────────────────────────────────────────────

export default function Report({ sessionId, onBack }) {
  const [session,     setSession]     = useState(null)
  const [patient,     setPatient]     = useState(null)
  const [diagnosis,   setDiagnosis]   = useState(null)
  const [predictions, setPredictions] = useState([])
  const [modelInfo,   setModelInfo]   = useState(null)
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState('')
  const printRef = useRef()

  // ── Fetch all data ────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    let cancelled = false

    async function load() {
      setLoading(true)
      setError('')
      try {
        const sess = await api.sessions.get(sessionId)
        if (cancelled) return
        setSession(sess)

        const [pat, diag, hist, info] = await Promise.allSettled([
          api.patients.get(sess.patient_id),
          api.sessions.diagnosis(sessionId),
          api.sessions.predictions(sessionId),
          api.predictions.modelInfo(),
        ])

        if (cancelled) return
        if (pat.status   === 'fulfilled') setPatient(pat.value)
        if (diag.status  === 'fulfilled') setDiagnosis(diag.value)
        if (hist.status  === 'fulfilled') setPredictions(hist.value?.predictions ?? [])
        if (info.status  === 'fulfilled') setModelInfo(info.value)
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [sessionId])

  // ── Print ─────────────────────────────────────────────────────────
  function handlePrint() { window.print() }

  // ── Loading / error ───────────────────────────────────────────────
  if (loading) return <LoadingScreen />
  if (error)   return <ErrorScreen message={error} onBack={onBack} />
  if (!session || !diagnosis) return <ErrorScreen message="Phiên khám chưa có báo cáo." onBack={onBack} />

  const cfg = DIAG[diagnosis.diagnosis] ?? DIAG.Uncertain

  // ── Render ────────────────────────────────────────────────────────
  return (
    <>
      {/* Print CSS */}
      <style>{printCSS}</style>

      <div style={s.root} ref={printRef}>

        {/* ── Screen-only toolbar ── */}
        <div style={s.toolbar} className="no-print">
          <button style={s.backBtn} onClick={onBack}>← Quay lại</button>
          <div style={s.toolbarRight}>
            <span style={s.sessionId}>Phiên #{session.id}</span>
            <button style={s.printBtn} onClick={handlePrint}>🖨 In báo cáo</button>
          </div>
        </div>

        {/* ── Report body ── */}
        <div style={s.paper}>

          {/* ── Print header (chỉ hiện khi in) ── */}
          <div style={s.printHeader} className="print-only">
            <div style={s.printHeaderLeft}>
              <div style={s.printLogo}>Smart Insole</div>
              <div style={s.printSubtitle}>Hệ thống phân tích dáng đi thông minh</div>
            </div>
            <div style={s.printDate}>Ngày in: {fmtDate(new Date().toISOString())}</div>
          </div>

          {/* ── Section 1: Thông tin phiên ── */}
          <Section title="Thông tin phiên khám">
            <div style={s.infoGrid}>
              <InfoRow label="Mã phiên"    value={`#${session.id}`} />
              <InfoRow label="Bắt đầu"     value={fmt(session.started_at)} />
              <InfoRow label="Kết thúc"    value={fmt(session.ended_at)} />
              <InfoRow label="Thời gian đo" value={session.duration_sec != null ? `${Math.round(session.duration_sec)} giây` : '—'} />
              {session.notes && <InfoRow label="Ghi chú" value={session.notes} span />}
            </div>
          </Section>

          {/* ── Section 2: Thông tin bệnh nhân ── */}
          {patient && (
            <Section title="Thông tin bệnh nhân">
              <div style={s.infoGrid}>
                <InfoRow label="Họ tên"    value={patient.full_name} />
                <InfoRow label="Tuổi"      value={`${calcAge(patient.birth_year)} tuổi (${patient.birth_year})`} />
                <InfoRow label="Giới tính" value={{ male: 'Nam', female: 'Nữ', other: 'Khác' }[patient.gender] ?? patient.gender} />
                {patient.phone && <InfoRow label="Điện thoại" value={patient.phone} />}
                {patient.notes && <InfoRow label="Tiền sử"   value={patient.notes} span />}
              </div>
            </Section>
          )}

          {/* ── Section 3: Kết quả chẩn đoán ── */}
          <Section title="Kết quả chẩn đoán">

            {/* Diagnosis card */}
            <div style={{ ...s.diagCard, background: cfg.bg, borderColor: cfg.border }}>
              <div style={{ ...s.diagIcon, color: cfg.color, borderColor: cfg.border }}>
                {cfg.icon}
              </div>
              <div style={s.diagBody}>
                <div style={{ ...s.diagLabel, color: cfg.color }}>{cfg.label}</div>
                <div style={s.diagDesc}>{cfg.desc}</div>
              </div>
              <div style={s.diagConf}>
                <div style={{ ...s.diagConfValue, color: cfg.color }}>
                  {pct(diagnosis.confidence_mean)}
                </div>
                <div style={s.diagConfLabel}>Confidence</div>
              </div>
            </div>

            {/* Stats grid */}
            <div style={s.statsGrid}>
              <StatCard label="Tổng windows"    value={diagnosis.total_windows} />
              <StatCard label="Confident"       value={diagnosis.confident_windows} color="#10B981" />
              <StatCard label="Tỉ lệ confident" value={pct(diagnosis.confident_ratio)} color="#3B82F6" />
              <StatCard label="Đủ dữ liệu"      value={diagnosis.sufficient_data ? 'Có' : 'Không'}
                        color={diagnosis.sufficient_data ? '#10B981' : '#EF4444'} />
            </div>

            {/* Vote distribution */}
            {diagnosis.vote_distribution && (
              <div style={s.voteSection}>
                <div style={s.subTitle}>Phân bố vote</div>
                <div style={s.voteGrid}>
                  {Object.entries(diagnosis.vote_distribution).map(([name, count]) => {
                    const ratio = diagnosis.total_windows > 0 ? count / diagnosis.total_windows : 0
                    const dc = DIAG[name]
                    return (
                      <div key={name} style={s.voteRow}>
                        <span style={{ ...s.voteName, color: dc?.color ?? '#94A3B8' }}>{name}</span>
                        <div style={s.voteTrack}>
                          <div style={{
                            ...s.voteFill,
                            width: `${ratio * 100}%`,
                            background: dc?.color ?? '#94A3B8',
                            opacity: name === diagnosis.diagnosis ? 1 : 0.35,
                          }} />
                        </div>
                        <span style={{ ...s.voteCount, color: dc?.color ?? '#94A3B8' }}>
                          {count} ({(ratio * 100).toFixed(0)}%)
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </Section>

          {/* ── Section 4: Window-by-window ── */}
          {predictions.length > 0 && (
            <Section title={`Chi tiết từng window (${predictions.length} windows)`}>
              <WindowTimeline predictions={predictions} totalWindows={diagnosis.total_windows} />
              <WindowTable predictions={predictions} />
            </Section>
          )}

          {/* ── Section 5: Thông tin model ── */}
          {modelInfo && (
            <Section title="Thông tin mô hình AI">
              <div style={s.infoGrid}>
                <InfoRow label="Classes"   value={(modelInfo.classes ?? []).join(' / ')} />
                <InfoRow label="Accuracy"  value={modelInfo.eval_accuracy != null ? pct(modelInfo.eval_accuracy) : '—'} />
                <InfoRow label="ROC-AUC"   value={modelInfo.eval_roc_auc  != null ? modelInfo.eval_roc_auc.toFixed(3) : '—'} />
                <InfoRow label="Macro F1"  value={modelInfo.eval_macro_f1 != null ? modelInfo.eval_macro_f1.toFixed(3) : '—'} />
                <InfoRow label="Threshold" value={modelInfo.confidence_threshold != null ? pct(modelInfo.confidence_threshold) : '—'} />
              </div>
              <div style={s.modelNote}>
                Kết quả do mô hình AI phân tích. Cần kết hợp với đánh giá lâm sàng của bác sĩ.
              </div>
            </Section>
          )}

          {/* ── Print footer ── */}
          <div style={s.printFooter} className="print-only">
            <div>Smart Insole — Hệ thống phân tích dáng đi thông minh</div>
            <div>Phiên #{session.id} · {fmt(session.started_at)}</div>
          </div>

        </div>
      </div>
    </>
  )
}

// ── WindowTimeline ─────────────────────────────────────────────────────────
// Visual bar: mỗi ô = 1 window, màu theo diagnosis

function WindowTimeline({ predictions }) {
  return (
    <div style={s.timeline}>
      <div style={s.timelineLabel}>Tiến trình theo thời gian</div>
      <div style={s.timelineBar}>
        {predictions.map((p) => {
          const dc = DIAG[p.label] ?? DIAG.Uncertain
          return (
            <div
              key={p.window_id}
              title={`Window ${p.window_id}: ${p.label} (${pct(p.confidence)})`}
              style={{
                ...s.timelineCell,
                background: p.is_uncertain ? '#2D3548' : dc.color,
                opacity: p.is_uncertain ? 0.4 : (0.4 + p.confidence * 0.6),
              }}
            />
          )
        })}
      </div>
      <div style={s.timelineLegend}>
        {Object.entries(DIAG).filter(([k]) => k !== 'Uncertain').map(([key, dc]) => (
          <span key={key} style={s.legendItem}>
            <span style={{ ...s.legendDot, background: dc.color }} />
            {dc.label}
          </span>
        ))}
        <span style={s.legendItem}>
          <span style={{ ...s.legendDot, background: '#2D3548' }} />
          Uncertain
        </span>
      </div>
    </div>
  )
}

// ── WindowTable ────────────────────────────────────────────────────────────

function WindowTable({ predictions }) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? predictions : predictions.slice(0, 10)

  return (
    <div style={s.tableWrap}>
      <table style={s.table}>
        <thead>
          <tr>
            {['Window', 'Kết quả', 'Confidence', 'Uncertain'].map(h => (
              <th key={h} style={s.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((p) => {
            const dc = DIAG[p.label] ?? DIAG.Uncertain
            return (
              <tr key={p.window_id} style={s.tr}>
                <td style={s.td}>#{p.window_id}</td>
                <td style={s.td}>
                  <span style={{ ...s.badge, color: dc.color, background: dc.bg }}>
                    {p.label}
                  </span>
                </td>
                <td style={s.td}>
                  <div style={s.confCell}>
                    <div style={s.miniTrack}>
                      <div style={{ ...s.miniFill, width: pct(p.confidence), background: dc.color }} />
                    </div>
                    <span style={{ color: dc.color }}>{pct(p.confidence)}</span>
                  </div>
                </td>
                <td style={s.td}>
                  {p.is_uncertain
                    ? <span style={{ color: '#6B7280', fontSize: 11 }}>Có</span>
                    : <span style={{ color: '#2D3548', fontSize: 11 }}>—</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {predictions.length > 10 && (
        <button style={s.expandBtn} onClick={() => setExpanded(e => !e)}>
          {expanded ? '▲ Thu gọn' : `▼ Xem thêm ${predictions.length - 10} windows`}
        </button>
      )}
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────

function Section({ title, children }) {
  return (
    <div style={s.section}>
      <div style={s.sectionHeader}>
        <span style={s.sectionLine} />
        <span style={s.sectionTitle}>{title}</span>
      </div>
      <div style={s.sectionBody}>{children}</div>
    </div>
  )
}

function InfoRow({ label, value, span }) {
  return (
    <div style={{ ...s.infoRow, ...(span ? { gridColumn: '1 / -1' } : {}) }}>
      <span style={s.infoLabel}>{label}</span>
      <span style={s.infoValue}>{value}</span>
    </div>
  )
}

function StatCard({ label, value, color = '#E2E8F0' }) {
  return (
    <div style={s.statCard}>
      <div style={{ ...s.statValue, color }}>{value}</div>
      <div style={s.statLabel}>{label}</div>
    </div>
  )
}

function LoadingScreen() {
  return (
    <div style={s.fullCenter}>
      <div style={{ fontSize: 13, color: '#475569' }}>Đang tải báo cáo…</div>
    </div>
  )
}

function ErrorScreen({ message, onBack }) {
  return (
    <div style={s.fullCenter}>
      <div style={{ fontSize: 28, marginBottom: 12 }}>⚠️</div>
      <div style={{ fontSize: 13, color: '#FCA5A5', marginBottom: 16 }}>{message}</div>
      <button style={s.backBtn} onClick={onBack}>← Quay lại</button>
    </div>
  )
}

// ── Print CSS ──────────────────────────────────────────────────────────────

const printCSS = `
  .print-only { display: none !important; }

  @media print {
    .no-print  { display: none !important; }
    .print-only { display: flex !important; }

    body { background: #fff !important; color: #000 !important; }

    /* Override dark theme for print */
    [data-report-paper] {
      background: #fff !important;
      color: #111 !important;
      box-shadow: none !important;
      max-width: 100% !important;
      padding: 0 !important;
    }
  }
`

// ── Styles ─────────────────────────────────────────────────────────────────

const s = {
  root: {
    minHeight: '100vh',
    background: '#0D1117',
    color: '#E2E8F0',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
  },

  // Toolbar (screen only)
  toolbar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 28px',
    borderBottom: '1px solid #1E2433',
    background: '#0D1117',
    position: 'sticky', top: 0, zIndex: 10,
  },
  toolbarRight: { display: 'flex', alignItems: 'center', gap: 12 },
  sessionId: { fontSize: 12, color: '#475569', fontFamily: 'monospace' },
  backBtn: {
    background: 'transparent', border: '1px solid #2D3548',
    borderRadius: 7, padding: '7px 14px',
    color: '#94A3B8', fontSize: 13, cursor: 'pointer',
  },
  printBtn: {
    background: '#10B981', border: 'none',
    borderRadius: 7, padding: '8px 18px',
    color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer',
  },

  // Paper container
  paper: {
    maxWidth: 820,
    margin: '28px auto',
    padding: '0 24px 48px',
  },

  // Print header/footer
  printHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
    paddingBottom: 16, marginBottom: 24,
    borderBottom: '2px solid #E5E7EB',
  },
  printHeaderLeft: { display: 'flex', flexDirection: 'column', gap: 2 },
  printLogo:     { fontSize: 18, fontWeight: 700, color: '#111' },
  printSubtitle: { fontSize: 11, color: '#6B7280' },
  printDate:     { fontSize: 11, color: '#6B7280' },
  printFooter: {
    marginTop: 32, paddingTop: 12,
    borderTop: '1px solid #E5E7EB',
    display: 'flex', justifyContent: 'space-between',
    fontSize: 10, color: '#9CA3AF',
  },

  // Section
  section: { marginBottom: 32 },
  sectionHeader: {
    display: 'flex', alignItems: 'center', gap: 10,
    marginBottom: 16,
  },
  sectionLine: {
    width: 3, height: 16, borderRadius: 2,
    background: '#10B981', flexShrink: 0,
  },
  sectionTitle: { fontSize: 13, fontWeight: 600, color: '#94A3B8', letterSpacing: '0.06em', textTransform: 'uppercase' },
  sectionBody:  {},

  // Info grid
  infoGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: '0',
    background: '#1E2433', border: '1px solid #2D3548', borderRadius: 10, overflow: 'hidden',
  },
  infoRow: {
    display: 'flex', flexDirection: 'column', gap: 3,
    padding: '12px 16px',
    borderBottom: '1px solid #2D354860',
  },
  infoLabel: { fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' },
  infoValue: { fontSize: 13, color: '#E2E8F0' },

  // Diagnosis card
  diagCard: {
    display: 'flex', alignItems: 'center', gap: 16,
    border: '1px solid', borderRadius: 12,
    padding: '20px 24px', marginBottom: 16,
  },
  diagIcon: {
    width: 44, height: 44, borderRadius: 12,
    border: '1px solid',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 20, fontWeight: 700, flexShrink: 0,
  },
  diagBody: { flex: 1 },
  diagLabel: { fontSize: 18, fontWeight: 700, marginBottom: 4 },
  diagDesc:  { fontSize: 13, color: '#94A3B8', lineHeight: 1.5 },
  diagConf:  { textAlign: 'right', flexShrink: 0 },
  diagConfValue: { fontSize: 26, fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  diagConfLabel: { fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' },

  // Stats grid
  statsGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20,
  },
  statCard: {
    background: '#1E2433', border: '1px solid #2D3548', borderRadius: 10,
    padding: '14px 12px', textAlign: 'center',
    display: 'flex', flexDirection: 'column', gap: 4,
  },
  statValue: { fontSize: 22, fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  statLabel: { fontSize: 11, color: '#475569' },

  // Vote
  voteSection: { display: 'flex', flexDirection: 'column', gap: 10 },
  subTitle: { fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em' },
  voteGrid: { display: 'flex', flexDirection: 'column', gap: 8 },
  voteRow:  { display: 'flex', alignItems: 'center', gap: 10 },
  voteName: { fontSize: 12, fontWeight: 500, width: 80, flexShrink: 0 },
  voteTrack: { flex: 1, height: 6, background: '#0D1117', borderRadius: 999, overflow: 'hidden' },
  voteFill:  { height: '100%', borderRadius: 999, transition: 'width 0.5s ease' },
  voteCount: { fontSize: 12, fontVariantNumeric: 'tabular-nums', width: 80, textAlign: 'right', flexShrink: 0 },

  // Timeline
  timeline: {
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 10, padding: '14px 16px', marginBottom: 16,
  },
  timelineLabel: { fontSize: 11, color: '#475569', marginBottom: 8 },
  timelineBar: {
    display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 10,
  },
  timelineCell: {
    width: 12, height: 12, borderRadius: 2, flexShrink: 0,
    transition: 'opacity 0.2s',
    cursor: 'default',
  },
  timelineLegend: { display: 'flex', flexWrap: 'wrap', gap: 12 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#64748B' },
  legendDot:  { width: 8, height: 8, borderRadius: 2, flexShrink: 0 },

  // Table
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: {
    padding: '9px 14px', textAlign: 'left',
    fontSize: 11, color: '#475569',
    textTransform: 'uppercase', letterSpacing: '0.05em',
    borderBottom: '1px solid #2D3548',
    fontWeight: 500,
  },
  tr: { borderBottom: '1px solid #1E243380' },
  td: { padding: '9px 14px', color: '#CBD5E1', verticalAlign: 'middle' },
  badge: {
    fontSize: 11, fontWeight: 500,
    padding: '3px 10px', borderRadius: 999,
  },
  confCell:  { display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontVariantNumeric: 'tabular-nums' },
  miniTrack: { width: 60, height: 4, background: '#0D1117', borderRadius: 999, overflow: 'hidden', flexShrink: 0 },
  miniFill:  { height: '100%', borderRadius: 999 },
  expandBtn: {
    width: '100%', marginTop: 8,
    background: 'transparent', border: '1px solid #2D3548',
    borderRadius: 6, padding: '7px 0',
    color: '#64748B', fontSize: 12, cursor: 'pointer',
  },

  // Model note
  modelNote: {
    marginTop: 12,
    padding: '10px 14px',
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 8, fontSize: 12, color: '#64748B',
    lineHeight: 1.6,
  },

  // Misc
  fullCenter: {
    minHeight: '80vh',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 4,
  },
}
