/**
 * Dashboard.jsx — Trang chủ bác sĩ
 * ===================================
 * Hiển thị danh sách bệnh nhân + lịch sử phiên khám gần đây.
 * Không phụ thuộc BLE — chỉ gọi REST API.
 *
 * Props: không có (lấy data qua api service)
 * Navigate:
 *   → /examination?patientId=X  (bắt đầu khám)
 *   → /report/:sessionId         (xem báo cáo)
 */

import { useState, useEffect, useCallback } from 'react'
import api from '../services/apiService'
import PatientForm from '../components/PatientsForm'

// ── Constants ─────────────────────────────────────────────────────────────

const DIAGNOSIS_CONFIG = {
  Normal:    { color: '#10B981', bg: '#10B98118', label: 'Bình thường',          dot: '#10B981' },
  Parkinson: { color: '#F59E0B', bg: '#F59E0B18', label: 'Nghi ngờ Parkinson',   dot: '#F59E0B' },
  Abnormal:  { color: '#EF4444', bg: '#EF444418', label: 'Dáng đi bất thường',   dot: '#EF4444' },
  active:    { color: '#3B82F6', bg: '#3B82F618', label: 'Đang khám',             dot: '#3B82F6' },
}

const GENDER_LABEL = { male: 'Nam', female: 'Nữ', other: 'Khác' }

const PER_PAGE = 10

// ── Helpers ───────────────────────────────────────────────────────────────

function calcAge(birthYear) {
  return new Date().getFullYear() - birthYear
}

function formatDate(isoString) {
  if (!isoString) return '—'
  const d = new Date(isoString)
  return d.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' })
}

function formatTime(isoString) {
  if (!isoString) return ''
  const d = new Date(isoString)
  return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Dashboard({ onNavigate }) {
  // State
  const [patients,     setPatients]     = useState([])
  const [total,        setTotal]        = useState(0)
  const [page,         setPage]         = useState(1)
  const [search,       setSearch]       = useState('')
  const [searchInput,  setSearchInput]  = useState('')
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')

  const [selected,     setSelected]     = useState(null)   // PatientResponse đang xem
  const [sessions,     setSessions]     = useState([])
  const [sessLoading,  setSessLoading]  = useState(false)

  const [showForm,     setShowForm]     = useState(false)

  // ── Fetch patients ──────────────────────────────────────────────
  const fetchPatients = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await api.patients.list({ page, per_page: PER_PAGE, search: search || undefined })
      setPatients(res.items ?? [])
      setTotal(res.total ?? 0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [page, search])

  useEffect(() => { fetchPatients() }, [fetchPatients])

  // ── Fetch sessions khi chọn bệnh nhân ──────────────────────────
  const fetchSessions = useCallback(async (patient) => {
    setSelected(patient)
    setSessions([])
    setSessLoading(true)
    try {
      const data = await api.patients.sessions(patient.id, 8)
      setSessions(data ?? [])
    } catch {
      setSessions([])
    } finally {
      setSessLoading(false)
    }
  }, [])

  // ── Search debounce ─────────────────────────────────────────────
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(searchInput)
      setPage(1)
    }, 350)
    return () => clearTimeout(t)
  }, [searchInput])

  // ── Patient created callback ────────────────────────────────────
  function handlePatientCreated(patient) {
    setShowForm(false)
    fetchPatients()
    fetchSessions(patient)
  }

  const totalPages = Math.ceil(total / PER_PAGE)

  // ── Render ──────────────────────────────────────────────────────
  return (
    <div style={s.root}>

      {/* ── Header ── */}
      <header style={s.header}>
        <div style={s.headerLeft}>
          <div style={s.logoMark}>SI</div>
          <div>
            <div style={s.appName}>Smart Insole</div>
            <div style={s.appSub}>Hệ thống phân tích dáng đi</div>
          </div>
        </div>
        <button style={s.btnPrimary} onClick={() => setShowForm(true)}>
          <span style={{ fontSize: 16 }}>+</span> Thêm bệnh nhân
        </button>
      </header>

      {/* ── Body: 2 column ── */}
      <div style={s.body}>

        {/* ── LEFT: danh sách bệnh nhân ── */}
        <section style={s.leftPane}>

          {/* Search */}
          <div style={s.searchWrap}>
            <span style={s.searchIcon}>⌕</span>
            <input
              style={s.searchInput}
              placeholder="Tìm theo tên bệnh nhân…"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
            />
            {searchInput && (
              <button style={s.clearBtn} onClick={() => setSearchInput('')}>✕</button>
            )}
          </div>

          {/* Count */}
          <div style={s.listMeta}>
            {loading ? 'Đang tải…' : `${total} bệnh nhân${search ? ` khớp "${search}"` : ''}`}
          </div>

          {/* Error */}
          {error && <div style={s.errorBar}>{error}</div>}

          {/* List */}
          <div style={s.patientList}>
            {loading && patients.length === 0 ? (
              <LoadingSkeleton />
            ) : patients.length === 0 ? (
              <EmptyPatients hasSearch={!!search} onAdd={() => setShowForm(true)} />
            ) : (
              patients.map(p => (
                <PatientRow
                  key={p.id}
                  patient={p}
                  isActive={selected?.id === p.id}
                  onClick={() => fetchSessions(p)}
                />
              ))
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={s.pagination}>
              <button
                style={s.pageBtn}
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
              >‹</button>
              <span style={s.pageInfo}>{page} / {totalPages}</span>
              <button
                style={s.pageBtn}
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
              >›</button>
            </div>
          )}
        </section>

        {/* ── RIGHT: chi tiết bệnh nhân ── */}
        <section style={s.rightPane}>
          {showForm ? (
            <PatientForm
              onSuccess={handlePatientCreated}
              onCancel={() => setShowForm(false)}
            />
          ) : selected ? (
            <PatientDetail
              patient={selected}
              sessions={sessions}
              loading={sessLoading}
              onStartExam={() => onNavigate?.('examination', { patientId: selected.id })}
              onViewReport={sessionId => onNavigate?.('report', { sessionId })}
            />
          ) : (
            <EmptyDetail onAdd={() => setShowForm(true)} />
          )}
        </section>

      </div>
    </div>
  )
}

// ── PatientRow ─────────────────────────────────────────────────────────────

function PatientRow({ patient, isActive, onClick }) {
  return (
    <button style={{ ...s.patientRow, ...(isActive ? s.patientRowActive : {}) }} onClick={onClick}>
      <div style={s.avatar}>
        {patient.full_name.charAt(0).toUpperCase()}
      </div>
      <div style={s.patientInfo}>
        <div style={s.patientName}>{patient.full_name}</div>
        <div style={s.patientMeta}>
          {calcAge(patient.birth_year)} tuổi · {GENDER_LABEL[patient.gender] ?? patient.gender}
          {patient.phone && <span> · {patient.phone}</span>}
        </div>
      </div>
      <div style={s.patientDate}>{formatDate(patient.created_at)}</div>
    </button>
  )
}

// ── PatientDetail ──────────────────────────────────────────────────────────

function PatientDetail({ patient, sessions, loading, onStartExam, onViewReport }) {
  return (
    <div style={s.detailWrap}>

      {/* Patient header */}
      <div style={s.detailHeader}>
        <div style={{ ...s.avatar, width: 52, height: 52, fontSize: 22, borderRadius: 14 }}>
          {patient.full_name.charAt(0).toUpperCase()}
        </div>
        <div style={{ flex: 1 }}>
          <div style={s.detailName}>{patient.full_name}</div>
          <div style={s.detailMeta}>
            {calcAge(patient.birth_year)} tuổi · {GENDER_LABEL[patient.gender] ?? patient.gender}
            {patient.phone && <> · {patient.phone}</>}
          </div>
        </div>
        <button style={s.btnExam} onClick={onStartExam}>
          ▶ Bắt đầu khám
        </button>
      </div>

      {/* Notes */}
      {patient.notes && (
        <div style={s.notesBox}>
          <span style={s.notesLabel}>Ghi chú</span>
          <span style={s.notesText}>{patient.notes}</span>
        </div>
      )}

      {/* Stats row */}
      <div style={s.statsRow}>
        <StatChip label="Tổng phiên" value={sessions.length} />
        <StatChip
          label="Nghi Parkinson"
          value={sessions.filter(s => s.diagnosis === 'Parkinson').length}
          color="#F59E0B"
        />
        <StatChip
          label="Bất thường"
          value={sessions.filter(s => s.diagnosis === 'Abnormal').length}
          color="#EF4444"
        />
      </div>

      {/* Sessions history */}
      <div style={s.sessionsSection}>
        <div style={s.sectionTitle}>Lịch sử phiên khám</div>

        {loading ? (
          <div style={s.sessLoading}>Đang tải…</div>
        ) : sessions.length === 0 ? (
          <div style={s.sessEmpty}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🩺</div>
            <div>Chưa có phiên khám nào</div>
            <button style={{ ...s.btnPrimary, marginTop: 12 }} onClick={onStartExam}>
              Bắt đầu khám ngay
            </button>
          </div>
        ) : (
          <div style={s.sessionList}>
            {sessions.map(sess => (
              <SessionRow
                key={sess.id}
                session={sess}
                onView={() => onViewReport?.(sess.id)}
              />
            ))}
          </div>
        )}
      </div>

    </div>
  )
}

// ── SessionRow ─────────────────────────────────────────────────────────────

function SessionRow({ session, onView }) {
  const isActive = session.status === 'active'
  const diagKey  = isActive ? 'active' : (session.diagnosis ?? null)
  const cfg      = diagKey ? DIAGNOSIS_CONFIG[diagKey] : null

  return (
    <div style={s.sessionRow}>
      <div style={s.sessionLeft}>
        {/* Status dot */}
        {cfg && <span style={{ ...s.dot, background: cfg.dot }} />}

        <div>
          <div style={s.sessionDate}>
            {formatDate(session.started_at)} &nbsp;
            <span style={s.sessionTime}>{formatTime(session.started_at)}</span>
          </div>
          {session.duration_sec != null && (
            <div style={s.sessionDuration}>
              {Math.round(session.duration_sec)}s đo
            </div>
          )}
        </div>
      </div>

      <div style={s.sessionRight}>
        {/* Diagnosis badge */}
        {cfg && (
          <span style={{ ...s.diagBadge, color: cfg.color, background: cfg.bg }}>
            {cfg.label}
          </span>
        )}

        {/* Action */}
        {isActive ? (
          <span style={s.sessionActive}>Đang chạy</span>
        ) : session.status === 'completed' ? (
          <button style={s.viewBtn} onClick={onView}>Xem báo cáo</button>
        ) : (
          <span style={s.sessionCancelled}>Đã hủy</span>
        )}
      </div>
    </div>
  )
}

// ── Minor sub-components ───────────────────────────────────────────────────

function StatChip({ label, value, color = '#E2E8F0' }) {
  return (
    <div style={s.statChip}>
      <span style={{ ...s.statValue, color }}>{value}</span>
      <span style={s.statLabel}>{label}</span>
    </div>
  )
}

function EmptyDetail({ onAdd }) {
  return (
    <div style={s.emptyDetail}>
      <div style={s.emptyDetailIcon}>👤</div>
      <div style={s.emptyDetailTitle}>Chọn bệnh nhân để xem chi tiết</div>
      <div style={s.emptyDetailSub}>hoặc tạo mới để bắt đầu</div>
      <button style={s.btnPrimary} onClick={onAdd}>+ Thêm bệnh nhân</button>
    </div>
  )
}

function EmptyPatients({ hasSearch, onAdd }) {
  return (
    <div style={s.emptyPatients}>
      <div style={{ fontSize: 32, marginBottom: 8 }}>{hasSearch ? '🔍' : '🏥'}</div>
      <div style={{ color: '#94A3B8', fontSize: 13 }}>
        {hasSearch ? 'Không tìm thấy bệnh nhân khớp' : 'Chưa có bệnh nhân nào'}
      </div>
      {!hasSearch && (
        <button style={{ ...s.btnPrimary, marginTop: 12 }} onClick={onAdd}>
          + Thêm bệnh nhân đầu tiên
        </button>
      )}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <>
      {[...Array(5)].map((_, i) => (
        <div key={i} style={{ ...s.patientRow, opacity: 0.4 }}>
          <div style={{ ...s.avatar, background: '#2D3548' }} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ height: 13, width: '60%', background: '#2D3548', borderRadius: 4 }} />
            <div style={{ height: 11, width: '40%', background: '#2D3548', borderRadius: 4 }} />
          </div>
        </div>
      ))}
    </>
  )
}

// ── Styles ─────────────────────────────────────────────────────────────────

const s = {
  root: {
    minHeight: '100vh',
    background: '#0D1117',
    color: '#E2E8F0',
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    display: 'flex',
    flexDirection: 'column',
  },

  // Header
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 28px',
    borderBottom: '1px solid #1E2433',
    background: '#0D1117',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 12 },
  logoMark: {
    width: 36, height: 36,
    background: '#10B981',
    borderRadius: 9,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13, fontWeight: 700, color: '#fff', letterSpacing: '0.05em',
    flexShrink: 0,
  },
  appName: { fontSize: 15, fontWeight: 600, color: '#F1F5F9' },
  appSub:  { fontSize: 11, color: '#475569', marginTop: 1 },

  // Body
  body: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: '340px 1fr',
    minHeight: 0,
  },

  // Left pane
  leftPane: {
    borderRight: '1px solid #1E2433',
    display: 'flex',
    flexDirection: 'column',
    padding: '20px 0 0',
    overflowY: 'auto',
    maxHeight: 'calc(100vh - 64px)',
  },
  searchWrap: {
    position: 'relative',
    margin: '0 16px 12px',
    display: 'flex',
    alignItems: 'center',
  },
  searchIcon: {
    position: 'absolute', left: 10,
    fontSize: 18, color: '#475569', pointerEvents: 'none',
  },
  searchInput: {
    width: '100%',
    background: '#1E2433',
    border: '1px solid #2D3548',
    borderRadius: 8,
    padding: '8px 32px 8px 32px',
    color: '#E2E8F0',
    fontSize: 13,
    outline: 'none',
    boxSizing: 'border-box',
  },
  clearBtn: {
    position: 'absolute', right: 8,
    background: 'none', border: 'none',
    color: '#475569', cursor: 'pointer',
    fontSize: 12, padding: '2px 4px',
  },
  listMeta: {
    fontSize: 11, color: '#475569',
    padding: '0 20px 8px',
    letterSpacing: '0.02em',
  },
  errorBar: {
    margin: '0 16px 10px',
    background: '#2D1515',
    border: '1px solid #EF444430',
    borderRadius: 6,
    padding: '8px 12px',
    fontSize: 12, color: '#FCA5A5',
  },
  patientList: {
    flex: 1,
    overflowY: 'auto',
  },
  patientRow: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 20px',
    background: 'transparent',
    border: 'none',
    borderLeft: '3px solid transparent',
    textAlign: 'left',
    cursor: 'pointer',
    transition: 'background 0.12s',
  },
  patientRowActive: {
    background: '#1E2433',
    borderLeftColor: '#10B981',
  },
  avatar: {
    width: 38, height: 38,
    borderRadius: 10,
    background: '#1E2433',
    border: '1px solid #2D3548',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 15, fontWeight: 600, color: '#94A3B8',
    flexShrink: 0,
  },
  patientInfo: { flex: 1, minWidth: 0 },
  patientName: {
    fontSize: 13, fontWeight: 500, color: '#E2E8F0',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  patientMeta: { fontSize: 11, color: '#475569', marginTop: 2 },
  patientDate: { fontSize: 11, color: '#334155', flexShrink: 0 },

  // Pagination
  pagination: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12,
    padding: '12px 0',
    borderTop: '1px solid #1E2433',
  },
  pageBtn: {
    width: 28, height: 28,
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 6, color: '#94A3B8',
    cursor: 'pointer', fontSize: 16,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  pageInfo: { fontSize: 12, color: '#475569' },

  // Right pane
  rightPane: {
    padding: '28px 32px',
    overflowY: 'auto',
    maxHeight: 'calc(100vh - 64px)',
  },

  // Detail
  detailWrap: { display: 'flex', flexDirection: 'column', gap: 20 },
  detailHeader: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '20px 24px',
    background: '#1E2433',
    border: '1px solid #2D3548',
    borderRadius: 12,
  },
  detailName: { fontSize: 18, fontWeight: 600, color: '#F1F5F9' },
  detailMeta: { fontSize: 13, color: '#64748B', marginTop: 3 },

  notesBox: {
    display: 'flex', gap: 8,
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 8, padding: '10px 14px',
  },
  notesLabel: { fontSize: 11, color: '#475569', flexShrink: 0, paddingTop: 1 },
  notesText:  { fontSize: 13, color: '#94A3B8', lineHeight: 1.5 },

  statsRow: {
    display: 'flex', gap: 12,
  },
  statChip: {
    flex: 1,
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 10, padding: '14px 10px',
    gap: 4,
  },
  statValue: { fontSize: 22, fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  statLabel: { fontSize: 11, color: '#475569', textAlign: 'center' },

  // Sessions
  sessionsSection: { display: 'flex', flexDirection: 'column', gap: 10 },
  sectionTitle: { fontSize: 12, color: '#475569', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' },
  sessLoading: { fontSize: 13, color: '#475569', padding: '20px 0', textAlign: 'center' },
  sessEmpty: {
    padding: '32px 0', textAlign: 'center',
    fontSize: 13, color: '#475569',
    display: 'flex', flexDirection: 'column', alignItems: 'center',
  },
  sessionList: { display: 'flex', flexDirection: 'column', gap: 8 },
  sessionRow: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 10, padding: '12px 16px',
  },
  sessionLeft:    { display: 'flex', alignItems: 'center', gap: 10 },
  sessionDate:    { fontSize: 13, color: '#CBD5E1', fontWeight: 500 },
  sessionTime:    { color: '#475569', fontWeight: 400 },
  sessionDuration:{ fontSize: 11, color: '#475569', marginTop: 2 },
  sessionRight:   { display: 'flex', alignItems: 'center', gap: 10 },
  sessionActive:  { fontSize: 11, color: '#3B82F6', fontWeight: 500 },
  sessionCancelled: { fontSize: 11, color: '#475569' },
  dot: {
    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
  },

  // Diagnosis badge
  diagBadge: {
    fontSize: 11, fontWeight: 500,
    padding: '3px 10px', borderRadius: 999,
  },

  // Buttons
  btnPrimary: {
    display: 'flex', alignItems: 'center', gap: 6,
    background: '#10B981', color: '#fff',
    border: 'none', borderRadius: 8,
    padding: '9px 18px',
    fontSize: 13, fontWeight: 600, cursor: 'pointer',
    flexShrink: 0,
    transition: 'opacity 0.15s',
  },
  btnExam: {
    background: '#10B981', color: '#fff',
    border: 'none', borderRadius: 8,
    padding: '9px 18px',
    fontSize: 13, fontWeight: 600, cursor: 'pointer',
    flexShrink: 0,
    whiteSpace: 'nowrap',
  },
  viewBtn: {
    background: 'transparent',
    border: '1px solid #2D3548',
    borderRadius: 6, padding: '5px 12px',
    color: '#94A3B8', fontSize: 11,
    cursor: 'pointer',
  },

  // Empty states
  emptyDetail: {
    height: '60vh',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 8, opacity: 0.7,
  },
  emptyDetailIcon:  { fontSize: 40, marginBottom: 4 },
  emptyDetailTitle: { fontSize: 15, color: '#94A3B8', fontWeight: 500 },
  emptyDetailSub:   { fontSize: 13, color: '#475569' },
  emptyPatients: {
    padding: '32px 0',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', gap: 4,
  },
}
