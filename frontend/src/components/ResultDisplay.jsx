/**
 * ResultDisplay.jsx — Hiển thị kết quả ML
 * ==========================================
 * Hiển thị kết quả predict: label lớn + confidence bar + vote distribution.
 * Dùng cho cả window_result (real-time) và diagnosis (tổng kết).
 *
 * Props:
 *   windowResult  {WindowPrediction|null}   kết quả window gần nhất
 *   diagnosis     {SessionDiagnosis|null}   kết quả tổng hợp session
 *   isActive      {boolean}                 session đang chạy
 */

const LABEL_CONFIG = {
  Normal:    { icon: '✅', color: '#10B981', bg: '#10B98115', text: 'Bình thường' },
  Parkinson: { icon: '⚠️', color: '#F59E0B', bg: '#F59E0B15', text: 'Nghi ngờ Parkinson' },
  Abnormal:  { icon: '🔴', color: '#EF4444', bg: '#EF444415', text: 'Dáng đi bất thường' },
  Uncertain: { icon: '❓', color: '#6B7280', bg: '#6B728015', text: 'Chưa xác định' },
}

const CLASS_NAMES = ['Normal', 'Parkinson', 'Abnormal']

export default function ResultDisplay({ windowResult, diagnosis, isActive }) {

  // ── Ưu tiên hiển thị diagnosis nếu session đã kết thúc ──────────
  const primary = diagnosis ?? windowResult

  if (!primary && !isActive) {
    return (
      <div style={styles.empty}>
        <span style={styles.emptyIcon}>🧠</span>
        <span style={styles.emptyText}>Kết quả sẽ hiển thị sau khi bắt đầu đo</span>
      </div>
    )
  }

  if (!primary && isActive) {
    return (
      <div style={styles.empty}>
        <span style={{ ...styles.emptyIcon, animation: 'spin 2s linear infinite' }}>⚙️</span>
        <span style={styles.emptyText}>Đang tích lũy dữ liệu… (cần 2 giây)</span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  const label   = primary?.diagnosis ?? primary?.label ?? 'Uncertain'
  const cfg     = LABEL_CONFIG[label] ?? LABEL_CONFIG.Uncertain
  const conf    = primary?.confidence_mean ?? primary?.confidence ?? 0
  const isDiag  = !!diagnosis
  const probs   = primary?.probabilities ?? primary?.vote_distribution

  return (
    <div style={styles.wrap}>

      {/* Badge: window vs diagnosis */}
      <div style={styles.modeBadge}>
        {isDiag
          ? '📋 Chẩn đoán tổng hợp'
          : `🔄 Window #${primary?.window_id ?? 0}`}
      </div>

      {/* Label lớn */}
      <div style={{ ...styles.labelCard, background: cfg.bg, borderColor: cfg.color + '50' }}>
        <span style={styles.labelIcon}>{cfg.icon}</span>
        <div>
          <div style={{ ...styles.labelText, color: cfg.color }}>{cfg.text}</div>
          {primary?.is_uncertain && (
            <div style={styles.uncertainNote}>Confidence thấp — cần thêm dữ liệu</div>
          )}
        </div>
      </div>

      {/* Confidence bar */}
      <div style={styles.confWrap}>
        <div style={styles.confHeader}>
          <span style={styles.confLabel}>Confidence</span>
          <span style={{ ...styles.confValue, color: cfg.color }}>
            {(conf * 100).toFixed(1)}%
          </span>
        </div>
        <div style={styles.confTrack}>
          <div style={{
            ...styles.confFill,
            width: `${conf * 100}%`,
            background: cfg.color,
            boxShadow: `0 0 8px ${cfg.color}60`,
          }} />
        </div>
      </div>

      {/* Phân bố xác suất / vote */}
      {probs && (
        <div style={styles.probsWrap}>
          <div style={styles.probsTitle}>
            {isDiag ? 'Phân bố vote' : 'Xác suất từng class'}
          </div>
          {CLASS_NAMES.map((name) => {
            const raw = probs[name] ?? 0
            // vote distribution là số nguyên, probabilities là 0-1 float
            const pct = isDiag
              ? (primary.total_windows > 0 ? raw / primary.total_windows : 0)
              : raw
            const c = LABEL_CONFIG[name]
            return (
              <div key={name} style={styles.probRow}>
                <span style={styles.probName}>{name}</span>
                <div style={styles.probTrack}>
                  <div style={{
                    ...styles.probFill,
                    width: `${pct * 100}%`,
                    background: c.color,
                    opacity: name === label ? 1 : 0.4,
                  }} />
                </div>
                <span style={{ ...styles.probPct, color: c.color }}>
                  {isDiag ? `${raw}` : `${(pct * 100).toFixed(0)}%`}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {/* Thống kê session nếu là diagnosis */}
      {isDiag && (
        <div style={styles.statsGrid}>
          <StatBox label="Tổng windows"    value={diagnosis.total_windows} />
          <StatBox label="Confident"       value={diagnosis.confident_windows} color="#10B981" />
          <StatBox label="Tỉ lệ"          value={`${(diagnosis.confident_ratio * 100).toFixed(0)}%`} />
          <StatBox label="Đủ dữ liệu"     value={diagnosis.sufficient_data ? 'Có' : 'Không'}
                   color={diagnosis.sufficient_data ? '#10B981' : '#EF4444'} />
        </div>
      )}

    </div>
  )
}

function StatBox({ label, value, color = '#E2E8F0' }) {
  return (
    <div style={styles.statBox}>
      <span style={{ ...styles.statValue, color }}>{value}</span>
      <span style={styles.statLabel}>{label}</span>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────
const styles = {
  wrap: {
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 12, padding: '16px 20px',
    display: 'flex', flexDirection: 'column', gap: 14,
  },
  empty: {
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 12, padding: '32px 20px',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', gap: 8,
  },
  emptyIcon: { fontSize: 28 },
  emptyText: { fontSize: 13, color: '#475569', textAlign: 'center' },
  modeBadge: {
    fontSize: 11, color: '#64748B',
    fontFamily: 'monospace', letterSpacing: '0.03em',
  },
  labelCard: {
    display: 'flex', alignItems: 'center', gap: 12,
    border: '1px solid', borderRadius: 10, padding: '14px 16px',
  },
  labelIcon: { fontSize: 24, flexShrink: 0 },
  labelText: { fontSize: 17, fontWeight: 700, letterSpacing: '-0.01em' },
  uncertainNote: { fontSize: 11, color: '#6B7280', marginTop: 3 },
  confWrap: { display: 'flex', flexDirection: 'column', gap: 6 },
  confHeader: { display: 'flex', justifyContent: 'space-between' },
  confLabel: { fontSize: 11, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.06em' },
  confValue: { fontSize: 13, fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  confTrack: { height: 6, background: '#0D1117', borderRadius: 999, overflow: 'hidden' },
  confFill:  { height: '100%', borderRadius: 999, transition: 'width 0.4s ease' },
  probsWrap: { display: 'flex', flexDirection: 'column', gap: 7 },
  probsTitle: { fontSize: 11, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.06em' },
  probRow:   { display: 'flex', alignItems: 'center', gap: 8 },
  probName:  { fontSize: 11, color: '#94A3B8', width: 72, flexShrink: 0 },
  probTrack: { flex: 1, height: 4, background: '#0D1117', borderRadius: 999, overflow: 'hidden' },
  probFill:  { height: '100%', borderRadius: 999, transition: 'width 0.4s ease' },
  probPct:   { fontSize: 11, fontVariantNumeric: 'tabular-nums', width: 36, textAlign: 'right' },
  statsGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr',
    gap: 8, background: '#0D1117', borderRadius: 8, padding: 12,
  },
  statBox:   { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 },
  statValue: { fontSize: 16, fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  statLabel: { fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' },
}
