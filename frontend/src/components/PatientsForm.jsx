/**
 * PatientForm.jsx — Form tạo bệnh nhân
 * =======================================
 * Form nhập thông tin bệnh nhân mới, POST /api/v1/patients.
 *
 * Props:
 *   onSuccess  {function(patient)}  — gọi sau khi tạo thành công
 *   onCancel   {function}           — gọi khi bấm Hủy
 */

import { useState } from 'react'
import api from '../services/apiService'

const GENDERS = [
  { value: 'male',   label: 'Nam' },
  { value: 'female', label: 'Nữ' },
  { value: 'other',  label: 'Khác' },
]

const CURRENT_YEAR = new Date().getFullYear()

export default function PatientForm({ onSuccess, onCancel }) {
  const [form, setForm]       = useState({
    full_name: '', birth_year: '', gender: 'male', phone: '', notes: '',
  })
  const [errors,  setErrors]  = useState({})
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')

  // ── Validation ──────────────────────────────────────────────────
  function validate() {
    const e = {}
    if (!form.full_name.trim()) e.full_name = 'Vui lòng nhập họ tên'
    else if (form.full_name.trim().length < 2) e.full_name = 'Họ tên ít nhất 2 ký tự'

    const year = parseInt(form.birth_year, 10)
    if (!form.birth_year) e.birth_year = 'Vui lòng nhập năm sinh'
    else if (isNaN(year) || year < 1900 || year > CURRENT_YEAR)
      e.birth_year = `Năm sinh phải từ 1900 đến ${CURRENT_YEAR}`

    if (form.phone && !/^[0-9+\- ]{6,20}$/.test(form.phone))
      e.phone = 'Số điện thoại không hợp lệ'

    setErrors(e)
    return Object.keys(e).length === 0
  }

  // ── Handlers ────────────────────────────────────────────────────
  function handleChange(field, value) {
    setForm((f) => ({ ...f, [field]: value }))
    if (errors[field]) setErrors((e) => ({ ...e, [field]: undefined }))
    setApiError('')
  }

  async function handleSubmit() {
    if (!validate()) return
    setLoading(true)
    setApiError('')
    try {
      const patient = await api.patients.create({
        full_name:  form.full_name.trim(),
        birth_year: parseInt(form.birth_year, 10),
        gender:     form.gender,
        phone:      form.phone.trim() || undefined,
        notes:      form.notes.trim() || undefined,
      })
      onSuccess?.(patient)
    } catch (err) {
      setApiError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ── Render ──────────────────────────────────────────────────────
  return (
    <div style={styles.wrap}>
      <h3 style={styles.heading}>👤 Thêm bệnh nhân mới</h3>

      <Field label="Họ và tên *" error={errors.full_name}>
        <input
          style={inputStyle(!!errors.full_name)}
          placeholder="Nguyễn Văn A"
          value={form.full_name}
          onChange={(e) => handleChange('full_name', e.target.value)}
          autoFocus
        />
      </Field>

      <div style={styles.row}>
        <Field label="Năm sinh *" error={errors.birth_year} style={{ flex: 1 }}>
          <input
            style={inputStyle(!!errors.birth_year)}
            placeholder="1955"
            type="number"
            min={1900} max={CURRENT_YEAR}
            value={form.birth_year}
            onChange={(e) => handleChange('birth_year', e.target.value)}
          />
        </Field>

        <Field label="Giới tính" style={{ flex: 1 }}>
          <div style={styles.genderRow}>
            {GENDERS.map(({ value, label }) => (
              <label key={value} style={styles.radioLabel}>
                <input
                  type="radio" name="gender" value={value}
                  checked={form.gender === value}
                  onChange={() => handleChange('gender', value)}
                  style={{ accentColor: '#10B981' }}
                />
                {label}
              </label>
            ))}
          </div>
        </Field>
      </div>

      <Field label="Số điện thoại" error={errors.phone}>
        <input
          style={inputStyle(!!errors.phone)}
          placeholder="0912 345 678"
          value={form.phone}
          onChange={(e) => handleChange('phone', e.target.value)}
        />
      </Field>

      <Field label="Ghi chú">
        <textarea
          style={{ ...inputStyle(false), resize: 'vertical', minHeight: 72 }}
          placeholder="Tiền sử bệnh, thuốc đang dùng…"
          value={form.notes}
          onChange={(e) => handleChange('notes', e.target.value)}
        />
      </Field>

      {apiError && <div style={styles.apiError}>{apiError}</div>}

      <div style={styles.actions}>
        <button style={styles.btnCancel} onClick={onCancel} disabled={loading}>
          Hủy
        </button>
        <button
          style={{ ...styles.btnSubmit, opacity: loading ? 0.7 : 1 }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? '⏳ Đang lưu…' : '✓ Tạo bệnh nhân'}
        </button>
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────
function Field({ label, error, children, style }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, ...style }}>
      <label style={styles.label}>{label}</label>
      {children}
      {error && <span style={styles.error}>{error}</span>}
    </div>
  )
}

// ── Style helpers ─────────────────────────────────────────────────
function inputStyle(hasError) {
  return {
    background: '#0D1117',
    border: `1px solid ${hasError ? '#EF4444' : '#2D3548'}`,
    borderRadius: 6, padding: '8px 12px',
    color: '#E2E8F0', fontSize: 13,
    outline: 'none', width: '100%',
    transition: 'border-color 0.15s',
    boxSizing: 'border-box',
  }
}

const styles = {
  wrap: {
    background: '#1E2433', border: '1px solid #2D3548',
    borderRadius: 12, padding: '20px 24px',
    display: 'flex', flexDirection: 'column', gap: 14,
    maxWidth: 480,
  },
  heading: { margin: 0, fontSize: 15, fontWeight: 600, color: '#E2E8F0' },
  label:   { fontSize: 12, color: '#94A3B8', fontWeight: 500 },
  error:   { fontSize: 11, color: '#F87171' },
  apiError: {
    background: '#2D1515', border: '1px solid #EF444440',
    borderRadius: 6, padding: '8px 12px',
    fontSize: 12, color: '#FCA5A5',
  },
  row:      { display: 'flex', gap: 12 },
  genderRow: { display: 'flex', gap: 14, alignItems: 'center', padding: '8px 0' },
  radioLabel: { display: 'flex', alignItems: 'center', gap: 5, fontSize: 13, color: '#CBD5E1', cursor: 'pointer' },
  actions:  { display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 4 },
  btnCancel: {
    padding: '8px 20px', borderRadius: 6,
    border: '1px solid #2D3548', background: 'transparent',
    color: '#94A3B8', fontSize: 13, cursor: 'pointer',
  },
  btnSubmit: {
    padding: '8px 20px', borderRadius: 6,
    border: 'none', background: '#10B981',
    color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer',
  },
}
