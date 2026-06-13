/**
 * apiService.js — FastAPI Client
 * ================================
 * Axios wrapper gọi tất cả REST endpoints + WebSocket helper.
 *
 * Dùng trong components và hooks:
 *   import api from './apiService'
 *   const patient = await api.patients.create({...})
 *   const ws = api.ws.connect(sessionId, onMessage)
 */

import axios from 'axios'

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const WS_URL   = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1')
  .replace(/^http/, 'ws')   // http→ws, https→wss

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

// ---------------------------------------------------------------------------
// Interceptors
// ---------------------------------------------------------------------------

// Request: log trong dev
http.interceptors.request.use((config) => {
  if (import.meta.env.DEV) {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
  }
  return config
})

// Response: chuẩn hóa lỗi
http.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const status  = err.response?.status
    const detail  = err.response?.data?.detail || err.response?.data?.error || err.message
    const message = `[${status ?? 'Network'}] ${detail}`
    console.error('[API Error]', message)
    return Promise.reject(new Error(message))
  },
)

// ---------------------------------------------------------------------------
// Patients
// ---------------------------------------------------------------------------

const patients = {
  /**
   * Tạo bệnh nhân mới.
   * @param {{ full_name, birth_year, gender, phone?, notes? }} data
   * @returns {Promise<PatientResponse>}
   */
  create: (data) => http.post('/patients', data),

  /**
   * Danh sách bệnh nhân (phân trang).
   * @param {{ page?, per_page?, search? }} params
   */
  list: (params = {}) => http.get('/patients', { params }),

  /**
   * Chi tiết 1 bệnh nhân.
   * @param {number} id
   */
  get: (id) => http.get(`/patients/${id}`),

  /**
   * Lịch sử phiên khám của bệnh nhân.
   * @param {number} id
   * @param {number} limit
   */
  sessions: (id, limit = 10) => http.get(`/patients/${id}/sessions`, { params: { limit } }),
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

const sessions = {
  /**
   * Bắt đầu phiên khám mới.
   * @param {{ patient_id, notes? }} data
   */
  create: (data) => http.post('/sessions', data),

  /**
   * Chi tiết phiên.
   * @param {number} id
   */
  get: (id) => http.get(`/sessions/${id}`),

  /**
   * Kết thúc phiên + nhận chẩn đoán tổng hợp.
   * @param {number} id
   * @param {{ notes? }} data
   * @returns {Promise<SessionDiagnosis>}
   */
  end: (id, data = {}) => http.patch(`/sessions/${id}/end`, data),

  /**
   * Ghi 1 window sensor vào InfluxDB.
   * Dùng khi không dùng WebSocket.
   * @param {number} id
   * @param {SensorWindow} window
   */
  ingestWindow: (id, window) => http.post(`/sessions/${id}/data`, window),

  /**
   * Lịch sử predict của phiên.
   * @param {number} id
   */
  predictions: (id) => http.get(`/sessions/${id}/predictions`),

  /**
   * Chẩn đoán tổng hợp (sau khi session kết thúc).
   * @param {number} id
   */
  diagnosis: (id) => http.get(`/sessions/${id}/diagnosis`),
}

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------

const predictions = {
  /**
   * Predict 1 window qua REST (thay thế khi không dùng WebSocket).
   * @param {{ session_id, window: SensorWindow }} data
   * @returns {Promise<PredictionResponse>}
   */
  predict: (data) => http.post('/predict', data),

  /**
   * Thông tin model đang chạy.
   * @returns {Promise<ModelInfo>}
   */
  modelInfo: () => http.get('/model/info'),
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

const BACKEND_ROOT = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1')
  .replace(/\/api\/v1$/, '')

const health = {
  check: () => http.get('/health', { baseURL: BACKEND_ROOT }),
}

// ---------------------------------------------------------------------------
// WebSocket helper
// ---------------------------------------------------------------------------

/**
 * WSConnection — quản lý 1 WebSocket session.
 *
 * Dùng:
 *   const conn = ws.connect(sessionId, {
 *     onWindowResult: (pred) => updateChart(pred),
 *     onDiagnosis:    (diag) => showResult(diag),
 *     onError:        (msg)  => alert(msg),
 *     onClose:        ()     => setConnected(false),
 *   })
 *
 *   // Gửi sample BLE
 *   conn.sendSample({ s1, s2, s3, s4, ts })
 *
 *   // Kết thúc session
 *   conn.endSession()
 *
 *   // Đóng
 *   conn.close()
 */
class WSConnection {
  constructor(sessionId, callbacks = {}) {
    this._sessionId = sessionId
    this._callbacks = callbacks
    this._socket    = null
    this._isOpen    = false

    const url = `${WS_URL}/ws/session/${sessionId}`
    this._socket = new WebSocket(url)

    this._socket.onopen = () => {
      this._isOpen = true
      console.log(`[WS] Connected: session=${sessionId}`)
    }

    this._socket.onmessage = (event) => {
      try {
        const { type, payload } = JSON.parse(event.data)
        this._dispatch(type, payload)
      } catch (err) {
        console.warn('[WS] Parse error:', err)
      }
    }

    this._socket.onerror = (err) => {
      console.error('[WS] Error:', err)
      callbacks.onError?.('WebSocket error')
    }

    this._socket.onclose = (event) => {
      this._isOpen = false
      console.log(`[WS] Closed: code=${event.code}`)
      callbacks.onClose?.()
    }
  }

  /**
   * Gửi 1 ADC sample từ BLE lên backend.
   * @param {{ s1, s2, s3, s4, ts }} sample
   */
  sendSample({ s1, s2, s3, s4, ts }) {
    if (!this._isOpen) return
    this._send('sensor_sample', { s1, s2, s3, s4, ts })
  }

  /**
   * Báo hiệu kết thúc session — backend sẽ flush và trả về diagnosis.
   */
  endSession() {
    if (!this._isOpen) return
    this._send('session_end', {})
  }

  /** Đóng WebSocket. */
  close() {
    if (this._socket) {
      this._socket.close()
      this._socket = null
    }
  }

  get isOpen() { return this._isOpen }

  _send(type, payload) {
    if (this._socket?.readyState === WebSocket.OPEN) {
      this._socket.send(JSON.stringify({ type, payload }))
    }
  }

  _dispatch(type, payload) {
    switch (type) {
      case 'window_result':
        this._callbacks.onWindowResult?.(payload)
        break
      case 'diagnosis':
        this._callbacks.onDiagnosis?.(payload)
        break
      case 'error':
        console.error('[WS] Server error:', payload.message)
        this._callbacks.onError?.(payload.message)
        break
      default:
        console.warn('[WS] Unknown message type:', type)
    }
  }
}

const ws = {
  /**
   * Tạo WebSocket connection mới cho 1 session.
   * @param {number} sessionId
   * @param {{ onWindowResult, onDiagnosis, onError, onClose }} callbacks
   * @returns {WSConnection}
   */
  connect: (sessionId, callbacks) => new WSConnection(sessionId, callbacks),
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

const api = { patients, sessions, predictions, health, ws }
export default api

// Named exports để dùng trực tiếp nếu cần
export { patients, sessions, predictions, health, ws, WSConnection }