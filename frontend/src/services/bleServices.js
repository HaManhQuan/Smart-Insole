/**
 * bleService.js — Web Bluetooth API Layer
 * ==========================================
 * Quản lý toàn bộ vòng đời BLE: scan → connect → subscribe notify → parse packet → disconnect.
 *
 * Dùng trong useBLE.js hook:
 *   import bleService from './bleService'
 *   await bleService.connect(onSample, onStatusChange)
 *   bleService.disconnect()
 *
 * Packet format từ firmware (smart_insole.ino):
 *   8 bytes: [S1_H, S1_L, S2_H, S2_L, S3_H, S3_L, S4_H, S4_L]
 *   S1, S2: ADS1115 16-bit (0–32767)
 *   S3, S4: ESP32 ADC 12-bit (0–4095)
 *
 * Web Bluetooth chỉ chạy trên Chrome / Edge (HTTPS hoặc localhost).
 */

// ---------------------------------------------------------------------------
// BLE UUIDs — phải khớp với firmware smart_insole.ino
// ---------------------------------------------------------------------------

const BLE_SERVICE_UUID        = '12345678-1234-5678-1234-56789abcdef0'
const BLE_CHARACTERISTIC_UUID = '12345678-1234-5678-1234-56789abcdef1'
const DEVICE_NAME_PREFIX      = 'SmartInsole'

// ---------------------------------------------------------------------------
// Connection status enum
// ---------------------------------------------------------------------------

export const BLEStatus = {
  DISCONNECTED: 'disconnected',
  SCANNING:     'scanning',
  CONNECTING:   'connecting',
  CONNECTED:    'connected',
  ERROR:        'error',
}

// ---------------------------------------------------------------------------
// Parse packet 8 bytes → 4 sensor values
// ---------------------------------------------------------------------------

/**
 * Parse ArrayBuffer 8 bytes thành object {s1, s2, s3, s4, ts}.
 *
 * Byte layout:
 *   [0,1] = S1 big-endian uint16  (ESP32 ADC)
 *   [2,3] = S2 big-endian uint16  (ADS1115)
 *   [4,5] = S3 big-endian uint16  (ESP32 ADC)
 *   [6,7] = S4 big-endian uint16  (ESP32 ADC)
 *
 * @param {ArrayBuffer} buffer
 * @returns {{s1: number, s2: number, s3: number, s4: number, ts: number}}
 */
export function parsePacket(buffer) {
  if (buffer.byteLength < 8) {
    throw new Error(`Packet quá ngắn: ${buffer.byteLength} bytes, cần ≥ 8`)
  }

  const view = new DataView(buffer)
  return {
    s1: view.getUint16(0, false),   // false = big-endian
    s2: view.getUint16(2, false),
    s3: view.getUint16(4, false),
    s4: view.getUint16(6, false),
    ts: Date.now(),                 // timestamp ms khi nhận packet
  }
}

// ---------------------------------------------------------------------------
// BLEService class
// ---------------------------------------------------------------------------

class BLEService {
  constructor() {
    this._device         = null
    this._server         = null
    this._characteristic = null
    this._status         = BLEStatus.DISCONNECTED
    this._onSample       = null    // callback(sample) mỗi khi nhận packet
    this._onStatus       = null    // callback(status, message?) khi trạng thái thay đổi
    this._sampleCount    = 0
    this._startTime      = null
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Bắt đầu scan và kết nối thiết bị.
   *
   * @param {function} onSample   - callback nhận {s1,s2,s3,s4,ts} mỗi sample
   * @param {function} onStatus   - callback nhận (BLEStatus, message?)
   * @returns {Promise<void>}
   *
   * @throws {Error} nếu browser không hỗ trợ Web Bluetooth
   * @throws {Error} nếu người dùng cancel dialog chọn thiết bị
   */
  async connect(onSample, onStatus) {
    this._onSample = onSample
    this._onStatus = onStatus

    if (!navigator.bluetooth) {
      const msg = 'Web Bluetooth không được hỗ trợ. Dùng Chrome hoặc Edge.'
      this._setStatus(BLEStatus.ERROR, msg)
      throw new Error(msg)
    }

    try {
      // 1. Scan — mở dialog chọn thiết bị
      this._setStatus(BLEStatus.SCANNING)
      this._device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: [BLE_SERVICE_UUID],
      })

      // Lắng nghe sự kiện thiết bị bị ngắt kết nối
      this._device.addEventListener('gattserverdisconnected', this._onDisconnected.bind(this))

      // 2. Kết nối GATT server
      this._setStatus(BLEStatus.CONNECTING)
      this._server = await this._device.gatt.connect()

      // 3. Lấy service
      const service = await this._server.getPrimaryService(BLE_SERVICE_UUID)

      // 4. Lấy characteristic
      this._characteristic = await service.getCharacteristic(BLE_CHARACTERISTIC_UUID)

      // 5. Subscribe notifications
      await this._characteristic.startNotifications()
      this._characteristic.addEventListener(
        'characteristicvaluechanged',
        this._onNotify.bind(this),
      )

      this._sampleCount = 0
      this._startTime   = Date.now()
      this._setStatus(BLEStatus.CONNECTED, `Đã kết nối: ${this._device.name}`)

    } catch (err) {
      if (err.name === 'NotFoundError') {
        // Người dùng cancel dialog — không phải lỗi thật
        this._setStatus(BLEStatus.DISCONNECTED, 'Đã hủy chọn thiết bị')
      } else {
        this._setStatus(BLEStatus.ERROR, `Lỗi kết nối: ${err.message}`)
      }
      throw err
    }
  }

  /**
   * Ngắt kết nối an toàn.
   */
  disconnect() {
    if (this._characteristic) {
      try {
        this._characteristic.removeEventListener(
          'characteristicvaluechanged',
          this._onNotify.bind(this),
        )
        this._characteristic.stopNotifications().catch(() => {})
      } catch (_) {}
      this._characteristic = null
    }

    if (this._server?.connected) {
      try { this._server.disconnect() } catch (_) {}
    }

    this._device  = null
    this._server  = null
    this._setStatus(BLEStatus.DISCONNECTED)
  }

  /** Tên thiết bị đang kết nối (null nếu chưa kết nối). */
  get deviceName() {
    return this._device?.name ?? null
  }

  /** Trạng thái hiện tại. */
  get status() {
    return this._status
  }

  get isConnected() {
    return this._status === BLEStatus.CONNECTED
  }

  /** Số sample đã nhận từ lúc kết nối. */
  get sampleCount() {
    return this._sampleCount
  }

  /**
   * Tần số thực tế (Hz) tính từ lúc kết nối.
   * Dùng để debug — nên xấp xỉ 50Hz.
   */
  get actualHz() {
    if (!this._startTime || this._sampleCount === 0) return 0
    const elapsedSec = (Date.now() - this._startTime) / 1000
    return elapsedSec > 0 ? this._sampleCount / elapsedSec : 0
  }

  // ------------------------------------------------------------------
  // Private handlers
  // ------------------------------------------------------------------

  /**
   * Gọi khi nhận BLE notification từ firmware.
   * @param {Event} event
   */
  _onNotify(event) {
    try {
      const sample = parsePacket(event.target.value.buffer)
      this._sampleCount++

      if (this._onSample) {
        this._onSample(sample)
      }
    } catch (err) {
      console.warn('[BLE] Parse packet error:', err.message)
    }
  }

  /**
   * Gọi khi thiết bị ngắt kết nối ngoài ý muốn (hết pin, ra xa...).
   */
  _onDisconnected() {
    console.warn('[BLE] Device disconnected unexpectedly')
    this._characteristic = null
    this._server         = null
    this._setStatus(BLEStatus.DISCONNECTED, 'Thiết bị bị ngắt kết nối')
  }

  _setStatus(status, message = null) {
    this._status = status
    if (this._onStatus) {
      this._onStatus(status, message)
    }
    console.log(`[BLE] Status: ${status}${message ? ' — ' + message : ''}`)
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

const bleService = new BLEService()
export default bleService