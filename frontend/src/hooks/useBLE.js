/**
 * useBLE.js — Custom React Hook
 * ================================
 * Bọc bleService.js thành React hook — quản lý state BLE trong component.
 *
 * Dùng trong bất kỳ component nào cần BLE:
 *   const { isConnected, status, sensorData, connect, disconnect, stats } = useBLE()
 *
 * Tích hợp với WebSocket:
 *   const { isConnected, connect, disconnect } = useBLE({ wsConnection })
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import bleService, { BLEStatus } from '../services/bleServices'

/**
 * @param {object} options
 * @param {object|null} options.wsConnection  - WSConnection từ apiService, nếu null thì
 *                                              hook tự buffer sample (dùng cho REST mode)
 * @param {number} options.bufferSize         - Số sample giữ lại cho chart (default: 200)
 */
export default function useBLE({ wsConnection = null, bufferSize = 200 } = {}) {

  // BLE state
  const [status,      setStatus]      = useState(BLEStatus.DISCONNECTED)
  const [statusMsg,   setStatusMsg]   = useState('')
  const [deviceName,  setDeviceName]  = useState(null)

  // Sensor data — giữ bufferSize samples gần nhất cho chart
  const [sensorData, setSensorData] = useState({
    s1: [], s2: [], s3: [], s4: [], timestamps: [],
  })

  // Stats
  const [sampleCount, setSampleCount] = useState(0)
  const [actualHz,    setActualHz]    = useState(0)

  // Ref để tránh stale closure trong callback
  const wsRef        = useRef(wsConnection)
  const bufferRef    = useRef({ s1: [], s2: [], s3: [], s4: [], timestamps: [] })
  const hzTimerRef   = useRef(null)

  // Đồng bộ wsConnection ref khi prop thay đổi
  useEffect(() => { wsRef.current = wsConnection }, [wsConnection])

  // Cleanup khi unmount
  useEffect(() => {
    return () => {
      bleService.disconnect()
      if (hzTimerRef.current) clearInterval(hzTimerRef.current)
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Callback xử lý mỗi sample BLE (50Hz)
  // ---------------------------------------------------------------------------

  const handleSample = useCallback((sample) => {
    // 1. Gửi qua WebSocket nếu có connection
    if (wsRef.current?.isOpen) {
      wsRef.current.sendSample(sample)
    }

    // 2. Cập nhật buffer cho chart
    const buf = bufferRef.current
    buf.s1.push(sample.s1)
    buf.s2.push(sample.s2)
    buf.s3.push(sample.s3)
    buf.s4.push(sample.s4)
    buf.timestamps.push(sample.ts)

    // Giữ bufferSize samples gần nhất
    if (buf.s1.length > bufferSize) {
      buf.s1.shift(); buf.s2.shift()
      buf.s3.shift(); buf.s4.shift()
      buf.timestamps.shift()
    }

    // 3. Cập nhật React state — throttle để không re-render 50 lần/giây
    // setSensorData mỗi 5 sample = 10Hz re-render (đủ mượt cho chart)
    if (buf.s1.length % 5 === 0) {
      setSensorData({
        s1:         [...buf.s1],
        s2:         [...buf.s2],
        s3:         [...buf.s3],
        s4:         [...buf.s4],
        timestamps: [...buf.timestamps],
      })
    }

    setSampleCount((n) => n + 1)
  }, [bufferSize])

  // ---------------------------------------------------------------------------
  // Callback khi BLE status thay đổi
  // ---------------------------------------------------------------------------

  const handleStatus = useCallback((newStatus, message) => {
    setStatus(newStatus)
    setStatusMsg(message || '')

    if (newStatus === BLEStatus.CONNECTED) {
      setDeviceName(bleService.deviceName)
      // Cập nhật Hz mỗi 2 giây
      hzTimerRef.current = setInterval(() => {
        setActualHz(Math.round(bleService.actualHz * 10) / 10)
      }, 2000)
    }

    if (newStatus === BLEStatus.DISCONNECTED || newStatus === BLEStatus.ERROR) {
      setDeviceName(null)
      if (hzTimerRef.current) {
        clearInterval(hzTimerRef.current)
        hzTimerRef.current = null
      }
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Public actions
  // ---------------------------------------------------------------------------

  const connect = useCallback(async () => {
    // Reset buffer và stats
    bufferRef.current = { s1: [], s2: [], s3: [], s4: [], timestamps: [] }
    setSensorData({ s1: [], s2: [], s3: [], s4: [], timestamps: [] })
    setSampleCount(0)
    setActualHz(0)

    await bleService.connect(handleSample, handleStatus)
  }, [handleSample, handleStatus])

  const disconnect = useCallback(() => {
    bleService.disconnect()
    if (hzTimerRef.current) {
      clearInterval(hzTimerRef.current)
      hzTimerRef.current = null
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------

  return {
    // State
    status,
    statusMsg,
    deviceName,
    isConnected:    status === BLEStatus.CONNECTED,
    isScanning:     status === BLEStatus.SCANNING || status === BLEStatus.CONNECTING,
    hasError:       status === BLEStatus.ERROR,

    // Data
    sensorData,     // {s1[], s2[], s3[], s4[], timestamps[]} — bufferSize samples gần nhất

    // Stats
    stats: {
      sampleCount,
      actualHz,
      durationSec: sampleCount > 0 ? Math.round(sampleCount / 50) : 0,
    },

    // Actions
    connect,
    disconnect,
  }
}