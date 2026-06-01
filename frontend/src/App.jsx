/**
 * App.jsx — Root Component
 * ==========================
 * Router đơn giản dùng state (không cần react-router-dom).
 * Ghép: Dashboard → Examination → Report
 *
 * Nếu sau này muốn dùng react-router-dom, chỉ cần thay phần
 * navigate state bằng useNavigate() — logic các page không đổi.
 */

import { useState } from 'react'
import Dashboard   from './pages/Dashboard'
import Examination from './pages/Examination'
import Report      from './pages/Report'

export default function App() {
  // { name: 'dashboard' | 'examination' | 'report', patientId?, sessionId? }
  const [page, setPage] = useState({ name: 'dashboard' })

  function navigate(name, params = {}) {
    setPage({ name, ...params })
  }

  switch (page.name) {
    case 'examination':
      return (
        <Examination
          patientId={page.patientId ?? null}
          onBack={() => navigate('dashboard')}
          onFinish={(sessionId) => navigate('report', { sessionId })}
        />
      )

    case 'report':
      return (
        <Report
          sessionId={page.sessionId}
          onBack={() => navigate('dashboard')}
        />
      )

    default:
      return (
        <Dashboard
          onNavigate={(name, params) => navigate(name, params)}
        />
      )
  }
}
