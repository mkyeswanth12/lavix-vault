import React from 'react'
import { createRoot } from 'react-dom/client'
import App, { ErrorBoundary } from './App.jsx'
import './thumbnail-generators.js'
import 'highlight.js/styles/github-dark.min.css'
import './index.css'

const root = createRoot(document.getElementById('root'))
root.render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
)
