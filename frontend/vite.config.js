import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: process.env.PORT || 5173,
     allowedHosts: ['http://127.0.0.1:8000/predict','http://localhost:5173'],
  },
})
