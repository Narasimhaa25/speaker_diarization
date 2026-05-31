import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/analyse': 'http://localhost:5001',
      '/analyse_sample': 'http://localhost:5001',
      '/analyse_realtime': 'http://localhost:5001',
      '/enroll': 'http://localhost:5001',
      '/staff': 'http://localhost:5001',
    },
  },
  build: { outDir: 'dist' },
})
