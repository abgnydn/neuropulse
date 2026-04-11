import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  root: '.',
  base: './',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        // Landing page at /
        main: resolve(__dirname, 'index.html'),
        // Live demo at /app/
        app: resolve(__dirname, 'app/index.html'),
      },
    },
  },
  server: { port: 4000 },
})
