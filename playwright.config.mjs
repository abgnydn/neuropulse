import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  fullyParallel: false,
  retries: 2,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4000',
    trace: 'off',
    headless: true,
  },
})
