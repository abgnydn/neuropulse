import { defineConfig } from 'vite'
import { existsSync, readdirSync, statSync, createReadStream } from 'node:fs'
import { join, resolve } from 'node:path'
import { homedir } from 'node:os'

/**
 * Local MLC-weights mirror for e2e testing without re-downloading 2 GB.
 *
 * Prime once:
 *   huggingface-cli download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC
 *
 * Serves:
 *   ~/mlc-weights/Phi-3-mini-4k-instruct-q4f16_1-MLC/*           → /local-weights/*
 *   (or)   ~/.cache/huggingface/hub/models--mlc-ai--Phi-3-.../
 *          snapshots/<hash>/*                                    → /local-weights/*
 *
 * Weight loader tries /local-weights/<file> first (tier 0) when
 * `import.meta.env.DEV`. Falls through to OPFS / browser cache / HF if unprimed.
 */
function findMlcSnapshotDir(): string | null {
  const flatMirror = join(homedir(), 'mlc-weights', 'Phi-3-mini-4k-instruct-q4f16_1-MLC')
  if (existsSync(join(flatMirror, 'ndarray-cache.json'))) return flatMirror

  const cacheRoot = join(
    homedir(),
    '.cache',
    'huggingface',
    'hub',
    'models--mlc-ai--Phi-3-mini-4k-instruct-q4f16_1-MLC',
    'snapshots',
  )
  if (!existsSync(cacheRoot)) return null
  const entries = readdirSync(cacheRoot)
    .map((name) => join(cacheRoot, name))
    .filter((p) => statSync(p).isDirectory())
  entries.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
  return entries[0] ?? null
}

function localWeightsPlugin() {
  return {
    name: 'local-mlc-weights',
    configureServer(server: import('vite').ViteDevServer) {
      const snapshot = findMlcSnapshotDir()
      if (!snapshot) {
        console.log('[local-weights] No MLC snapshot found — /local-weights/* will 404.')
        console.log('[local-weights] Prime with: huggingface-cli download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC')
        return
      }
      console.log(`[local-weights] Serving /local-weights/* from ${snapshot}`)
      server.middlewares.use('/local-weights', (req, res) => {
        const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0])
        const target = resolve(snapshot, '.' + urlPath)
        // Path-traversal guard
        if (!target.startsWith(snapshot)) {
          res.statusCode = 403
          res.end('Forbidden')
          return
        }
        if (!existsSync(target) || !statSync(target).isFile()) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const st = statSync(target)
        res.setHeader('Content-Length', String(st.size))
        res.setHeader(
          'Content-Type',
          target.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        )
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        createReadStream(target).pipe(res)
      })
    },
  }
}

export default defineConfig({
  root: '.',
  base: './',
  plugins: [localWeightsPlugin()],
  server: {
    port: 4000,
    fs: {
      // Allow serving files from the HF cache and mlc-weights mirror outside repo root.
      allow: [
        resolve(__dirname),
        join(homedir(), '.cache', 'huggingface'),
        join(homedir(), 'mlc-weights'),
      ],
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        app: resolve(__dirname, 'app/index.html'),
      },
    },
  },
})
