import { defineConfig } from 'vite';

// Multi-page setup: every HTML file listed in rollupOptions.input becomes a
// separate entry. The dev server serves them by path automatically — visit
// `/` for the main FC-network teaching page, `/convolutional/` for the
// stripped-down CNN page. Worker is auto-bundled when imported via
// `new Worker(new URL('./trainer.worker.ts', import.meta.url), {type:'module'})`.
export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? './',
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: 'index.html',
        convolutional: 'convolutional/index.html',
      },
    },
  },
  worker: {
    format: 'es',
  },
});
