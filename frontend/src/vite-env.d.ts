/// <reference types="vite/client" />

/**
 * Typed environment variables.
 *
 * Declaring these gives `import.meta.env.VITE_API_URL` a real type instead of
 * `any`, so a typo in the variable name is a compile error rather than an
 * undefined at runtime.
 */
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
