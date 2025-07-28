// Global type declarations for missing packages

declare module '@babel/core' {
  const babel: any;
  export = babel;
}

declare module '@babel/generator' {
  const generator: any;
  export = generator;
}

declare module '@babel/template' {
  const template: any;
  export = template;
}

declare module '@babel/traverse' {
  const traverse: any;
  export = traverse;
}

declare module 'parse-json' {
  const parseJson: any;
  export = parseJson;
}

declare module 'webxr' {
  const webxr: any;
  export = webxr;
}

// Extend the ImportMeta interface to include env
declare interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_APP_TITLE?: string;
  // Add other environment variables as needed
  [key: string]: any;
}