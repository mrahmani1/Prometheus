import type { StorybookConfig } from '@storybook/react-vite';
import { mergeConfig } from 'vite';
import path from 'path';

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx|mdx)'],
  addons: [
    '@storybook/addon-links',
    '@storybook/addon-essentials',
    '@storybook/addon-onboarding',
    '@storybook/addon-interactions',
  ],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  docs: {
    autodocs: 'tag',
  },
  viteFinal: async (config) => {
    return mergeConfig(config, {
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '../src'),
          '@components': path.resolve(__dirname, '../src/components'),
          '@pages': path.resolve(__dirname, '../src/pages'),
          '@hooks': path.resolve(__dirname, '../src/hooks'),
          '@utils': path.resolve(__dirname, '../src/utils'),
          '@store': path.resolve(__dirname, '../src/store'),
          '@types': path.resolve(__dirname, '../src/types'),
          '@assets': path.resolve(__dirname, '../src/assets'),
          '@geological': path.resolve(__dirname, '../src/geological'),
          '@visualization': path.resolve(__dirname, '../src/visualization'),
        },
      },
    });
  },
};

export default config;