import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.kssriharshith.dcsudoku',
  appName: 'DC Sudoku',
  webDir: 'dist',
  bundledWebRuntime: false,
  server: {
    androidScheme: 'http'
  }
};

export default config;
