import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.kssriharshith.dcsudoku',
  appName: 'DC Sudoku',
  webDir: 'dist',
  bundledWebRuntime: false,
  server: {
    androidScheme: 'http',
    // Hosts the OTA loader may fetch remote app modules from. The modules
    // execute in the local origin, so app saves in localStorage carry over.
    allowNavigation: [
      'reboot2004.github.io',
      'cdn.jsdelivr.net',
      'raw.githubusercontent.com'
    ]
  }
};

export default config;
