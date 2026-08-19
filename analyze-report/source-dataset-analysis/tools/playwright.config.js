const path = require('path');

module.exports = {
  testDir: __dirname,
  outputDir: path.resolve(__dirname, '..', 'current-dataset-counts', 'audit', 'test-results'),
  use: {
    channel: 'msedge',
    headless: true,
  },
};
