const path = require('path');

const reportRoot = process.env.REPORT_ROOT;
if (!reportRoot) throw new Error('REPORT_ROOT is required');

module.exports = {
  testDir: __dirname,
  testMatch: 'converted_dataset_audit.spec.js',
  outputDir: path.join(path.resolve(reportRoot), 'audit', 'test-results'),
  workers: 1,
  reporter: [['line']],
  use: {
    channel: 'msedge',
    headless: true,
  },
};
