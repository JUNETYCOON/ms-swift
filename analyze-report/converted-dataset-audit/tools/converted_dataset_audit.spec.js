const {test, expect} = require('playwright/test');
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');

const root = path.resolve(process.env.REPORT_ROOT || '');
if (!process.env.REPORT_ROOT) throw new Error('REPORT_ROOT is required');
const overall = JSON.parse(fs.readFileSync(path.join(root, 'overall-summary.json'), 'utf8'));
const datasetFiles = overall.datasets.map(item => `${item.name}.html`);
const htmlFiles = ['index.html', ...datasetFiles];
const screenshots = path.join(root, 'audit', 'screenshots');
const results = [];

test.describe.configure({mode: 'serial'});
test.beforeAll(() => fs.mkdirSync(screenshots, {recursive: true}));
test.afterAll(() => {
  const failures = results.filter(item => item.errors.length || item.brokenImages.length || item.overflow > 1);
  fs.writeFileSync(
    path.join(root, 'browser-validation.json'),
    `${JSON.stringify({
      status: failures.length ? 'failed' : 'passed',
      browser: 'Microsoft Edge via Playwright',
      source: 'file://',
      html_pages: htmlFiles.length,
      viewport_checks: results.length,
      failures,
      pages: results,
    }, null, 2)}\n`,
    'utf8',
  );
});

for (const viewport of [
  {name: 'desktop', width: 1440, height: 900},
  {name: 'mobile', width: 390, height: 844},
]) {
  test(`${viewport.name} offline report audit`, async ({browser}) => {
    for (const filename of htmlFiles) {
      const page = await browser.newPage({viewport: {width: viewport.width, height: viewport.height}});
      const errors = [];
      page.on('console', message => {
        if (message.type() === 'error') errors.push(`console: ${message.text()}`);
      });
      page.on('pageerror', error => errors.push(`page: ${error.message}`));
      page.on('requestfailed', request => errors.push(`request: ${request.url()} ${request.failure()?.errorText || ''}`));
      await page.goto(pathToFileURL(path.join(root, filename)).href, {waitUntil: 'load', timeout: 60_000});
      await page.evaluate(() => document.querySelectorAll('img').forEach(image => { image.loading = 'eager'; }));
      await page.waitForFunction(
        () => Array.from(document.images).every(image => image.complete),
        null,
        {timeout: 30_000},
      );
      const metrics = await page.evaluate(() => ({
        charset: document.characterSet,
        overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth,
        brokenImages: Array.from(document.images)
          .filter(image => !image.complete || image.naturalWidth === 0)
          .map(image => image.getAttribute('src')),
        sampleCards: document.querySelectorAll('.sample').length,
        blankTextElements: Array.from(document.querySelectorAll('.sample-id'))
          .filter(node => !node.textContent.trim()).length,
      }));
      expect(metrics.charset).toBe('UTF-8');
      expect(metrics.overflow).toBeLessThanOrEqual(1);
      expect(metrics.brokenImages).toEqual([]);
      expect(metrics.blankTextElements).toBe(0);
      expect(errors).toEqual([]);
      if (filename !== 'index.html') expect(metrics.sampleCards).toBe(100);

      if (filename === datasetFiles[0]) {
        const firstId = (await page.locator('.sample-id').first().textContent()).trim();
        await page.locator('#filter-search').fill(firstId);
        await expect(page.locator('.sample:visible')).toHaveCount(1);
        await page.locator('#filter-search').fill('');
        const statusValue = await page.locator('#filter-status option').nth(1).getAttribute('value');
        await page.locator('#filter-status').selectOption(statusValue);
        expect(await page.locator('.sample:visible').count()).toBeGreaterThan(0);
        await page.locator('#filter-status').selectOption('');
        await page.locator('#filter-format').selectOption('pass');
        expect(await page.locator('.sample:visible').count()).toBeGreaterThan(0);
        await page.locator('#filter-format').selectOption('');
      }

      if (filename === 'index.html') {
        await page.screenshot({path: path.join(screenshots, `${viewport.name}-index.png`)});
      }
      if (['ai2d.html', 'robovqa.html', 'pixmo-cap.html'].includes(filename)) {
        await expect(page.locator('.sample:visible')).toHaveCount(100);
        const firstSample = page.locator('.sample:visible').first();
        await page.evaluate(() => { document.documentElement.style.scrollBehavior = 'auto'; });
        await firstSample.evaluate(node => {
          const filters = document.querySelector('.filters');
          const stickyOffset = filters && getComputedStyle(filters).position === 'sticky'
            ? filters.getBoundingClientRect().height + 8
            : 8;
          window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - stickyOffset);
        });
        const box = await firstSample.boundingBox();
        expect(box).not.toBeNull();
        expect(box.y + box.height).toBeGreaterThan(0);
        expect(box.y).toBeLessThan(viewport.height / 2);
        await page.screenshot({path: path.join(screenshots, `${viewport.name}-${filename.replace('.html', '')}-samples.png`)});
      }
      results.push({
        viewport: viewport.name,
        width: viewport.width,
        height: viewport.height,
        file: filename,
        errors,
        brokenImages: metrics.brokenImages,
        overflow: metrics.overflow,
        sampleCards: metrics.sampleCards,
      });
      await page.close();
    }
  });
}
