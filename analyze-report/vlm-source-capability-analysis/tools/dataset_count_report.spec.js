const {test, expect} = require('@playwright/test');
const path = require('path');
const {pathToFileURL} = require('url');

const reportPath = process.env.REPORT_PATH;
if (!reportPath) throw new Error('REPORT_PATH is required');
const reportUrl = pathToFileURL(path.resolve(reportPath)).href;
const screenshotDir = path.resolve(path.dirname(reportPath), 'audit', 'screenshots');

for (const viewport of [
  {name: 'desktop', width: 1440, height: 900},
  {name: 'mobile', width: 390, height: 844},
]) {
  test(`${viewport.name} report behavior`, async ({page}) => {
    const errors = [];
    page.on('console', message => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('pageerror', error => errors.push(error.message));
    await page.setViewportSize(viewport);
    await page.goto(reportUrl, {waitUntil: 'load'});
    await expect(page.locator('h1')).toContainText('数据集样本量汇总');
    await expect(page.locator('tbody tr')).toHaveCount(18);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBeTruthy();

    await page.locator('#status').selectOption('missing');
    await expect(page.locator('#shown')).toHaveText('17');
    await page.locator('#status').selectOption('unconfigured');
    await expect(page.locator('#shown')).toHaveText('1');
    await page.locator('#status').selectOption('');
    await page.locator('#search').fill('PixMo');
    await expect(page.locator('#shown')).toHaveText('2');
    await page.locator('#search').fill('');
    await expect(page.locator('#shown')).toHaveText('18');

    await page.screenshot({path: path.join(screenshotDir, `${viewport.name}.png`), fullPage: true});
    expect(errors).toEqual([]);
  });
}
