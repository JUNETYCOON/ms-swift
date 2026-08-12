#!/usr/bin/env node

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const outputDir = path.join(root, "browser-check");
const browserPath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const datasets = [
  "COCO",
  "VQAv2",
  "VisualGenome",
  "GQA",
  "TextVQA",
  "ChartQA",
  "AI2D",
  "LLaVA-Instruct",
  "VLM-R1",
  "Robo2VLM",
  "RoboVQA",
  "SpatialVLM",
  "PixMo-Cap",
  "PixMo-Points",
  "Molmo2-VideoCapQA",
  "Molmo2-VideoPoint",
  "Molmo2-VideoSubtitleQA",
  "Molmo2-VideoTrack",
];
const rootHtmlFiles = ["index.html", ...datasets.map((name) => `${name}.html`)];
const htmlFiles = rootHtmlFiles;
const viewports = [
  { width: 1440, height: 1000, name: "desktop" },
  { width: 390, height: 844, name: "mobile" },
];
const screenshots = new Set([
  "desktop:index.html",
  "desktop:VisualGenome.html",
  "desktop:Molmo2-VideoCapQA.html",
  "mobile:index.html",
  "mobile:RoboVQA.html",
  "mobile:Molmo2-VideoTrack.html",
]);
const indexScreenshotTargets = [
  { name: "inventory", selector: "#inventory" },
  { name: "tasks", selector: "#tasks" },
  { name: "distribution", selector: "#distribution" },
  { name: "task-distribution", heading: "任务分布堆叠图" },
  { name: "annotation-density", heading: "每媒体标注数量分布" },
  { name: "text-length", heading: "输入输出长度分布" },
  { name: "embodied", selector: "#embodied" },
  { name: "recipes", selector: "#recipes" },
  { name: "references", heading: "参考文献与方案映射" },
];

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function fetchJson(url, options = {}, timeout = 20_000) {
  const deadline = Date.now() + timeout;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, options);
      if (response.ok) return response.json();
      lastError = new Error(`${response.status} ${response.statusText}`);
    } catch (error) {
      lastError = error;
    }
    await delay(100);
  }
  throw lastError || new Error(`Timed out fetching ${url}`);
}

class CdpClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.nextId = 1;
    this.pending = new Map();
    this.waiters = new Map();
    this.listeners = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => this.handleMessage(event));
  }

  handleMessage(event) {
    const message = JSON.parse(event.data);
    if (message.id) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message));
      else pending.resolve(message.result);
      return;
    }
    for (const listener of this.listeners.get(message.method) || []) listener(message.params || {});
    const waiter = (this.waiters.get(message.method) || []).shift();
    if (waiter) {
      clearTimeout(waiter.timer);
      waiter.resolve(message.params || {});
    }
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) || [];
    listeners.push(listener);
    this.listeners.set(method, listeners);
  }

  waitFor(method, timeout = 60_000) {
    return new Promise((resolve, reject) => {
      const waiters = this.waiters.get(method) || [];
      const waiter = {
        resolve,
        reject,
        timer: setTimeout(() => {
          const index = waiters.indexOf(waiter);
          if (index >= 0) waiters.splice(index, 1);
          reject(new Error(`Timed out waiting for ${method}`));
        }, timeout),
      };
      waiters.push(waiter);
      this.waiters.set(method, waiters);
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.socket.close();
  }
}

function consoleText(params) {
  return (params.args || [])
    .map((argument) => argument.value ?? argument.description ?? argument.type)
    .join(" ");
}

async function audit() {
  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });
  const profile = await mkdtemp(path.join(os.tmpdir(), "vlm-source-browser-"));
  const port = await freePort();
  const browser = spawn(
    browserPath,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      "--allow-file-access-from-files",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: "ignore", windowsHide: true },
  );

  let cdp;
  try {
    await fetchJson(`http://127.0.0.1:${port}/json/version`);
    const target = await fetchJson(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" });
    cdp = new CdpClient(target.webSocketDebuggerUrl);
    await cdp.connect();
    await Promise.all([
      cdp.send("Page.enable"),
      cdp.send("Runtime.enable"),
      cdp.send("Network.enable"),
      cdp.send("Log.enable"),
    ]);

    let current = null;
    cdp.on("Runtime.consoleAPICalled", (params) => {
      if (current && params.type === "error") current.consoleErrors.push(consoleText(params));
    });
    cdp.on("Runtime.exceptionThrown", (params) => {
      if (current) current.pageErrors.push(params.exceptionDetails?.text || "Runtime exception");
    });
    cdp.on("Log.entryAdded", (params) => {
      if (current && params.entry?.level === "error") current.consoleErrors.push(params.entry.text);
    });
    cdp.on("Network.loadingFailed", (params) => {
      if (current && !params.canceled) current.networkErrors.push(params.errorText || "Network loading failed");
    });

    const pages = [];
    const screenshotFiles = [];
    for (const viewport of viewports) {
      await cdp.send("Emulation.setDeviceMetricsOverride", {
        width: viewport.width,
        height: viewport.height,
        deviceScaleFactor: 1,
        mobile: false,
        screenWidth: viewport.width,
        screenHeight: viewport.height,
      });
      for (const htmlFile of htmlFiles) {
        current = { consoleErrors: [], pageErrors: [], networkErrors: [] };
        const filePath = path.join(root, htmlFile);
        const loaded = cdp.waitFor("Page.loadEventFired");
        await cdp.send("Page.navigate", { url: pathToFileURL(filePath).href });
        await loaded;
        const evaluated = await cdp.send("Runtime.evaluate", {
          awaitPromise: true,
          returnByValue: true,
          expression: `
            (async () => {
              const allImages = Array.from(document.images);
              const allVideos = Array.from(document.querySelectorAll("video"));
              const sampleRecords = Array.from(document.querySelectorAll("details.sample-record"));
              const probeLimit = sampleRecords.length ? 3 : Number.POSITIVE_INFINITY;
              const images = allImages.slice(0, probeLimit);
              const videos = allVideos.slice(0, probeLimit);
              [...images, ...videos].forEach((media) => {
                const record = media.closest("details.sample-record");
                if (record) record.open = true;
              });
              images.forEach((img) => { img.loading = "eager"; });
              videos.forEach((video) => {
                video.preload = "metadata";
                if (video.readyState < 1 && !video.error) video.load();
              });
              await Promise.all(images.map((img) => {
                if (img.complete) return Promise.resolve();
                return new Promise((resolve) => {
                  img.addEventListener("load", resolve, { once: true });
                  img.addEventListener("error", resolve, { once: true });
                  setTimeout(resolve, 10000);
                });
              }));
              await Promise.all(videos.map((video) => {
                if (video.readyState >= 1 || video.error) return Promise.resolve();
                return new Promise((resolve) => {
                  video.addEventListener("loadedmetadata", resolve, { once: true });
                  video.addEventListener("error", resolve, { once: true });
                  setTimeout(resolve, 15000);
                });
              }));
              await Promise.all(videos.map((video) => {
                if (
                  video.error ||
                  video.readyState < 1 ||
                  !Number.isFinite(video.duration) ||
                  video.duration <= 0
                ) return Promise.resolve();
                return new Promise((resolve) => {
                  const finish = () => resolve();
                  video.addEventListener("seeked", finish, { once: true });
                  video.addEventListener("loadeddata", finish, { once: true });
                  video.addEventListener("error", finish, { once: true });
                  video.currentTime = Math.min(0.1, video.duration / 2);
                  setTimeout(resolve, 5000);
                });
              }));
              const scrollWidth = Math.max(
                document.documentElement.scrollWidth,
                document.body?.scrollWidth || 0,
              );
              const brokenImages = images
                .filter((img) => !img.complete || img.naturalWidth === 0)
                .map((img) => img.getAttribute("src"));
              const videoStates = videos.map((video) => ({
                src: video.currentSrc || video.getAttribute("src"),
                readyState: video.readyState,
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                duration: Number.isFinite(video.duration) ? video.duration : null,
                error: video.error ? { code: video.error.code, message: video.error.message } : null,
              }));
              const brokenVideos = videoStates.filter((video) =>
                video.error || video.readyState < 1 || video.videoWidth <= 0 || video.videoHeight <= 0
              );
              const detailRoot = document.querySelector("main[data-sample-id]");
              const citationLinks = Array.from(document.querySelectorAll('.citations a[href^="#ref-"]'));
              const inventoryDatasetLinks = Array.from(
                document.querySelectorAll("#inventory tbody tr > td:first-child > a"),
              );
              return {
                title: document.title,
                charset: document.characterSet,
                overflow: scrollWidth - document.documentElement.clientWidth,
                brokenImages,
                imageCount: allImages.length,
                imageProbeCount: images.length,
                decodedImageCount: images.filter((img) => img.complete && img.naturalWidth > 0).length,
                videoCount: allVideos.length,
                videoProbeCount: videos.length,
                playableVideoCount: videoStates.length - brokenVideos.length,
                videoStates,
                brokenVideos,
                isDetailPage: Boolean(detailRoot),
                dataset: document.querySelector("main")?.dataset.dataset || "",
                sampleRecordCount: sampleRecords.length,
                sampleId: detailRoot?.dataset.sampleId || "",
                qaCount: document.querySelectorAll(".qa-item").length,
                taxonomyCount: Array.from(document.querySelectorAll(".qa-item .qa-fields dt"))
                  .filter((node) => node.textContent.trim() === "统一任务分类").length,
                rawRecordCount: document.querySelectorAll("#raw .raw-json").length,
                schemaLinkCount: document.querySelectorAll('a[href*="source-schemas/"]').length,
                referenceCount: document.querySelectorAll('.reference-list li[id^="ref-"]').length,
                citationLinkCount: citationLinks.length,
                brokenCitationTargets: citationLinks
                  .filter((link) => !document.querySelector(link.getAttribute("href")))
                  .map((link) => link.getAttribute("href")),
                inventoryDatasetLinkCount: inventoryDatasetLinks.length,
              };
            })()
          `,
        });
        if (!evaluated.result || !("value" in evaluated.result)) {
          throw new Error(
            `Page audit evaluation failed for ${viewport.name}:${htmlFile}: ${JSON.stringify(evaluated.exceptionDetails || evaluated)}`,
          );
        }
        const metrics = evaluated.result.value;
        await delay(100);
        const pageResult = {
          viewport: viewport.name,
          width: viewport.width,
          file: htmlFile,
          ...metrics,
          ...current,
        };
        pages.push(pageResult);

        if (screenshots.has(`${viewport.name}:${htmlFile}`)) {
          const capture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
          const stem = htmlFile.replace(/\.html$/, "").replace(/[\\/]+/g, "-");
          const topName = `${viewport.name}-${stem}.png`;
          await writeFile(path.join(outputDir, topName), Buffer.from(capture.data, "base64"));
          screenshotFiles.push(topName);
          if (htmlFile === "index.html") {
            for (const targetSpec of indexScreenshotTargets) {
              const targetExpression = targetSpec.selector
                ? `document.querySelector(${JSON.stringify(targetSpec.selector)})`
                : `Array.from(document.querySelectorAll("h3")).find((node) => node.textContent.trim() === ${JSON.stringify(targetSpec.heading)})`;
              await cdp.send("Runtime.evaluate", {
                expression: `(() => {
                  document.documentElement.style.scrollBehavior = "auto";
                  const target = ${targetExpression};
                  if (target) target.scrollIntoView({ block: "start" });
                  return Boolean(target);
                })()`,
              });
              await delay(100);
              const sectionCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
              const sectionName = `${viewport.name}-index-${targetSpec.name}.png`;
              await writeFile(
                path.join(outputDir, sectionName),
                Buffer.from(sectionCapture.data, "base64"),
              );
              screenshotFiles.push(sectionName);
            }
          } else if (metrics.isDetailPage) {
            for (const targetSpec of [
              { name: "media", selector: "#media" },
              { name: "qa", selector: "#qa" },
            ]) {
              await cdp.send("Runtime.evaluate", {
                expression: `(() => {
                  document.documentElement.style.scrollBehavior = "auto";
                  const target = document.querySelector(${JSON.stringify(targetSpec.selector)});
                  if (target) target.scrollIntoView({ block: "start" });
                  return Boolean(target);
                })()`,
              });
              await delay(200);
              const detailCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
              const detailName = `${viewport.name}-${stem}-${targetSpec.name}.png`;
              await writeFile(
                path.join(outputDir, detailName),
                Buffer.from(detailCapture.data, "base64"),
              );
              screenshotFiles.push(detailName);
            }
          } else {
            const selector = metrics.sampleRecordCount
              ? "#samples"
              : (metrics.imageCount ? ".gallery" : ".record-grid");
            await cdp.send("Runtime.evaluate", {
              expression: `(() => {
                document.documentElement.style.scrollBehavior = "auto";
                const target = document.querySelector(${JSON.stringify(selector)});
                if (target) target.scrollIntoView({ block: "start" });
                return Boolean(target);
              })()`,
            });
            await delay(150);
            const evidenceCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
            const evidenceName = `${viewport.name}-${stem}-evidence.png`;
            await writeFile(
              path.join(outputDir, evidenceName),
              Buffer.from(evidenceCapture.data, "base64"),
            );
            screenshotFiles.push(evidenceName);
          }
        }
      }
    }

    const failures = pages.filter(
      (page) =>
        page.overflow > 1 ||
        page.brokenImages.length ||
        page.brokenVideos.length ||
        (page.file === "index.html" && (
          page.inventoryDatasetLinkCount !== 18 ||
          page.referenceCount !== 10 ||
          page.citationLinkCount < 10 ||
          page.brokenCitationTargets.length
        )) ||
        (page.dataset && !page.isDetailPage && (
          page.sampleRecordCount !== 200 ||
          page.qaCount < 200 ||
          page.taxonomyCount !== page.qaCount
        )) ||
        (page.isDetailPage && (
          !page.sampleId ||
          page.qaCount < 1 ||
          page.taxonomyCount !== page.qaCount ||
          page.rawRecordCount < 1 ||
          page.schemaLinkCount < 1
        )) ||
        page.consoleErrors.length ||
        page.pageErrors.length ||
        page.networkErrors.length,
    );
    const result = {
      status: failures.length ? "failed" : "passed",
      browser: "Microsoft Edge (CDP)",
      source: "file://",
      htmlPages: htmlFiles.length,
      viewportChecks: pages.length,
      screenshots: screenshotFiles,
      failures,
      pages,
    };
    await writeFile(path.join(outputDir, "browser-audit.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
    console.log(JSON.stringify({
      status: result.status,
      htmlPages: result.htmlPages,
      viewportChecks: result.viewportChecks,
      failureCount: failures.length,
      decodedImages: pages.reduce((total, page) => total + page.decodedImageCount, 0),
      playableVideos: pages.reduce((total, page) => total + page.playableVideoCount, 0),
    }, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally {
    if (cdp) {
      try {
        await cdp.send("Browser.close");
      } catch {
        browser.kill();
      }
      cdp.close();
    } else {
      browser.kill();
    }
    const temporaryRoot = `${path.resolve(os.tmpdir())}${path.sep}`;
    if (path.resolve(profile).startsWith(temporaryRoot)) {
      await delay(300);
      await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    }
  }
}

await audit();
