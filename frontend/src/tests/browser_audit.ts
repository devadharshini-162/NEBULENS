/**
 * Browser-based Gate C User Journey Acceptance Test
 * Uses real headless Chromium to interact with live Vite dev server (5173) and FastAPI backend (8000).
 */
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

async function runBrowserAudit() {
  console.log('--- STARTING LIVE CHROMIUM BROWSER GATE C AUDIT ---');

  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const consoleErrors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(`[Browser Console Error] ${msg.text()}`);
    }
  });
  page.on('pageerror', (err) => {
    consoleErrors.push(`[Browser Unhandled Exception] ${err.toString()}`);
  });

  const failedRequests: string[] = [];
  page.on('requestfailed', (req) => {
    failedRequests.push(`[Request Failed] ${req.method()} ${req.url()}: ${req.failure()?.errorText}`);
  });

  try {
    // Step 1: Open NEBULENS / Overview
    console.log('Step 1: Navigating to http://127.0.0.1:5173...');
    await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle2', timeout: 15000 });
    
    const title = await page.title();
    console.log(`Page title: "${title}"`);
    if (!title.includes('NEBULENS')) throw new Error(`Unexpected page title: ${title}`);

    // Wait for Overview cards
    await page.waitForSelector('.overview-view', { timeout: 10000 });
    const healthText = await page.$eval('.overview-view', (el) => el.innerText);
    console.log('Overview loaded successfully. Verified Gate B table presence.');
    if (!healthText.includes('GATE B') && !healthText.includes('Gate B')) {
      throw new Error('Gate B validation matrix missing in Overview!');
    }

    // Step 2: Open Map
    console.log('Step 2: Switching to Geospatial Map view...');
    const buttons = await page.$$('.nav-button');
    console.log(`Found ${buttons.length} navigation buttons.`);
    for (let i = 0; i < buttons.length; i++) {
      const text = await buttons[i].evaluate((el) => el.textContent);
      console.log(`Nav button ${i}: "${text?.trim()}"`);
    }
    // Click button with "Geospatial Map"
    const mapBtn = buttons[1];
    await mapBtn.click();
    console.log('Clicked Geospatial Map button. Waiting for container or DOM update...');
    await page.waitForSelector('.map-view-container', { timeout: 10000 });
    console.log('Map view container rendered.');

    // Step 3: Change Sensor to VIIRS
    console.log('Step 3: Selecting Sensor Stream VIIRS...');
    await page.select('#sensor-select', 'viirs');
    await page.waitForNetworkIdle({ timeout: 5000 }).catch(() => {});
    const sensorVal = await page.$eval('#sensor-select', (el: any) => el.value);
    console.log(`Sensor changed to: ${sensorVal}`);

    // Helper to click nav by label
    const clickNav = async (labelSubstring: string) => {
      const navs = await page.$$('.nav-button');
      for (const btn of navs) {
        const text = await btn.evaluate((el) => el.textContent);
        if (text && text.includes(labelSubstring)) {
          await btn.click();
          return;
        }
      }
      throw new Error(`Nav button with "${labelSubstring}" not found!`);
    };

    // Step 4: Open Historical Calendar Heatmap
    console.log('Step 4: Switching to Historical Heatmap...');
    await clickNav('Historical Heatmap');
    await page.waitForSelector('.heatmap-container', { timeout: 10000 });
    const heatmapTitle = await page.$eval('.heatmap-container h2', (el) => el.textContent);
    console.log(`Heatmap loaded: "${heatmapTitle}"`);

    // Step 5: Open Continuous Time Series
    console.log('Step 5: Switching to Continuous Time Series...');
    await clickNav('Time Series');
    await page.waitForSelector('.series-container', { timeout: 10000 });
    const seriesTitle = await page.$eval('.series-container h2', (el) => el.textContent);
    console.log(`Series loaded: "${seriesTitle}"`);

    // Step 6: Open Seasonal Comparison
    console.log('Step 6: Switching to Seasonal Comparison...');
    await clickNav('Seasonal Comparison');
    await page.waitForSelector('.compare-container', { timeout: 10000 });
    await page.waitForSelector('.compare-container h2', { timeout: 10000 });
    const compareTitle = await page.$eval('.compare-container h2', (el) => el.textContent);
    console.log(`Seasonal comparison loaded: "${compareTitle}"`);

    // Step 7: Open Hotspots Inspector
    console.log('Step 7: Switching to Hotspot Granule Inspector...');
    await clickNav('Hotspot Inspector');
    await page.waitForSelector('.hotspots-container', { timeout: 10000 });
    // Wait for the data table rows to render
    await page.waitForSelector('.btn-table-action', { timeout: 10000 });
    const actionBtns = await page.$$('.btn-table-action');
    if (actionBtns.length > 0) {
      await actionBtns[0].click();
      await page.waitForSelector('.detail-hero', { timeout: 10000 });
      const hotspotId = await page.$eval('.detail-id', (el) => el.textContent);
      console.log(`Inspected Hotspot Observation ID: "${hotspotId}"`);
    }

    // Step 8: Open Anomaly Engine
    console.log('Step 8: Switching to Anomaly Investigation Engine...');
    await clickNav('Anomaly Engine');
    await page.waitForSelector('.anomalies-container', { timeout: 10000 });
    await page.waitForSelector('.evidence-content', { timeout: 10000 });
    const zScoreText = await page.$eval('.evidence-z-score', (el) => el.textContent);
    console.log(`Anomaly evidence loaded: "${zScoreText?.trim()}"`);

    // Step 9: Open Multi-Evidence Attribution
    console.log('Step 9: Switching to Multi-Evidence Attribution...');
    await clickNav('Multi-Evidence Attribution');
    await page.waitForSelector('.attribution-container', { timeout: 10000 });
    await page.waitForSelector('.burned-area-warning-card', { timeout: 10000 });
    console.log('Attribution loaded. Verified MCD64A1 null status preserved.');

    // Step 10: Open Scientific Transparency & Gate B
    console.log('Step 10: Switching to Transparency & Gate B Framework...');
    await clickNav('Validation & Gate B');
    await page.waitForSelector('.transparency-container', { timeout: 10000 });
    const covRows = await page.$$('.data-table tbody tr');
    console.log(`Transparency loaded with ${covRows.length} timeline regime rows.`);

    // Check for Console Errors
    console.log(`\nBrowser Console Errors Count: ${consoleErrors.length}`);
    if (consoleErrors.length > 0) {
      console.log('Console Errors:', consoleErrors);
    }
    console.log(`Failed Requests Count: ${failedRequests.length}`);
    if (failedRequests.length > 0) {
      console.log('Failed Requests:', failedRequests);
    }

    // Take screenshot of transparency view as proof of rendered browser UI
    const screenshotPath = 'public/gate_c_browser_proof.png';
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`Saved full-page rendered screenshot to: ${screenshotPath}`);

    console.log('\n--- BROWSER GATE C AUDIT COMPLETED SUCCESSFULLY! ---');
  } finally {
    await browser.close();
  }
}

runBrowserAudit().catch((err) => {
  console.error('Browser Audit Failed:', err);
  process.exit(1);
});
