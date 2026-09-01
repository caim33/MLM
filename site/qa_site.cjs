const { chromium } = require("playwright");
const { pathToFileURL } = require("url");
const path = require("path");

(async () => {
  const root = __dirname;
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("requestfailed", (request) => failedRequests.push(`${request.url()} :: ${request.failure()?.errorText}`));

  await page.goto(pathToFileURL(path.join(root, "index.html")).href, { waitUntil: "load" });
  await page.waitForSelector(".paper-card");
  await page.locator(".paper-card img").evaluateAll((images) => images.forEach((image) => { image.loading = "eager"; }));
  await page.waitForFunction(() =>
    [...document.querySelectorAll(".paper-card img")].every((image) => image.complete && image.naturalWidth > 0)
  );
  const initialCards = await page.locator(".paper-card").count();
  const brokenImages = await page.locator(".paper-card img").evaluateAll((images) =>
    images.filter((image) => !image.complete || image.naturalWidth === 0).map((image) => image.src)
  );

  await page.locator("#paperSearch").fill("GRPO");
  const searchCards = await page.locator(".paper-card").count();
  await page.locator("#paperSearch").fill("");
  await page.locator('[data-filter="direct"]').click();
  const directCards = await page.locator(".paper-card").count();
  await page.locator('[data-filter="all"]').click();
  await page.locator(".paper-figure").first().click();
  const dialogOpened = await page.locator("#figureDialog").evaluate((dialog) => dialog.open);
  await page.locator("#dialogClose").click();

  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(100);
  await page.screenshot({ path: path.join(root, "site_preview_top.png"), fullPage: false });
  await page.screenshot({ path: path.join(root, "site_preview.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload({ waitUntil: "load" });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(100);
  await page.screenshot({ path: path.join(root, "site_preview_mobile.png"), fullPage: false });

  console.log(JSON.stringify({ initialCards, searchCards, directCards, brokenImages, dialogOpened, consoleErrors, failedRequests }, null, 2));
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
