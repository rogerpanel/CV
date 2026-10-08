// Render poster.html to an A0 portrait PDF and a PNG preview.
const { chromium } = require(process.env.PW || 'playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  const page = await browser.newPage({ viewport: { width: 3179, height: 4494 } }); // A0 @ 96dpi
  await page.goto('file://' + path.resolve(__dirname, 'poster.html'));
  await page.waitForLoadState('networkidle');
  const over = await page.evaluate(() => {
    const p = document.querySelector('.poster');
    return { scroll: p.scrollHeight, client: p.clientHeight,
      main: document.querySelector('main').scrollHeight - document.querySelector('main').clientHeight };
  });
  console.log('overflow check', over);
  await page.pdf({ path: path.resolve(__dirname, 'EMNLP 2026_Find-5651.pdf'), width: '841mm', height: '1189mm', printBackground: true });
  const small = await browser.newPage({ viewport: { width: 3179, height: 4494 }, deviceScaleFactor: 0.35 });
  await small.goto('file://' + path.resolve(__dirname, 'poster.html'));
  await small.waitForLoadState('networkidle');
  await small.screenshot({ path: path.resolve(__dirname, 'poster_preview.png') });
  await browser.close();
})();
