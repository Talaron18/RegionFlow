/* Run the UI's JavaScript against a DOM stub:
 *     node page_harness.js short_term/app.py     (it extracts the <script>)
 *     node page_harness.js page.js               (a bare JS file)
 *
 * WHY THIS EXISTS. Everything else in this project is testable on a CPU-only
 * box, but the browser page was not, and it showed: a stylesheet naming ids
 * that no longer existed laid three canvases out side by side, a function that
 * was CALLED but never DEFINED turned every finished run into a caught
 * "ReferenceError", and a stray brace stopped the whole script from parsing --
 * which silently disables every upload box and button on the page.
 *
 * None of those are visible from the server side, and a syntax error is
 * invisible from the Python side too, so the page is now parsed and EXECUTED
 * here. `python short_term/app.py --self-test` calls this when node is
 * available and skips it with a printed note when it is not.
 *
 * What it verifies: the script parses; loading it does not throw; the functions
 * the page calls actually exist; the three-state touch-up layer counts and
 * encodes correctly; and render() produces the result panel for a real
 * response, including the null that a sanitised NaN becomes.
 */
'use strict';
const fs = require('fs');

const path = process.argv[2];
if (!path) {
  console.error('usage: node page_harness.js <app.py|page.js>');
  process.exit(2);
}

let fails = 0;
function check(ok, msg, extra) {
  if (ok) { console.log('  ok   ' + msg); return; }
  fails++;
  console.error('  FAIL ' + msg + (extra === undefined ? '' : '\n       ' + extra));
}

function ctx2d(w, h) {
  const px = () => new Uint8ClampedArray(w * h * 4);
  return {
    fillStyle: '', strokeStyle: '', lineWidth: 1, lineCap: '', globalAlpha: 1,
    globalCompositeOperation: 'source-over', imageSmoothingEnabled: true,
    fillRect() {}, clearRect() {}, strokeRect() {}, drawImage() {},
    setLineDash() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    getImageData: (x, y, a, b) => ({ data: px(), width: a || w, height: b || h }),
    createImageData: (a, b) => ({ data: px(), width: a, height: b }),
    putImageData() {},
  };
}

const els = {};
function el(id) {
  if (!els[id]) {
    const w = 512, h = 512;
    els[id] = {
      id: id, value: '20', checked: false, textContent: '', innerHTML: '',
      className: '', disabled: false, files: [], width: w, height: h,
      style: {}, title: '',
      getContext: () => ctx2d(w, h),
      addEventListener() {}, removeEventListener() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: w, height: h }),
      toDataURL: () => 'data:image/png;base64,QUJD',
    };
  }
  return els[id];
}

global.document = {
  getElementById: el,
  createElement: (tag) => { const o = el('__' + tag + fails + Math.random()); o.tagName = tag; return o; },
};
global.window = { addEventListener() {}, location: { href: 'http://x/' } };
// fires onload synchronously, so the upload handlers can be driven for real
global.Image = class {
  constructor() { this.onload = null; this.naturalWidth = 1600; this.naturalHeight = 1200; }
  set src(v) { if (this.onload) this.onload(); }
};
global.URL = { createObjectURL: () => 'blob:x' };
// captures the POST body: the browser->server contract is checkable on a CPU
let posted = null;
global.fetch = (url, opts) => {
  posted = { url: url, body: JSON.parse(opts.body) };
  return Promise.resolve({ json: async () => ({ ok: false, error: 'harness' }) });
};

let src;
try {
  src = fs.readFileSync(path, 'utf8');
} catch (e) {
  console.error('cannot read ' + path + ': ' + e.message);
  process.exit(2);
}

// app.py hands us its whole PAGE string: take the <script> body out of it, the
// same way the browser does, so there is no second copy of the JS to rot.
if (/\.py$/i.test(path)) {
  const open = src.indexOf('<script>');
  const close = src.indexOf('</script>');
  if (open < 0 || close < 0) {
    console.error('  FAIL no <script> block found in ' + path);
    process.exit(1);
  }
  src = src.slice(open + '<script>'.length, close);
  console.log('  ok   extracted ' + src.length + ' bytes of JS from ' + path);
}

// 1. parse AND execute. A syntax error here is the whole page dead.
try {
  // indirect eval keeps the page's declarations out of this scope, so the
  // "was it defined at all" checks below stay honest
  (0, eval)(src);
} catch (e) {
  console.error('  FAIL the page script does not run: ' + e);
  process.exit(1);
}
check(true, 'the page script parses and runs');

// 2. the functions the page calls must EXIST. render() was called by the Run
//    handler and never defined: every finished run ended in a caught
//    ReferenceError reported as a bare "error", with no picture and no table.
const needed = ['render', 'setMode', 'refinePng', 'refineStats', 'blankRefine',
                'drawMaskPreview', 'drawPlacement', 'patchMaskFrac'];
for (const n of needed) {
  check((0, eval)('typeof ' + n) === 'function', n + '() is defined');
}
// the initial mode must have been APPLIED, not just defaulted: with mode=place
// but the stylesheet's pointer-events:none still on the patch canvas, dragging
// the patch silently did nothing until a mode button was clicked
check(els['patchc'].style.pointerEvents === 'auto',
      'the patch canvas is interactive on load (setMode was called)',
      String(els['patchc'].style.pointerEvents));
check(els['refc'].style.pointerEvents === 'none',
      'the touch-up layer is not interactive until its mode is chosen',
      String(els['refc'].style.pointerEvents));

// 3. the frame: W/H must reach the canvases, and the source crop must follow
//    the FRAME's aspect. A 4:3 photo in a 4:3 frame must be used WHOLE -- the
//    old code cropped to a square unconditionally and threw away the sides.
const needed2 = ['layoutFrame', 'setFrame', 'cropBox', 'settingsView'];
for (const n of needed2) {
  check((0, eval)('typeof ' + n) === 'function', n + '() is defined');
}
(0, eval)('layoutFrame')(512, 384);
for (const id of ['srcc', 'maskprev', 'patchc', 'refc']) {
  check(els[id].width === 512 && els[id].height === 384,
        id + ' canvas is 512x384 after layoutFrame(512,384)',
        els[id].width + 'x' + els[id].height);
}
check(els['stack'].style.height === '288px',
      'the stack box keeps the 4:3 aspect (384x288)',
      String(els['stack'].style.height));

// cropBox on a 4:3 photo and on a 16:9 one
const cb = (0, eval)('cropBox');
let box = cb({ w: 1600, h: 1200 });          // exactly 4:3
check(box.sw === 1600 && box.sh === 1200,
      'a 4:3 photo is used WHOLE in a 4:3 frame, not cropped', JSON.stringify(box));
box = cb({ w: 1920, h: 1080 });              // 16:9 into 4:3 -> crop the sides
check(Math.abs(box.sw / box.sh - 4 / 3) < 1e-9 && box.sw < 1920,
      'a 16:9 photo is centre-cropped to 4:3', JSON.stringify(box));
box = cb({ w: 900, h: 1600 });               // portrait into 4:3 -> crop top/bottom
check(Math.abs(box.sw / box.sh - 4 / 3) < 1e-9 && box.sh < 1600,
      'a portrait photo is centre-cropped to 4:3', JSON.stringify(box));

// and the square frame must still behave like it used to
(0, eval)('layoutFrame')(512, 512);
box = cb({ w: 1600, h: 1200 });
check(box.sw === 1200 && box.sh === 1200,
      'a 4:3 photo in a 1:1 frame is still cropped to a centred square',
      JSON.stringify(box));
check(els['stack'].style.height === '384px', 'a 1:1 frame is a square box',
      String(els['stack'].style.height));

// 4. DRIVE THE REAL UPLOAD HANDLERS. `let`-bound page state (rect, srcImg,
//    patchImg) is not reachable from another eval, so the only honest way to
//    inspect the placement is to drive the handlers the browser would.
els['frame'].value = '4:3';                 // back from the 1:1 check above
(0, eval)('setFrame')();
// a marked object on the patch: without a marking the page correctly REFUSES
// to run, so a run test must first mark something
els['pmc'].getContext = () => {
  const c = ctx2d(320, 320);
  const d = new Uint8ClampedArray(320 * 320 * 4);
  for (let y = 100; y < 160; y++) {
    for (let x = 100; x < 160; x++) {
      const i = (y * 320 + x) * 4;
      d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = 255;
    }
  }
  c.getImageData = () => ({ data: d, width: 320, height: 320 });
  return c;
};
els['pscale'].value = '30';
el('mock').checked = true;                  // exercise the no-GPU path
els['srcfile'].onchange({ target: { files: [{ name: 'src.png' }] } });
check(els['srcinfo'].textContent.includes('-> 512x384'),
      'uploading a photo draws it into the 4:3 frame',
      els['srcinfo'].textContent);
check(els['srcinfo'].textContent.includes('centre-cropped to 1.33'),
      'and reports the crop ratio', els['srcinfo'].textContent);
els['patchfile'].onchange({ target: { files: [{ name: 'p.png' }] } });
const info = els['patchinfo'].textContent;
// patch 1600x1200 at pscale 30% of a 512-wide frame -> placement box 153x115
// centred at (179,134), and fit=contain keeps that exact size for a 4:3 patch
check(/at 179,134/.test(info),
      'the patch placement box is centred in the 4:3 frame', info);

// THE PANEL MUST BE A SCALE OF THE PATCH, NOT A SQUARE THE PATCH SITS INSIDE.
// It used to be a fixed 320x320 with the patch letterboxed into it, so the mask
// was painted in one coordinate system and the content placed in another:
// measured drift 5 px at aspect 1.05, 30 px at 4:3, and 0 px only when square.
// Every earlier test used a square patch, which is why this survived, so the
// aspect is checked here on a deliberately NON-square one (1600x1200).
check(els['patc'].width === 320 && els['patc'].height === 240,
      'the panel takes the PATCH aspect (4:3 -> 320x240), so there is no letterbox',
      els['patc'].width + 'x' + els['patc'].height);
check(els['pmc'].width === 320 && els['pmc'].height === 240,
      'the marking canvas matches the panel exactly',
      els['pmc'].width + 'x' + els['pmc'].height);
const ps = (0, eval)('panelSize')({ w: 1600, h: 1200 });
check(Math.abs(ps.w / ps.h - 1600 / 1200) < 1e-9,
      'panelSize keeps the patch aspect');
// THE DRIFT CONDITION: if the dst box has the same aspect as the panel, then
// mapping the panel into it is a pure scale, so the mask and the content land
// on the same pixels. Different aspects = drift.
const dbox = (0, eval)('dstRect')();
check(Math.abs(dbox.w / dbox.h - els['patc'].width / els['patc'].height) < 0.02,
      'the placement box has the panel aspect, so panel->dst cannot drift',
      dbox.w + 'x' + dbox.h + ' vs panel ' + els['patc'].width + 'x'
      + els['patc'].height);
// and a square patch must still give a square panel (the old behaviour, kept)
(0, eval)('layoutPatch')();
check(els['patc'].width === 320 && els['patc'].height === 240,
      'layoutPatch is idempotent for the loaded patch');

// 5. the POST the page would make
(async () => {
  await els['run'].onclick();
  check(posted !== null, 'Run issued a POST');
  if (posted) {
    check(posted.url === 'api/run', 'the POST url is relative (proxy-safe)',
          posted.url);
    const b = posted.body;
    check(typeof b.source_png === 'string' && b.source_png.length > 0,
          'the source frame is sent');
    check(Array.isArray(b.patch_rect) && b.patch_rect.length === 4
          && b.patch_rect[2] >= 2 && b.patch_rect[3] >= 2,
          'the placement is sent and is not degenerate',
          JSON.stringify(b.patch_rect));
    check(b.patch_mask_png !== null && b.patch_mask_png !== undefined,
          'the patch marking is sent');
    check(b.mask_refine_png === null,
          'an untouched touch-up layer sends null, not a full-frame grey image',
          String(b.mask_refine_png));
    check(b.fit === 'contain' && b.mock === true,
          'fit/mock reach the server', b.fit + '/' + b.mock);
  }

  // 6. the three-state touch-up layer
  els['refc'].getContext = () => {
    const c = ctx2d(512, 384);
    const d = new Uint8ClampedArray(512 * 384 * 4);
    d[0] = 255; d[3] = 255;            // pixel 0: opaque white = force ON
    d[4] = 0; d[7] = 255;              // pixel 1: opaque black = force OFF
    c.getImageData = () => ({ data: d, width: 512, height: 384 });
    return c;
  };
  const st = (0, eval)('refineStats()');
  check(st.on === 1 && st.off === 1 && st.painted === 2,
        'refineStats counts the forced pixels', JSON.stringify(st));
  const png = (0, eval)('refinePng()');
  check(typeof png === 'string' && png.length > 3,
        'refinePng sends base64 when something was painted', String(png));

  // 7. render() against a real response shape. d_out_content is null because
  //    the server sanitises a NaN into null; the panel must print n/a.
  const j = {
    ok: true, mock: true, rundir: 'runs/st/ui/x', mask_frac: 0.00390625,
    mask_from_patch: true, refine_layer: true,
    refine_forced_on: 0.0002, refine_forced_off: 0.0,
    frame: [512, 384],
    result_png: 'AA', y_png: 'AA', mask_png: 'AA',
    metrics: {
      d_inside: 0.44, d_out_all: 0.01, d_out_content: null,
      d_out_dilution: 1.0, contrast: 3.5, sat_in: 0.28, struct_in: 0.9,
      verdict: { target_changed: false, background_held: true,
                 contrast_ok: false, valid_wholeframe: false },
    },
    y_metrics: {
      d_inside: 0.075, d_out_all: 0.016, d_out_content: 0.0166,
      d_out_dilution: 1.0, contrast: 10.1, sat_in: 1.05, struct_in: 0.9,
      verdict: { target_changed: true, background_held: true,
                 contrast_ok: true, valid_wholeframe: false },
    },
  };
  try {
    (0, eval)('render')(j);
  } catch (e) {
    check(false, 'render() did not throw', String(e));
    return finish();
  }
  const out = els['out'].innerHTML;
  for (const w of ['result', 'Y (the target asked for)', 'derived mask',
                   'd_inside', '0.4400', 'n/a', 'touch-up', 'forced ON',
                   '1.17x margin', '1.01x margin', 'NOT', 'runs/st/ui/x']) {
    check(out.includes(w), 'the result panel shows ' + JSON.stringify(w));
  }
  check(out.includes('class="bad">FAIL'), 'a failing criterion is shown as FAIL');
  check((out.match(/data:image\/png;base64,/g) || []).length >= 3,
        'the panel shows the three images');

  // 8. and it must survive a response with nothing in it
  try {
    (0, eval)('render')({ ok: true, mock: false, rundir: 'r', mask_frac: 0.1,
                          mask_from_patch: false, refine_layer: false,
                          result_png: 'AA', y_png: 'AA', mask_png: 'AA',
                          metrics: {}, y_metrics: {} });
    check(true, 'render() survives a response with no metrics and no verdict');
  } catch (e) {
    check(false, 'render() survives a response with no metrics and no verdict',
          String(e));
  }
  finish();
})();

function finish() {
  if (fails) {
    console.error(fails + ' check(s) failed');
    process.exit(1);
  }
  console.log('page harness OK');
}
