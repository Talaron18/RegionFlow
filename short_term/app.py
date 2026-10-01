#!/usr/bin/env python3
"""
A localhost UI for line 1, built for the interaction a real user actually has:
place a local patch, then draw the mask.

    source image      uploaded
    patch             any local crop, ANY SHAPE, dragged onto the source and
                      scaled with a slider -- the placement is the user's, not
                      an inference
    mask              drawn over the result (there is a "= patch box" button,
                      because the patch footprint IS the sensible initial mask)
    instruction       free text

then it runs the command line that is already measured to work:

    edit.py --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat

WHAT CHANGED FROM THE FIRST VERSION, AND WHY. The first version asked for a
"reference image" and inferred where its content should go from the mask's
bounding box. That is the wrong shape of problem: a user does not have "an
already-edited image", they have a crop of the thing they want -- a snap of a
similar object, from any photo, at any size. So the patch is now placed BY THE
USER (drag + scale) and the mask is drawn AFTER it. Nothing about the placement
is guessed, and `--strategy` is not consulted on this path.

WHY THIS WORKS BEHIND JUPYTERLAB. The page is served with RELATIVE urls
(`api/run`, not `/api/run`) and the handler accepts the request at any path
ending in `api/run`, so it works both at `http://host:PORT/` and at
`https://host/user/.../proxy/PORT/`. The first version posted to an absolute
`/api/run`, which under JupyterLab's proxy went to the notebook server root
instead of this app -- the page loaded and the Run button silently failed.

    # on a remote box, behind the notebook server's proxy
    python short_term/app.py --host 0.0.0.0 --port 8765
    #   then open  <your-jupyterlab-url>/proxy/8765/
    #   (AutoDL: use its "custom service" mapping of port 6006 instead)

    # no GPU / no fork: paste the patch into the mask and stop there. Still
    # exercises upload + mask encoding + patch placement, which is where the
    # shape and polarity bugs live, and doubles as the paste-only baseline.
    python short_term/app.py --host 0.0.0.0 --mock

    python short_term/app.py --self-test
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term.backends import load_image, save_image  # noqa: E402
from short_term.backends import parse_size  # noqa: E402
from short_term.propose_y import compose_patch  # noqa: E402
from short_term.score import score  # noqa: E402

#: Frame shapes the UI offers, WxH. Both dims are multiples of 8 (SD3's VAE
#: works on /8), and 512 is kept as the long edge so nothing is upscaled.
#: 4:3 is the default: a photo taken in 4:3 was previously CENTRE-CROPPED to a
#: square, which throws away the sides of every landscape picture for no reason.
FRAMES = {
    "4:3": (512, 384),
    "1:1": (512, 512),
    "3:4": (384, 512),
    "16:9": (512, 288),
}
DEFAULT_FRAME = "4:3"


def frame_size(name) -> tuple[int, int]:
    """'4:3' or '512x384' or [512, 384] -> (w, h). Validated here, once."""
    if isinstance(name, str) and name in FRAMES:
        return FRAMES[name]
    if name is None:
        return FRAMES[DEFAULT_FRAME]
    w, h = parse_size(name)
    if w % 8 or h % 8:
        raise SystemExit(f"frame {w}x{h} is not a multiple of 8; SD3's VAE "
                         f"works on /8")
    return w, h


#: The measured operating point (README SS5.3). Not knobs to explore here.
DEFAULTS = {
    "guide": "both",
    "beta": "5.0",
    "kappa": "0.6",
    "kappa_out": "1.0",
    "kappa_mode": "flat",
    "steps": 50,
    "image_guidance": 1.5,
    "guidance": 7.5,
    "seed": 0,
    # a concrete pixel size, NOT the frame NAME: handle_run overwrites this from
    # the canvas's real WxH, but it must still be a valid --size on its own
    # (passing "4:3" to edit.py raised "invalid literal for int()").
    "size": "512x384",
}

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>RegionFlow line 1</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#14161a;color:#e8e8ea}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c3038}
 h3{margin:0 0 6px;font-size:13px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.05em}
 .row{display:flex;gap:16px;padding:14px;align-items:flex-start;flex-wrap:wrap}
 .col{background:#1d2026;border:1px solid #2c3038;border-radius:8px;padding:12px}
 .stack{position:relative;width:384px;height:288px;background:#000;border-radius:6px;overflow:hidden;touch-action:none}
 /* width/height are overwritten from JS by layoutFrame(), which is the only
    place that knows W and H. A hardcoded square here is what would make a 4:3
    frame render as a squashed square. */
 .stack canvas{position:absolute;left:0;top:0;width:384px;height:288px}
 /* EVERY canvas gets an explicit box. These rules used to name ids that no
    longer exist (mskc, hl, refc, refsel), so the patch panel had NO positioning
    at all and its three 320px canvases rendered as three squares side by side
    instead of one stacked editor. A stylesheet that names dead ids fails
    silently -- there is no error, just a wrong-looking page. */
 .refwrap{position:relative;width:320px;height:320px;background:#000;border-radius:6px;overflow:hidden;touch-action:none}
 .refwrap canvas{position:absolute;left:0;top:0;width:320px;height:320px}
 #patc,#pml{pointer-events:none}
 #pmc{cursor:crosshair}
 #patchc,#refc{cursor:crosshair}
 #patchc{pointer-events:none}
 #refc{pointer-events:none}
 input[type=file]{color:#c9d1d9;max-width:250px}
 input[type=text],input[type=number],select{background:#0f1115;color:#e8e8ea;border:1px solid #333a45;border-radius:5px;padding:5px 7px}
 button{background:#2f6df6;color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer;font-weight:600}
 button.sec{background:#333a45}
 button.on{background:#16a34a}
 button:disabled{opacity:.5;cursor:default}
 .muted{color:#8b95a3;font-size:12px}
 table{border-collapse:collapse;font-size:13px;margin-top:6px} td,th{padding:3px 9px;text-align:right;border-bottom:1px solid #262a31}
 th:first-child,td:first-child{text-align:left}
 .good{color:#4ade80}.bad{color:#f87171}.warn{color:#fbbf24}
 code{background:#0f1115;padding:1px 5px;border-radius:4px}
 #log{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px;max-height:160px;overflow:auto;background:#0f1115;padding:8px;border-radius:6px}
</style></head><body>
<header><b>RegionFlow - line 1</b> &nbsp;<span class="muted">source + placed patch + drawn mask -&gt; guided edit</span></header>

<div class="row">
 <div class="col">
  <h3>1 - source</h3>
  <input type="file" id="srcfile" accept="image/*">
  <div class="stack" id="stack" style="margin-top:8px">
    <canvas id="srcc" width="512" height="512"></canvas>
    <canvas id="maskprev" width="512" height="512"></canvas>
    <canvas id="patchc" width="512" height="512"></canvas>
    <canvas id="refc" width="512" height="512"></canvas>
  </div>
  <div class="muted" id="srcinfo">no image</div>
   <div class="muted">frame
    <select id="frame">
     <option value="4:3" selected>4:3 &nbsp;512x384</option>
     <option value="1:1">1:1 &nbsp;512x512</option>
     <option value="3:4">3:4 &nbsp;384x512</option>
     <option value="16:9">16:9 &nbsp;512x288</option>
    </select>
    <label><input type="checkbox" id="squash"> squash to the frame (default: centre-crop)</label>
   </div>
  <div class="muted" id="maskinfo" style="margin-top:6px">derived mask: empty</div>
  <div class="muted" id="frameinfo"></div>
  <div class="muted">red = the derived mask (from the marking on the patch).
   <button class="sec" id="mrefine">touch up on source</button>
   <span id="refinehint"></span><br>
   green = you forced a pixel ON, blue = you forced a hole.
   The brush only works in "touch up" mode and starts out transparent, so the
   source stays visible; whatever you do not paint is left exactly as derived.
  </div>
 </div>

 <div class="col">
  <h3>2 - patch, then mark the object IN it</h3>
  <input type="file" id="patchfile" accept="image/*">
  <div class="refwrap" id="patwrap" style="margin-top:8px;background:#000">
    <canvas id="patc" width="320" height="320"></canvas>
    <canvas id="pmc"  width="320" height="320"></canvas>
    <canvas id="pml"  width="320" height="320"></canvas>
  </div>
  <div style="margin:8px 0">
   <button class="sec on" id="mplace">place / scale</button>
   <button class="sec" id="mmark">mark object</button>
  </div>
  <div class="muted" id="patchinfo">no patch - load one, then mark the object inside it</div>
  <div style="margin-top:6px" class="muted">
   size <input type="range" id="pscale" min="4" max="100" value="30" style="width:140px">
   <span id="pct">30%</span>
  </div>
  <div style="margin-top:6px" class="muted">
   brush <input type="range" id="brush" min="3" max="80" value="20" style="width:110px">
   <button class="sec" id="erase">erase</button>
   <button class="sec" id="pall">mark all</button>
   <button class="sec" id="pnone">clear</button>
  </div>
  <div class="muted" style="margin-top:6px">
   grow <input type="number" id="grow" value="0" min="0" max="60" style="width:56px">
   feather <input type="number" id="feather" value="0" min="0" max="40" style="width:56px">
   <label><input type="checkbox" id="stretch"> stretch instead of fit</label>
  </div>
 </div>

 <div class="col" style="min-width:300px">
  <h3>3 - instruction &amp; run</h3>
  <input type="text" id="instr" size="40" placeholder="e.g. A hot air balloon should also be floating."><br>
  <div class="muted" style="margin-top:8px">
   <label>seed <input type="number" id="seed" value="0" style="width:60px"></label>
   <label>steps <input type="number" id="steps" value="50" style="width:60px"></label>
   <label>&kappa; <input type="number" id="kappa" value="0.6" step="0.05" style="width:60px"></label>
   <label>&kappa;<sub>out</sub> <input type="number" id="kappa_out" value="1.0" step="0.05" style="width:60px"></label>
  </div>
  <div style="margin-top:8px"><label class="muted"><input type="checkbox" id="mock"> MOCK: paste the patch, no model</label></div>
  <div style="margin-top:10px"><button id="run">Run edit</button> <span class="muted" id="status"></span></div>
  <div id="out"></div>
 </div>
</div>

<div class="row"><div class="col" style="flex:1"><h3>log</h3><div id="log" class="muted">ready</div></div></div>

<script>
const $=id=>document.getElementById(id);
const PMAX=320;              // the panel's LONG edge, in its own pixels
// THE PANEL IS A SCALE OF THE PATCH, NOT A SQUARE THE PATCH SITS INSIDE.
// It used to be a fixed 320x320 with the patch letterboxed into it, which made
// the panel and the patch different coordinate systems: the mask was painted in
// one and the content was placed in the other, so a non-square patch drifted
// (measured: 5 px at aspect 1.05, 30 px at 4:3, 0 px only when square).
let PW=320, PH=320;
const FRAMES={'4:3':[512,384],'1:1':[512,512],'3:4':[384,512],'16:9':[512,288]};
let W=512, H=384;            // the SOURCE frame, width first
let S=W;                     // kept only so old square-shaped code reads clearly
let srcImg=null, srcNat=null, patchImg=null, patchNat=null;
let rect={x:150,y:150,w:150,h:150};      // placement box, in 512 coords
let drag=null, marking=false, erase=false, last=null, mode='mark';

function setMode(m){mode=m;
  $('mplace').className='sec'+(m==='place'?' on':'');
  $('mmark').className='sec'+(m==='mark'?' on':'');
  $('mrefine').className='sec'+(m==='refine'?' on':'');
  $('pmc').style.pointerEvents=(m==='mark')?'auto':'none';
  $('patchc').style.pointerEvents=(m==='place')?'auto':'none';
  $('refc').style.pointerEvents=(m==='refine')?'auto':'none';
  $('refinehint').textContent=(m==='refine')
    ?' painting on the SOURCE: brush adds (green), "erase" forces a hole'
    :'';}
$('mplace').onclick=()=>setMode('place');
$('mmark').onclick=()=>setMode('mark');
$('mrefine').onclick=()=>setMode('refine');
// ONE initialisation: geometry first, then the mode. setMode was never
// called at load, so the page STARTED with mode='place' but with the
// stylesheet's pointer-events:none still on the patch canvas -- dragging
// the patch did nothing until you clicked 'place / scale' first.
layoutFrame(512,384);
layoutPatch();
setMode('place');
$('erase').onclick=()=>{erase=!erase;$('erase').textContent=erase?'erase ON':'erase';};

// Crop the source to the FRAME's aspect ratio. This used to crop to a SQUARE
// unconditionally, so a 4:3 photo lost its left and right edges for no reason.
// Now a 4:3 photo in a 4:3 frame is used whole, and "squash" still stretches.
function cropBox(nat){
  if($('squash').checked) return {sx:0,sy:0,sw:nat.w,sh:nat.h};
  const target=W/H, ar=nat.w/nat.h;
  if(ar>target){const sh=nat.h, sw=sh*target; return {sx:(nat.w-sw)/2,sy:0,sw,sh};}
  const sw=nat.w, sh=sw/target; return {sx:0,sy:(nat.h-sh)/2,sw,sh};
}
function drawSrc(){
  const g=$('srcc').getContext('2d'); g.fillStyle='#000'; g.fillRect(0,0,W,H);
  if(!srcImg){$('srcinfo').textContent='no image';return;}
  const b=cropBox(srcNat);
  g.drawImage(srcImg,b.sx,b.sy,b.sw,b.sh,0,0,W,H);
  $('srcinfo').textContent=srcNat.w+'x'+srcNat.h+' -> '+W+'x'+H+' '
    +($('squash').checked?'stretched to the frame':'centre-cropped to '
      +(W/H).toFixed(2));
  drawPlacement(); drawMaskPreview();
}
// WHERE THE PATCH ACTUALLY LANDS -- the same computation the server does.
// 'fit' keeps the aspect and centres inside the box; 'stretch' fills it.
function dstRect(){
  if(!patchImg) return {x:rect.x,y:rect.y,w:rect.w,h:rect.h};
  if($('stretch').checked) return {x:rect.x,y:rect.y,w:rect.w,h:rect.h};
  const s=Math.min(rect.w/patchNat.w, rect.h/patchNat.h);
  const w=Math.max(1,Math.round(patchNat.w*s)), h=Math.max(1,Math.round(patchNat.h*s));
  return {x:Math.round(rect.x+(rect.w-w)/2), y:Math.round(rect.y+(rect.h-h)/2), w:w, h:h};
}
// The panel takes the PATCH's aspect, so the patch fills it exactly and there
// is no letterbox: one coordinate system, no offset to get wrong.
function panelSize(nat){
  const f=Math.min(PMAX/nat.w, PMAX/nat.h);
  return {w:Math.max(1,Math.round(nat.w*f)), h:Math.max(1,Math.round(nat.h*f))};
}
function layoutPatch(){
  if(!patchImg){PW=PH=PMAX; setPanelCss(); return;}
  const s=panelSize(patchNat);
  PW=s.w; PH=s.h;
  for(const id of ['patc','pmc','pml']){
    const c=$(id); c.width=PW; c.height=PH;
  }
  setPanelCss();
}
function setPanelCss(){
  const wrap=$('patwrap');
  wrap.style.width=PW+'px'; wrap.style.height=PH+'px';
  for(const id of ['patc','pmc','pml']){
    const c=$(id); c.style.width=PW+'px'; c.style.height=PH+'px';
  }
  blankPatchMask(); blankRefine(); drawPatchPanel(); drawPlacement(); drawMaskPreview();
}
function drawPatchPanel(){
  const g=$('patc').getContext('2d'); g.fillStyle='#000'; g.fillRect(0,0,PW,PH);
  if(!patchImg){$('patchinfo').textContent='no patch - load one, then mark the object inside it';return;}
  // FILL, not fit: the panel already has the patch's aspect (layoutPatch), so
  // filling it is exact and the marking is in the patch's own coordinates.
  g.drawImage(patchImg,0,0,PW,PH);
  const l=$('pml').getContext('2d'); l.clearRect(0,0,PW,PH);
  l.strokeStyle='#4ade80'; l.lineWidth=1; l.strokeRect(0.5,0.5,PW-1,PH-1);
}
// the composite preview: the patch's MARKED content only, at its placed size
function drawPlacement(){
  const g=$('patchc').getContext('2d'); g.clearRect(0,0,W,H);
  if(!patchImg) return;
  const d=dstRect();
  const tmp=document.createElement('canvas'); tmp.width=PW; tmp.height=PH;
  const tg=tmp.getContext('2d');
  tg.drawImage($('pmc'),0,0);
  tg.globalCompositeOperation='source-in';
  tg.drawImage(patchImg,0,0,PW,PH);
  tg.globalCompositeOperation='source-over';
  g.globalAlpha=0.85; g.drawImage(tmp,0,0,PW,PH,d.x,d.y,d.w,d.h); g.globalAlpha=1;
  g.strokeStyle='#4ade80'; g.lineWidth=2; g.setLineDash([7,5]);
  g.strokeRect(rect.x,rect.y,rect.w,rect.h); g.setLineDash([]);
  if(!$('stretch').checked){
    g.strokeStyle='#7aa2ff'; g.setLineDash([3,3]);
    g.strokeRect(d.x,d.y,d.w,d.h); g.setLineDash([]);}
  $('patchinfo').textContent='patch '+patchNat.w+'x'+patchNat.h+' (panel '+PW+'x'+PH
    +', same aspect) -> '+d.w+'x'+d.h+' at '+d.x+','+d.y
    +($('stretch').checked?' (stretched)':' (fit, aspect kept)');
}
function patchMaskFrac(){
  const d=$('pmc').getContext('2d').getImageData(0,0,PW,PH).data; let n=0;
  for(let i=3;i<d.length;i+=4) if(d[i]>127) n++; return n/(PW*PH);
}
// THE UNIFICATION, previewed: the marked patch mask warped into the source
// frame by the SAME placement the content got.
function drawMaskPreview(){
  const d=dstRect();
  const out=document.createElement('canvas'); out.width=W; out.height=H;
  const og=out.getContext('2d'); og.imageSmoothingEnabled=false;
  og.drawImage($('pmc'),0,0,PW,PH,d.x,d.y,d.w,d.h);
  const got=og.getImageData(0,0,W,H);
  const rf=$('refc').getContext('2d').getImageData(0,0,W,H);
  const g=$('maskprev').getContext('2d');
  const o=g.createImageData(W,H); let n=0, fon=0, foff=0;
  for(let i=0;i<got.data.length;i+=4){
    const a=rf.data[i+3], v=rf.data[i];
    // SAME rule as compose_patch: >0.75 ON, <0.25 OFF, else the derived mask
    const on = (a<64) ? got.data[i]>127 : v>127;
    if(a>=64){ if(v>127){o.data[i]=60;o.data[i+1]=230;o.data[i+2]=140;
                         o.data[i+3]=150; fon++;} 
               else    {o.data[i]=120;o.data[i+1]=110;o.data[i+2]=255;
                         o.data[i+3]=190; foff++;} }
    else      { o.data[i]=255;o.data[i+1]=60;o.data[i+2]=60;o.data[i+3]=on?130:0; }
    if(on)n++;}
  g.putImageData(o,0,0);
  $('maskinfo').textContent = (n
    ? 'derived mask: '+(100*n/(W*H)).toFixed(1)+'% of the frame'
    : 'derived mask: empty')
    + (fon||foff ? '   +'+fon+' px forced ON, -'+foff+' px forced OFF'
                 : '   (no touch-up: the mask is exactly the warped marking)');
}
function blankPatchMask(){
  const g=$('pmc').getContext('2d'); g.fillStyle='#000'; g.fillRect(0,0,PW,PH);
}
function blankRefine(){
  const g=$('refc').getContext('2d'); g.clearRect(0,0,W,H);
}
// ONE place that knows the canvas geometry. Called on load and whenever the
// frame changes, BEFORE anything draws.
function layoutFrame(w,h){
  W=w; H=h; S=W;
  const st=$('stack');
  st.style.width='384px'; st.style.height=Math.round(384*H/W)+'px';
  for(const id of ['srcc','maskprev','patchc','refc']){
    const c=$(id); c.width=W; c.height=H;
    c.style.width='384px'; c.style.height=Math.round(384*H/W)+'px';
  }
  if(patchImg){                       // re-centre the placement box
    const w0=W*(+$('pscale').value)/100;
    rect={x:Math.round((W-w0)/2),
          y:Math.round((H-w0*patchNat.h/patchNat.w)/2),
          w:w0, h:w0*patchNat.h/patchNat.w};
  } else {
    rect={x:Math.round(W*0.3),y:Math.round(H*0.3),
          w:Math.round(W*0.3),h:Math.round(H*0.3)};
  }
  blankRefine();
  if(srcImg) drawSrc(); else {drawPlacement(); drawMaskPreview();}
  settingsView();
}
function settingsView(){
  const s=FRAMES[$('frame').value]||[W,H];
  $('frameinfo').textContent='frame '+s[0]+'x'+s[1]+' (WxH, '+
    (s[0]/s[1]).toFixed(2)+':1)';
}
function setFrame(){
  const s=FRAMES[$('frame').value]||[W,H];
  layoutFrame(s[0],s[1]);
}
function refineStats(){
  const d=$('refc').getContext('2d').getImageData(0,0,W,H).data;
  let on=0, off=0;
  for(let i=0;i<d.length;i+=4){
    if(d[i+3]<64) continue;              // transparent -> abstain
    if(d[i]>127) on++; else off++;
  }
  return {on:on, off:off, painted:on+off};
}
// Send time: lay the touch-up over grey 128, because the PNG that reaches the
// server is converted to RGB and alpha would be discarded -- a transparent
// "abstain" pixel would arrive as BLACK, i.e. a forced hole. Grey survives.
function refinePng(){
  const st=refineStats(); if(!st.painted) return null;
  const tmp=document.createElement('canvas'); tmp.width=W; tmp.height=H;
  const g=tmp.getContext('2d');
  g.fillStyle='#808080'; g.fillRect(0,0,W,H);
  g.drawImage($('refc'),0,0);
  return tmp.toDataURL('image/png').split(',')[1];
}

$('srcfile').onchange=e=>{const f=e.target.files[0];if(!f)return;
  const im=new Image();im.onload=()=>{srcImg=im;srcNat={w:im.naturalWidth,h:im.naturalHeight};
    blankRefine(); drawSrc();};
  im.src=URL.createObjectURL(f);};
$('squash').onchange=drawSrc;
$('frame').onchange=setFrame;
$('stretch').onchange=()=>{drawPlacement();drawMaskPreview();};
$('grow').oninput=drawMaskPreview; $('feather').oninput=drawMaskPreview;
$('patchfile').onchange=e=>{const f=e.target.files[0];if(!f)return;
  const im=new Image();im.onload=()=>{patchImg=im;patchNat={w:im.naturalWidth,h:im.naturalHeight};
    const w=W*(+$('pscale').value)/100;
    rect={x:Math.round((W-w)/2),y:Math.round((H-w*patchNat.h/patchNat.w)/2),
          w:w,h:w*patchNat.h/patchNat.w};
    layoutPatch(); setMode('mark');};
  im.src=URL.createObjectURL(f);};
$('pscale').oninput=()=>{
  $('pct').textContent=$('pscale').value+'%';
  if(!patchImg)return;
  const cx=rect.x+rect.w/2, cy=rect.y+rect.h/2, w=W*(+$('pscale').value)/100;
  rect={x:cx-w/2,y:cy-w/2,w:w,h:w*patchNat.h/patchNat.w};
  drawPlacement(); drawMaskPreview();
};
function pos(e,c,W,H){const r=c.getBoundingClientRect();
  return {x:(e.clientX-r.left)*W/r.width, y:(e.clientY-r.top)*H/r.height};}

// place mode: drag the box on the source
const pCv=$('patchc');
pCv.addEventListener('mousedown',e=>{const p=pos(e,pCv,W,H);
  if(p.x>=rect.x&&p.x<=rect.x+rect.w&&p.y>=rect.y&&p.y<=rect.y+rect.h){
    drag={dx:p.x-rect.x,dy:p.y-rect.y}; e.preventDefault();}});
pCv.addEventListener('mousemove',e=>{if(!drag)return;const p=pos(e,pCv,W,H);
  rect.x=Math.max(-rect.w*0.5,Math.min(W-rect.w*0.5,p.x-drag.dx));
  rect.y=Math.max(-rect.h*0.5,Math.min(H-rect.h*0.5,p.y-drag.dy));
  drawPlacement(); drawMaskPreview();});
pCv.addEventListener('mouseleave',()=>{drag=null;});
window.addEventListener('mouseup',()=>{drag=null;marking=false;last=null;});

// mark mode: paint the OBJECT on the patch panel
const mCv=$('pmc');
function paint(e){
  if(!marking)return; const g=mCv.getContext('2d'), p=pos(e,mCv,PW,PH), w=+$('brush').value;
  g.strokeStyle=erase?'#000':'#fff'; g.lineWidth=w; g.lineCap='round';
  g.beginPath(); g.moveTo(last?last.x:p.x,last?last.y:p.y); g.lineTo(p.x,p.y); g.stroke();
  last=p; drawPlacement(); drawMaskPreview();
}
mCv.addEventListener('mousedown',e=>{marking=true;last=null;paint(e);});
mCv.addEventListener('mousemove',paint);
mCv.addEventListener('mouseleave',()=>{marking=false;last=null;});
// refine mode: paint on the SOURCE (force ON / force OFF)
const rCv=$('refc');
function paintRef(e){
  if(!marking)return; const g=rCv.getContext('2d'),
    p=pos(e,rCv,W,H), w=+$('brush').value*W/PMAX;
  g.globalCompositeOperation='source-over';
  g.strokeStyle=erase?'#000':'#fff'; g.lineWidth=w; g.lineCap='round';
  g.beginPath(); g.moveTo(last?last.x:p.x,last?last.y:p.y); g.lineTo(p.x,p.y); g.stroke();
  last=p; drawMaskPreview();
}
rCv.addEventListener('mousedown',e=>{marking=true;last=null;paintRef(e);});
rCv.addEventListener('mousemove',paintRef);
rCv.addEventListener('mouseleave',()=>{marking=false;last=null;});
$('pall').onclick=()=>{const g=mCv.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,PW,PH);
  drawPlacement();drawMaskPreview();};
$('pnone').onclick=()=>{blankPatchMask();drawPlacement();drawMaskPreview();};

function b64(c){return c.toDataURL('image/png').split(',')[1];}

// The result panel. This function was CALLED but never DEFINED, so every run
// ended in "ReferenceError: render is not defined" inside the try, was caught
// by the catch, and reported as a bare "error" with no picture -- the UI could
// run the model and still never show anything.
function render(j){
  const M=j.metrics||{}, Y=j.y_metrics||{};
  const f=x=>(x===null||x===undefined)?'n/a'
    :(typeof x==='number'?x.toFixed(4):String(x));
  const pc=x=>(100*x).toFixed(2)+'%';
  let h='<div class="muted" style="margin-top:8px">'
    +(j.mock?'MOCK (the patch pasted into the mask, no model)':'real model run')
    +' &middot; mask '+pc(j.mask_frac)+' of the frame'
    +' &middot; '+(j.mask_from_patch?'mask derived from the marking on the patch'
                                   :'mask = the whole footprint (nothing marked)')
    +(j.refine_layer?(' &middot; touch-up: '+j.refine_forced_on.toFixed(4)
                      +' forced ON, '+j.refine_forced_off.toFixed(4)+' forced OFF')
                    :' &middot; no touch-up')
    +'</div>';
  h+='<div class="muted" style="word-break:break-all">'+j.rundir+'</div>';
  const fig=(src,lab)=>'<figure style="margin:6px 8px 0 0;display:inline-block;'
    +'vertical-align:top"><img alt="'+lab+'" src="data:image/png;base64,'+src
    +'" style="width:148px;border:1px solid #2c3038;border-radius:4px"><br>'
    +'<figcaption class="muted">'+lab+'</figcaption></figure>';
  h+=fig(j.result_png,'result');
  h+=fig(j.y_png,'Y (the target asked for)');
  h+=fig(j.mask_png,'derived mask');
  const rows=[['d_inside','changed inside (higher = more)'],
              ['d_out_all','changed outside, all'],
              ['d_out_content','changed outside, non-white'],
              ['d_out_dilution','dilution (1.0 = clean)'],
              ['contrast','contrast ratio (flatters)'],
              ['sat_in','saturation in mask'],
              ['struct_in','structure in mask']];
  h+='<table><tr><th>metric</th><th>result</th><th>Y</th></tr>';
  for(const r of rows)
    h+='<tr><td>'+r[0]+'</td><td>'+f(M[r[0]])+'</td><td>'+f(Y[r[0]])+'</td></tr>';
  h+='</table>';
  // the server's own pre-registered criteria, rendered as pass/fail -- with the
  // two that have no measured margin marked, because reading them as verdicts
  // is how a change gets declared without evidence
  const NOMARGIN={'target_changed':'1.17x margin','valid_wholeframe':'1.01x margin'};
  const v=M.verdict||{}, vy=Y.verdict||{};
  if(Object.keys(v).length){
    h+='<table><tr><th>criterion</th><th>result</th><th>Y</th></tr>';
    for(const k in v)
      h+='<tr><td>'+k+(NOMARGIN[k]?' <span class="warn">('+NOMARGIN[k]+')</span>':'')
        +'</td><td class="'+(v[k]?'good':'bad')+'">'+(v[k]?'pass':'FAIL')+'</td>'
        +'<td class="'+((vy[k]===undefined)?'':(vy[k]?'good':'bad'))+'">'
        +((vy[k]===undefined)?'n/a':(vy[k]?'pass':'FAIL'))+'</td></tr>';
    h+='</table>';
    h+='<div class="muted">The criteria marked with a margin of ~1.0 are NOT '
      +'verdicts: on the dataset\'s own ground-truth edit they sit within 1.2x '
      +'of their threshold, so they cannot separate a good edit from a bad one. '
      +'Read d_inside and d_out_content against the Y column instead.</div>';
  }
  $('out').innerHTML=h;
}

$('run').onclick=async()=>{
  $('out').innerHTML='';
  if(!srcImg){$('log').textContent='load a source image first';return;}
  if(!patchImg){$('log').textContent='load a patch first';return;}
  if(patchMaskFrac()<=0){
    $('log').textContent='mark the object INSIDE the patch first: click "mark object", '
      +'then paint over the object. The mask is derived from that marking, so an '
      +'unmarked patch would edit nothing (or, worse, paste the patch\'s own background).';
    return;}
  $('run').disabled=true; $('status').textContent='running...'; $('log').textContent='';
  const payload={source_png:b64($('srcc')), patch_png:b64($('patc')),
    patch_mask_png:b64($('pmc')), patch_rect:[rect.x,rect.y,rect.w,rect.h],
    mask_refine_png:refinePng(),
    fit:$('stretch').checked?'stretch':'contain',
    grow:+$('grow').value, feather:+$('feather').value,
    instruction:$('instr').value, mock:$('mock').checked,
    seed:+$('seed').value, steps:+$('steps').value,
    kappa:+$('kappa').value, kappa_out:+$('kappa_out').value};
  try{
    const r=await fetch('api/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
    const j=await r.json();
    $('status').textContent=j.ok?'done':'FAILED';
    $('log').textContent=(j.log||'')+(j.error?('\n\n'+j.error):'');
    if(j.ok)render(j);
  }catch(e){$('status').textContent='error';$('log').textContent=String(e);}
  $('run').disabled=false;
};
</script></body></html>
"""


# ------------------------------------------------------------ pure helpers --

def b64_png_to_array(s: str) -> np.ndarray:
    """base64 PNG (from a browser canvas) -> (H,W,3) float [0,1]."""
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    return np.asarray(img, np.float32) / 255.0


def array_to_b64_png(a: np.ndarray) -> str:
    from PIL import Image
    arr = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    buf = io.BytesIO()
    Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def validate_rect(rect, size: int, frame_h: int | None = None
                  ) -> tuple[int, int, int, int]:
    """[x, y, w, h] in canvas pixels -> ints, sanity-checked.

    `size` is the frame WIDTH and `frame_h` its height, because the frame is not
    necessarily square. The parameter is deliberately NOT called `h`: the rect
    itself unpacks to an `h` two lines below, and when this parameter was named
    `h` the frame height silently became the RECT's height -- so a placement
    near the bottom of a square frame was refused as off-canvas "for a 512x90
    frame". Caught by lab.py's self-test, which places a patch at y=150 with
    height 90.

    A zero or negative size is the single most likely thing to arrive from a
    browser that has a patch loaded but has not drawn it yet, and it would
    otherwise paste nothing and look like a model failure.
    """
    if rect is None:
        raise SystemExit("no patch_rect: the browser did not report where the "
                         "patch was placed")
    try:
        x, y, w, h = (float(v) for v in rect)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"patch_rect must be four numbers, got {rect!r}") from exc
    if w < 2 or h < 2:
        raise SystemExit(f"patch_rect is degenerate ({w:g}x{h:g} px); load a "
                         "patch and scale it before running")
    fh = size if frame_h is None else frame_h
    if not (-size <= x <= size and -fh <= y <= fh):
        raise SystemExit(f"patch_rect is off-canvas at ({x:g},{y:g}) for a "
                         f"{size}x{fh} frame")
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def find_produced_output(rundir: Path,
                         exclude: tuple[str, ...] = ("src.png", "mask.png",
                                                     "y.png",
                                                     "out_inputs.png")
                         ) -> Path | None:
    """The image edit.py just wrote into this rundir.

    A GLOB, not a fixed name. The first version looked for `rundir/out.png`
    while edit.py appends the parameter tag (`out_both_b5_kin0.6_kout1_flat.png`
    -- `--out` is a stem), so a run that exited 0 and wrote a perfectly good
    image was reported to the user as "no output was written". The inputs the
    server itself wrote are excluded by name; whatever else is a PNG is the
    result. app.py now also passes `--out-exact`, so this is belt and braces.
    """
    if not rundir.is_dir():
        return None
    cands = [p for p in sorted(rundir.glob("*.png")) if p.name not in exclude]
    return cands[-1] if cands else None


def edit_failure(code, rundir, listing, stdout, stderr) -> str:
    """The message for a failed model run. It must be diagnosable WITHOUT a
    round trip.

    This used to report only the exit code and the directory listing, so the
    one thing that explains the failure -- which edit.py had already printed to
    stderr -- stayed in a log panel that is easy to miss. Reaching a GPU box to
    discover that a missing file caused it is the expensive way to find out.
    """
    tail = ((stderr or "").strip() or (stdout or "").strip())
    tail = tail[-1200:] if tail else "(edit.py printed nothing at all)"
    return (f"edit.py exited {code} and no result PNG was found in {rundir}. "
            f"Its contents are: {listing}. If a result IS there, this is a bug "
            "in the lookup, not in the run.\n\n"
            f"What edit.py said:\n{tail}\n\n"
            f"(also saved to {Path(rundir) / 'edit_log.txt'})")


def build_edit_argv(py: str, edit_py: Path, src: Path, mask: Path, y: Path,
                    instruction: str, p: dict) -> list[str]:
    """The command line, built in one place so it can be asserted on.

    Reusing edit.py through a subprocess rather than calling the classes in
    process: it runs the exact code path that is already measured, it cannot take
    the server down with it, and its .json sidecar comes back for free.
    """
    return [
        py, str(edit_py),
        "--image", str(src), "--mask", str(mask),
        "--guide-image", str(y), "--instruction", instruction,
        "--guide", p["guide"],
        "--beta", str(p["beta"]),
        "--kappa", str(p["kappa"]), "--kappa-out", str(p["kappa_out"]),
        "--kappa-mode", p["kappa_mode"],
        "--steps", str(p["steps"]),
        "--size", str(p["size"]),
        "--image-guidance", str(p["image_guidance"]),
        "--guidance", str(p["guidance"]),
        "--seed", str(p["seed"]),
        "--out", str(p["out"]),
        # The literal path, not a stem with the parameter tag appended: the
        # caller has to be able to find the file it asked for.
        "--out-exact",
    ]


def metrics_for(src: np.ndarray, out: np.ndarray, mask: np.ndarray) -> dict:
    s = score(src, out, mask > 0.5)
    return {k: s[k] for k in ("d_inside", "d_out_all", "d_out_content",
                              "d_out_dilution", "contrast", "sat_in",
                              "struct_in", "verdict")}


def handle_run(payload: dict, cfg) -> dict:
    """One request: place the patch, derive the mask FROM it, run, score.

    THE MASK IS NO LONGER DRAWN ON THE SOURCE. It is drawn on the PATCH -- the
    object inside the crop -- and then carried to the source frame by the same
    placement that positions the content (`compose_patch`). Before this, the
    mask was freehand on the source while the patch was pasted as a hard
    rectangle, so a patch with its own background pasted that background in
    (photographed: a black square over a diagram) and the mask and the content
    could disagree about where the edit was. One placement, one geometry.
    """
    src = b64_png_to_array(payload["source_png"])
    patch_png = payload.get("patch_png")
    if patch_png is None:
        return {"ok": False,
                "error": "no patch uploaded. It is the only input that says "
                         "WHAT the masked region should become (README "
                         "section 0: without it exp_gain stays at -5.4 and the "
                         "background is destroyed)."}
    patch = b64_png_to_array(patch_png)
    pm_png = payload.get("patch_mask_png")
    patch_mask = None if pm_png is None else b64_png_to_array(pm_png)[..., 0]

    try:
        # src is (H, W, 3); the frame is WxH
        rect = validate_rect(payload.get("patch_rect"),
                             src.shape[1], src.shape[0])
    except SystemExit as exc:
        return {"ok": False, "error": str(exc)}

    fit = str(payload.get("fit") or "contain")
    grow = float(payload.get("grow") or 0.0)
    feather = float(payload.get("feather") or 0.0)
    rf_png = payload.get("mask_refine_png")
    refine = None if rf_png is None else b64_png_to_array(rf_png)[..., 0]
    # counted once, used by both the sidecar and the response
    rf_on = float((refine > 0.75).mean()) if refine is not None else 0.0
    rf_off = float((refine < 0.25).mean()) if refine is not None else 0.0

    y, mask = compose_patch(src, patch, patch_mask, rect, fit=fit, grow=grow,
                           feather=feather, refine_mask=refine)
    if float((mask > 0.5).mean()) <= 0.0:
        return {"ok": False,
                "error": "the derived mask is empty. Mark the object INSIDE "
                         "the patch (the brush panel) -- an empty mask means "
                         "the guidance has nothing to act on."}

    p = dict(DEFAULTS)
    for k, cast in (("seed", int), ("steps", int),
                    ("kappa", float), ("kappa_out", float)):
        if payload.get(k) is not None:
            p[k] = cast(payload[k])
    # WxH, taken from the source the browser actually sent -- not from a
    # separate field that could disagree with it. edit.py resizes to this, so
    # without it a 4:3 frame would be squashed back to a square.
    p["size"] = f"{src.shape[1]}x{src.shape[0]}"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    rundir = Path(cfg.out_root) / stamp
    rundir.mkdir(parents=True, exist_ok=True)
    src_p, mask_p, y_p = rundir / "src.png", rundir / "mask.png", rundir / "y.png"
    save_image(src, src_p)
    save_image(mask, mask_p)
    save_image(y, y_p)

    log, record, out_arr, mock = "", {}, None, bool(payload.get("mock"))
    if mock:
        out_arr = y
        # Also write out.png, so the rundir looks the same in both modes. A
        # caller (or a person) looking for "the result" should not have to know
        # that mock mode called it something else.
        save_image(y, rundir / "out.png")
        record = {"mode": "MOCK (patch pasted into the mask; no model run)",
                  "mock": True}
        log = ("MOCK: the result IS the placed patch. This validates the "
               "upload, the mask encoding and the patch placement, and it is "
               "the \"what if we just pasted it\" baseline -- it is NOT the "
               "model's output and must not be scored as one.\n")
    else:
        out_png = rundir / "out.png"
        argv = build_edit_argv(sys.executable,
                               Path(__file__).parent / "edit.py",
                               src_p, mask_p, y_p,
                               str(payload.get("instruction") or ""),
                               {**p, "out": str(out_png)})
        log = " ".join(argv) + "\n"
        import subprocess
        r = subprocess.run(argv, capture_output=True, text=True)
        log += (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:]
        # Keep it on disk. The log panel is easy to close, and the one thing
        # needed to explain an exit 1 is the one thing that must not be lost.
        (rundir / "edit_log.txt").write_text(
            f"exit code: {r.returncode}\nargv: {' '.join(argv)}\n\n"
            f"--- stdout ---\n{r.stdout or ''}\n--- stderr ---\n{r.stderr or ''}")
        produced = find_produced_output(rundir)
        if r.returncode != 0 or produced is None:
            # Say what IS there. "no output was written" was wrong once and
            # cost a round trip; a directory listing cannot be wrong. AND say
            # why: see edit_failure().
            listing = sorted(p.name for p in rundir.iterdir())
            return {"ok": False, "log": log, "rundir": str(rundir),
                    "error": edit_failure(r.returncode, rundir, listing,
                                          r.stdout, r.stderr)}
        out_arr = load_image(produced, (src.shape[1], src.shape[0]))
        side = produced.with_suffix(".json")
        if side.is_file():
            record = json.loads(side.read_text())

    (rundir / "y_meta.json").write_text(json.dumps({
        "instruction": payload.get("instruction"), "patch_rect": list(rect),
        "fit": fit, "grow": grow, "feather": feather,
        "patch_mask_from_patch": patch_mask is not None,
        "refine_layer": refine is not None,
        "refine_forced_on": rf_on,
        "refine_forced_off": rf_off,
        "params": p, "mock": mock,
        "frame": [src.shape[1], src.shape[0]],
        "mask_frac": float((mask > 0.5).mean()),
        "patch_rect_frac": float((rect[2] * rect[3])
                                 / (src.shape[1] * src.shape[0])),
        "record": record,
    }, indent=1))
    save_image(mask, rundir / "derived_mask.png")

    return {
        "ok": True, "log": log, "mock": mock, "rundir": str(rundir),
        "record": record,
        "mask_frac": float((mask > 0.5).mean()),
        "mask_from_patch": patch_mask is not None,
        "refine_layer": refine is not None,
        "refine_forced_on": rf_on,
        "refine_forced_off": rf_off,
        "frame": [src.shape[1], src.shape[0]],
        "result_png": array_to_b64_png(out_arr),
        "y_png": array_to_b64_png(y),
        "mask_png": array_to_b64_png(mask),
        "metrics": metrics_for(src, out_arr, mask > 0.5),
        "y_metrics": metrics_for(src, y, mask > 0.5),
    }


# ------------------------------------------------------------- the server ---

def json_safe(obj):
    """Replace non-finite floats with None, recursively.

    `json.dumps` emits bare `NaN`, `Infinity` and `-Infinity` by default, which
    are NOT valid JSON: the browser's JSON.parse throws
    "Unexpected token 'N' ... is not valid JSON" and the whole response is lost
    -- which is how a completed run reported as "SyntaxError" with no result.
    The metrics genuinely contain NaN (a dilution ratio with no background to
    dilute) and inf (a perfect contrast), so this is not a hypothetical.

    `allow_nan=False` at the dump site is the belt to this pair of braces. It
    matters because np.float32 is NOT a subclass of float (`np.float64` is), so
    a metric that skipped its `float()` cast used to reach json.dumps as an
    unserialisable object, raise TypeError, and lose the whole response -- the
    same symptom (a finished run reported as a syntax error) from a second
    cause. The last line is the guarantee: an unrepresentable value becomes a
    string, never an exception, because the user's result is worth more than a
    tidy type.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):                 # np.float32/int64/bool_ ...
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def make_handler(cfg):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "short_term_ui"

        def log_message(self, fmt, *a):        # keep stderr readable
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _route(path: str) -> str:
            """'/api/run' and '/proxy/8765/api/run' must both mean the same
            thing. JupyterLab's proxy is why: it strips its own prefix, but
            which prefix depends on the deployment, so match on the SUFFIX."""
            return path.split("?")[0].rstrip("/")

        def do_GET(self):
            # Serve the page for anything that is not the API. That makes the
            # app work at '/', at '/index.html' and at any '/proxy/PORT/...'
            # prefix without knowing the prefix.
            if self._route(self.path).endswith("/api/run"):
                self._send(405, b"use POST", "text/plain")
                return
            self._send(200, PAGE.encode("utf-8"),
                       "text/html; charset=utf-8")

        def do_POST(self):
            if not self._route(self.path).endswith("/api/run"):
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("content-length") or 0)
            if n <= 0 or n > 64 * 1024 * 1024:
                self._send(400, b'{"ok":false,"error":"bad content-length"}',
                           "application/json")
                return
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                result = handle_run(payload, cfg)
            except SystemExit as exc:
                result = {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001 -- the UI must not 500
                result = {"ok": False,
                          "error": f"{type(exc).__name__}: {exc}"}
            try:
                body = json.dumps(json_safe(result),
                                  allow_nan=False).encode("utf-8")
            except Exception as exc:  # noqa: BLE001 -- never lose the response
                body = json.dumps(
                    {"ok": False,
                     "error": f"the result could not be encoded: "
                              f"{type(exc).__name__}: {exc}"}).encode("utf-8")
            self._send(200, body, "application/json")

    return Handler


def proxy_url(lab_url: str, port: int) -> str:
    """A JupyterLab page URL -> the URL that reaches THIS app through the
    notebook server's port proxy.

    Jupyter Server proxies any port at `<base>/proxy/<port>/`, and the base is
    whatever precedes `/lab/`, `/tree/`, `/notebooks/` ... in the address bar.
    So from

        https://h:8443/jupyter/lab/tree/RegionFlow/short_term

    the answer is

        https://h:8443/jupyter/proxy/8765/

    and under JupyterHub, `/user/alice/lab/tree/x` -> `/user/alice/proxy/8765/`,
    which is also right because the base there is `/user/alice/`.

    Doing this by hand is exactly the kind of thing that gets it wrong once and
    wastes an hour, so it is a function with a test.
    """
    from urllib.parse import urlsplit, urlunsplit

    u = urlsplit(lab_url if "://" in lab_url else "https://" + lab_url)
    # Segment-exact, not substring: a base like '/laboratory/' must not be
    # mistaken for a '/lab' marker, and a bare 'https://h/lab' must truncate to
    # '/' rather than be left alone.
    markers = ("lab", "tree", "notebooks", "terminals", "edit", "view",
               "nbconvert")
    segs = [s for s in (u.path or "/").split("/") if s]
    for i, s in enumerate(segs):
        if s in markers:
            segs = segs[:i]
            break
    path = "/" + "/".join(segs)
    if segs:
        path += "/"
    return urlunsplit((u.scheme or "https", u.netloc, f"{path}proxy/{port}/",
                       "", ""))


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Is something already listening on this port?

    Worth a pre-flight because on these GPU platforms the two published HTTP
    ports (6006/6008 on seetacloud and AutoDL) are exactly the ones something
    else may already occupy, and "Address already in use" from deep inside
    socketserver does not say which of the two options to take next.
    """
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def proxy_urls(lab_url: str, port: int) -> list[str]:
    """Both proxy URL forms, because which one exists depends on the install.

    `/proxy/<port>/` is NOT part of jupyter_server -- it is provided by the
    separate `jupyter-server-proxy` package (or the older notebook's own
    handler). If that package is missing the route simply does not exist and the
    server answers 404, which is exactly what happened on this box. The package
    documents two forms:

        <base>/proxy/<port>/           URL rewritten: the prefix is stripped
        <base>/proxy/absolute/<port>/  full URL forwarded

    `short_term`'s router matches on the SUFFIX (`.../api/run`), so it works
    under either. Printing both costs nothing and saves a round trip.
    """
    base = proxy_url(lab_url, port)
    absolute = base.replace("/proxy/", "/proxy/absolute/", 1)
    return [base, absolute] if absolute != base else [base]


def ssh_tunnel_command(ssh: str, port: int, local_port: int | None = None
                       ) -> str | None:
    """'root@connect.bjb1.seetacloud.com:20718' -> the `ssh -L` line that makes
    this app reachable at http://127.0.0.1:<port>/ on the user's own machine.

    This is the route that needs no cooperation from anyone: not the platform's
    port publishing, not Jupyter's proxy, not an install. If you can ssh to the
    box -- and anyone running a remote notebook usually can -- you can forward a
    port. Pure string work, so it is tested against the real connection string.
    """
    s = (ssh or "").strip()
    if not s:
        return None
    user = ""
    rest = s
    if "@" in s:
        user, _, rest = s.rpartition("@")
        user += "@"
    host, sep, ssh_port = rest.rpartition(":")
    if not sep or not ssh_port.isdigit():
        return None
    lp = local_port or port
    return f"ssh -p {ssh_port} -L {lp}:127.0.0.1:{port} {user}{host}"


def port_guidance(port: int, published: tuple[int, ...], lab_url,
                  reachable_via_proxy: bool = True,
                  ssh: str | None = None) -> list[str]:
    """What the user can actually do to open this thing, given which ports the
    platform publishes. Pure, so it can be tested without a socket."""
    out = []
    if ssh:
        cmd = ssh_tunnel_command(ssh, port)
        if cmd:
            out.append("  from your own machine, with SSH -- needs NOTHING "
                       "published and nothing installed:")
            out.append(f"      {cmd}")
            out.append(f"      then open  http://127.0.0.1:{port}/")
    if published and port in published:
        out.append(f"port {port} IS one the platform publishes "
                   f"({', '.join(str(p) for p in published)}): open its public "
                   "URL from the platform console, on the port/service page. "
                   "That bypasses Jupyter entirely and needs nothing installed.")
    else:
        pub = ", ".join(str(p) for p in published) if published else "(none)"
        out.append(f"port {port} is NOT one the platform publishes ({pub}). "
                   "Reaching it needs the Jupyter proxy below, or a re-run on a "
                   "published port.")
        if published:
            out.append(f"  to use a published port directly:  --host 0.0.0.0 "
                       f"--port {published[0]}   (check it is free first)")
    if reachable_via_proxy:
        if lab_url:
            urls = proxy_urls(lab_url, port)
        else:
            urls = [f"<your-jupyterlab-url>/proxy/{port}/",
                    f"<your-jupyterlab-url>/proxy/absolute/{port}/"]
        out.append("  through JupyterLab (needs the jupyter-server-proxy "
                   "package -- without it this 404s):")
        for u in urls:
            out.append(f"      {u}")
    out.append("  no port at all:      python -c \"from short_term import lab; "
               "lab.run_line1(...)\"   (see README section 0.3)")
    return out


def serve(cfg) -> int:
    from http.server import ThreadingHTTPServer

    def say(*a):
        # flush=True: started from a JupyterLab cell or redirected to a log,
        # stdout is a pipe and would otherwise block-buffer, so the URL the
        # whole tool depends on could sit invisible for minutes.
        print(*a, flush=True)

    handler = make_handler(cfg)
    if cfg.port and port_in_use(cfg.port, "127.0.0.1"):
        say(f"!! port {cfg.port} is ALREADY IN USE by something else on this "
            "machine.")
        say("   Binding would fail with 'Address already in use'. Try:")
        for p in (*cfg.published_ports, 0):
            say(f"     --port {p if p else 0}"
                + ("   (a published port - check the console for its URL)"
                   if p else "   (let the OS pick a free one)"))
        return 2
    try:
        httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    except OSError as exc:
        say(f"!! could not bind {cfg.host}:{cfg.port}: {exc}")
        say("   On a shared box the port may be taken, or the host may not "
            "allow it.")
        return 2
    port = httpd.server_address[1]
    local = f"http://127.0.0.1:{port}/"
    say(f"short_term UI listening on {cfg.host}:{port}")
    say(f"  from this machine     : {local}")
    for line in port_guidance(port, tuple(cfg.published_ports), cfg.lab_url,
                              ssh=getattr(cfg, "ssh", None)):
        say(line)
    say("    The page posts to a RELATIVE url, so a /proxy/PORT/ path works "
        "unchanged.")
    say("    404 on the proxy = the jupyter-server-proxy package is not "
        "installed (the")
    say("    /proxy/ route is not part of jupyter_server). Either pip install "
        "it and")
    say("    restart the Jupyter server, or use the platform's published-port "
        "URL above,")
    say("    which needs nothing installed. 403 = allow_remote_access.")
    if cfg.host == "127.0.0.1":
        say("    --host 0.0.0.0 is required for anything but a browser on this "
            "same machine.")
    if cfg.mock:
        say("  --mock: Run pastes the patch into the mask instead of calling "
            "the model. No GPU, no fork needed.")
    say("  NOTE: starting this does NOT change the JupyterLab page. The UI is a "
        "separate")
    say("        server at one of the addresses above; open that one.")
    say("  Ctrl-C to stop.")
    if cfg.open:
        threading.Timer(0.6, lambda: webbrowser.open(local)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        say("\nstopping")
    return 0


# ------------------------------------------------------------- self-test ----

def _self_test() -> int:
    """CPU, no model, no browser. Exercises the parts this box can reach: the
    pure composite, the request handler in --mock, the argv builder -- the last
    of which is run for real with --dry-run so the CLI has to accept it -- and
    the proxy-path routing, which is what broke the first version."""
    import tempfile
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    S = 64
    src = (rng.random((S, S, 3)) * 0.4 + 0.4).astype(np.float32)
    mask = np.zeros((S, S, 3), np.float32)
    mask[16:48, 24:40] = 1.0
    patch = (rng.random((S, S, 3)) * 0.4).astype(np.float32)

    # base64 round trip, exactly what the browser sends
    assert np.allclose(b64_png_to_array(array_to_b64_png(src)), src, atol=2e-2)

    # patch_rect validation: the browser sends floats, and an unplaced patch
    # sends a degenerate box
    assert validate_rect([10.4, 20.6, 30.2, 40.9], 512) == (10, 21, 30, 41)
    assert validate_rect([-5, 0, 20, 20], 512) == (-5, 0, 20, 20)
    # the frame HEIGHT must come from the caller, never from the rect. With the
    # parameter named `h` it was shadowed by the rect's own h, so this placement
    # -- well inside a 512-tall frame -- was refused as "off-canvas for a
    # 512x10 frame".
    assert validate_rect([0, 400, 10, 10], 512, 512) == (0, 400, 10, 10)
    assert validate_rect([0, 200, 10, 10], 512, 384) == (0, 200, 10, 10)
    try:
        validate_rect([0, 400, 10, 10], 512, 300)
    except SystemExit as exc:
        assert "512x300 frame" in str(exc), exc
    else:
        raise AssertionError("y beyond the frame height must be refused")
    for bad in (None, [1, 2, 3], [0, 0, 0, 10], [0, 0, 10, 0],
                [0, 0, 1, 1], [9000, 0, 10, 10]):
        try:
            validate_rect(bad, 512)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"patch_rect {bad!r} must be refused")

    # the page must be relative and proxy-safe
    assert "fetch('api/run'" in PAGE, "the POST must be relative or the " \
                                     "JupyterLab proxy path breaks it"
    assert "fetch('/api/run'" not in PAGE
    assert PAGE.isascii(), "keep the source ASCII so the repo's locale-" \
                           "sensitive check_names.py can read it"
    assert "patch_rect" in PAGE and "patch_mask_png" in PAGE
    assert "mark the object INSIDE the patch" in PAGE, \
        "the page must send a mask drawn on the PATCH, not on the source"
    # the touch-up layer, and the three-state encoding it must not lose
    assert "mask_refine_png" in PAGE and "refinePng" in PAGE
    assert "fillStyle='#808080'" in PAGE, \
        "the touch-up layer must travel as grey-128 for 'leave it alone', " \
        "because the PNG is converted to RGB and transparency would arrive as " \
        "black -- a forced hole everywhere the user did not paint"
    assert "clearRect(0,0,W,H)" in PAGE, \
        "the touch-up canvas must start transparent so the source shows through"

    # the frame: one knob, WxH, and every offered shape must be loadable by SD3
    for name, want in (("4:3", (512, 384)), ("1:1", (512, 512)),
                       ("3:4", (384, 512)), ("16:9", (512, 288))):
        assert frame_size(name) == want, (name, frame_size(name))
        w, h = frame_size(name)
        assert w % 8 == 0 and h % 8 == 0, f"{name} has a fractional latent grid"
    assert frame_size("512x384") == (512, 384)
    assert frame_size([512, 384]) == (512, 384)
    assert frame_size(None) == FRAMES[DEFAULT_FRAME] == (512, 384)
    assert parse_size(DEFAULTS["size"]) == FRAMES[DEFAULT_FRAME], \
        "the default --size must be the default frame in pixels, not its name"
    assert DEFAULT_FRAME in PAGE, "the page must default to the same frame"
    try:
        frame_size("500x384")
    except SystemExit as exc:
        assert "multiple of 8" in str(exc), exc
    else:
        raise AssertionError("a fractional latent grid must be refused")

    # the failure message must be diagnosable without a round trip
    msg = edit_failure(1, Path("/tmp/rd"), ["src.png", "mask.png"],
                       "stdout noise", "Traceback: boom\nSystemExit: nope")
    assert "exited 1" in msg and "src.png" in msg, msg
    assert "SystemExit: nope" in msg, \
        "the stderr tail is the whole point: without it the cause is invisible"
    assert "edit_log.txt" in msg, msg
    quiet = edit_failure(-9, Path("/tmp/rd"), [], "", "")
    assert "printed nothing at all" in quiet, quiet
    long = edit_failure(1, Path("/tmp/rd"), [], "x" * 9000, "")
    assert len(long) < 2500, len(long)

    # EVERY id the page names must exist, in the JS and in the stylesheet. The
    # photographed bug was a stylesheet naming ids that no longer existed
    # (mskc, hl, refc, refsel): nothing errors, the patch panel simply has no
    # positioning and its three canvases render side by side as three squares.
    # A stylesheet that names a dead id fails SILENTLY, so it needs a test.
    import re
    page_js = PAGE[PAGE.index("<script>") + len("<script>"):
                   PAGE.index("</script>")]
    page_css = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    defined = set(re.findall(r"""id=["']([A-Za-z0-9_-]+)["']""", PAGE))
    used = set(re.findall(r"""\$\(['"]([A-Za-z0-9_-]+)['"]\)""", page_js))
    selected = {n for n in re.findall(r"#([A-Za-z][A-Za-z0-9_-]*)", page_css)
                if not re.fullmatch(r"[0-9a-fA-F]{3,8}", n)}   # not a hex colour
    assert not (used - defined), \
        f"the page's JS reaches for {sorted(used - defined)}, which do not exist"
    assert not (selected - defined), \
        f"the stylesheet names {sorted(selected - defined)}: a dead id is not an " \
        "error, it is a wrongly laid out page"
    assert {"refc", "pmc", "maskprev"} <= defined, "the three mask layers"
    assert "function render(" in page_js, \
        "the Run handler calls render(j): while it was undefined, every " \
        "finished run ended in a caught ReferenceError shown as bare 'error'"

    # The page's JavaScript, actually parsed AND executed (a DOM stub stands in
    # for the browser). A syntax error in it disables every button on the page
    # and is invisible from here -- checking that the HTML merely mentions an id
    # is not enough. node is optional: without it this is skipped, not failed.
    import shutil
    import subprocess
    node = shutil.which("node")
    harness = Path(__file__).parent / "page_harness.js"
    if node and harness.is_file():
        with tempfile.TemporaryDirectory() as jsdir:
            js = Path(jsdir) / "page.js"
            js.write_text(page_js, encoding="utf-8")
            r = subprocess.run([node, str(harness), str(js)],
                               capture_output=True, text=True)
            assert r.returncode == 0, \
                "the page's JavaScript failed:\n" + r.stdout[-2000:] + \
                r.stderr[-2000:]
    else:
        print("  (node not found: the page's JavaScript was NOT executed)")

    # json_safe: the response must survive JSON.parse. NaN and inf are REAL
    # (a dilution ratio with no background to dilute, a perfect contrast), and
    # np.float32 is NOT a float subclass, so it used to reach json.dumps as an
    # unserialisable object and cost the user the whole response.
    nan, inf = float("nan"), float("inf")
    for obj, want in ((nan, None), (inf, None), (-inf, None), (0.5, 0.5),
                      (np.float32("nan"), None), (np.float32(0.25), 0.25),
                      (np.float64("inf"), None), (np.int64(7), 7),
                      (np.bool_(True), True), (True, True), (None, None),
                      (Path("x.png"), "x.png")):
        got = json_safe(obj)
        assert got == want or (got is None and want is None), (obj, got, want)
        assert type(got) in (type(want), bool), (obj, got, type(got))
    deep = json_safe({"a": [nan, {"b": np.float32("nan"), 1: inf}],
                      "m": np.array([1.0, nan]), "p": Path("x")})
    assert deep == {"a": [None, {"b": None, "1": None}],
                    "m": [1.0, None], "p": "x"}, deep
    body = json.dumps(deep, allow_nan=False)
    assert json.loads(body) == deep
    for bad in ("NaN", "Infinity"):
        assert bad not in body, body

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_root=tmp, host="127.0.0.1", port=0,
                              mock=True, open=False)
        # the patch is mostly BLACK with one bright square -- and only the
        # square is marked. That is the photographed failure: a hard-pasted
        # rectangle would put the black background into Y.
        pat = np.zeros((S, S, 3), np.float32)
        pat[8:24, 8:24] = 1.0
        pmask = np.zeros((S, S), np.float32)
        pmask[8:24, 8:24] = 1.0
        payload = {
            "source_png": array_to_b64_png(src),
            "patch_png": array_to_b64_png(pat),
            "patch_mask_png": array_to_b64_png(pmask),
            "patch_rect": [24, 16, 16, 16],
            "instruction": "make it something else", "mock": True,
            "seed": 0, "steps": 4, "kappa": 0.6, "kappa_out": 1.0,
        }
        res = handle_run(payload, cfg)
        assert res["ok"], res
        assert res["mock"] and res["record"]["mock"], res["record"]
        # every key the page's render() reads must be in the response, or the
        # result panel throws AFTER a successful, expensive run
        for k in ("log", "mock", "rundir", "record", "mask_frac",
                  "mask_from_patch", "refine_layer", "refine_forced_on",
                  "refine_forced_off", "result_png", "y_png", "mask_png",
                  "metrics", "y_metrics"):
            assert k in res, f"render() reads {k}, which the response lacks"
        for f in ("src.png", "mask.png", "y.png", "out.png",
                  "derived_mask.png", "y_meta.json"):
            assert Path(res["rundir"], f).is_file(), f
        m, ym = res["metrics"], res["y_metrics"]
        # in mock mode the result IS Y, so the two must be identical
        assert abs(m["d_inside"] - ym["d_inside"]) < 1e-9
        assert m["d_inside"] > 0.05, m["d_inside"]
        assert res["mask_from_patch"] is True
        assert 0.0 < res["mask_frac"] < 1.0
        # the derived mask must be strictly inside the placement box: the patch
        # was mostly unmarked background
        assert res["mask_frac"] < (16 * 16) / (S * S), res["mask_frac"]

        # WITHOUT a patch mask the whole footprint is used -- the old behaviour,
        # kept for a tightly cropped patch
        no_pm = dict(payload)
        no_pm.pop("patch_mask_png")
        r2 = handle_run(no_pm, cfg)
        assert r2["ok"] and r2["mask_from_patch"] is False, r2
        assert r2["mask_frac"] > res["mask_frac"]

        # an all-black patch mask derives an empty mask -> a real refusal, since
        # editing nothing is never what was asked for
        empty = dict(payload,
                     patch_mask_png=array_to_b64_png(np.zeros((S, S),
                                                              np.float32)))
        r = handle_run(empty, cfg)
        assert r["ok"] is False and "empty" in r["error"], r

        # THE TOUCH-UP LAYER. 128 = abstain, 255 = force ON, 0 = force OFF.
        base = res["mask_frac"]
        abstain = np.full((S, S), 0.5, np.float32)
        r3 = handle_run(dict(payload,
                             mask_refine_png=array_to_b64_png(abstain)), cfg)
        assert r3["ok"] and abs(r3["mask_frac"] - base) < 1e-6, (r3, base)
        assert r3["refine_layer"] and r3["refine_forced_on"] == 0.0

        # force ON well outside the derived mask -> the union is bigger
        on = abstain.copy()
        on[40:50, 40:50] = 1.0
        r4 = handle_run(dict(payload, mask_refine_png=array_to_b64_png(on)), cfg)
        assert r4["ok"], r4
        assert abs(r4["mask_frac"] * S * S - (base * S * S + 100)) <= 1.5, \
            (r4["mask_frac"], base)
        assert abs(r4["refine_forced_on"] - 100.0 / (S * S)) < 1e-6

        # force OFF over the ONLY marked content -> nothing is left, and that is
        # a refusal rather than a silent no-op edit
        off = abstain.copy()
        off[16:24, 24:32] = 0.0           # the warped marking, patch 8:24 -> 18:22,26:30
        r5 = handle_run(dict(payload, mask_refine_png=array_to_b64_png(off)), cfg)
        assert r5["ok"] is False and "empty" in r5["error"], r5

        # a wrong-sized layer is refused, not broadcast or cropped
        try:
            handle_run(dict(payload,
                            mask_refine_png=array_to_b64_png(
                                np.zeros((8, 8), np.float32))), cfg)
        except SystemExit as exc:
            assert "refine_mask" in str(exc), exc
        else:
            raise AssertionError("a wrong-sized refine layer must be refused")

        # error paths stay JSON and never 500
        for name, mod, want in (
                ("no patch", {"patch_png": None}, "no patch"),
                ("bad rect", {"patch_rect": [0, 0, 0, 0]}, "degenerate"),
                ("no rect", {"patch_rect": None}, "patch_rect")):
            r = handle_run(dict(payload, **mod), cfg)
            assert r["ok"] is False and want in r["error"], (name, r)

        # the argv builder, run FOR REAL with --dry-run so the CLI must accept it
        import subprocess
        d = Path(tmp)
        save_image(src, d / "src.png")
        save_image(mask[..., 0] > 0.5, d / "mask.png")
        save_image(patch, d / "y.png")
        argv = build_edit_argv(sys.executable, Path(__file__).parent / "edit.py",
                               d / "src.png", d / "mask.png", d / "y.png",
                               "make it something else",
                               {**DEFAULTS, "out": str(d / "out.png")})
        assert argv[argv.index("--guide") + 1] == "both"
        assert argv[argv.index("--kappa-out") + 1] == "1.0"
        assert "--out-exact" in argv, \
            "without this the output name gains a parameter tag and the caller " \
            "cannot find it (this exact bug reported a successful run as " \
            "'no output was written')"
        r = subprocess.run(argv + ["--dry-run"], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
        assert "tier D" in r.stdout, r.stdout[-500:]

        # and --out-exact refuses a sweep, where it would silently overwrite
        sweep = build_edit_argv(sys.executable,
                                Path(__file__).parent / "edit.py",
                                d / "src.png", d / "mask.png", d / "y.png", "x",
                                {**DEFAULTS, "out": str(d / "o.png")})
        sweep[sweep.index("--kappa") + 1] = "0.3 0.6"
        r = subprocess.run(sweep + ["--dry-run"], capture_output=True, text=True)
        assert r.returncode != 0 and "out-exact" in (r.stdout + r.stderr), \
            (r.stdout[-500:], r.stderr[-500:])

        # THE LOOKUP, which is what broke. A rundir holding the server's own
        # inputs plus edit.py's TAGGED output name must resolve to the output.
        rd = d / "lookup"
        rd.mkdir()
        for n in ("src.png", "mask.png", "y.png"):
            save_image(np.zeros((4, 4), np.float32), rd / n)
        tagged = rd / "out_both_b5_kin0.6_kout1_flat.png"
        save_image(patch, tagged)
        assert find_produced_output(rd) == tagged, find_produced_output(rd)
        # the old behaviour: looking for exactly out.png found nothing
        assert not (rd / "out.png").is_file()
        # inputs only -> None, so the caller reports a real failure
        rd2 = d / "lookup2"
        rd2.mkdir()
        for n in ("src.png", "mask.png", "y.png"):
            save_image(np.zeros((4, 4), np.float32), rd2 / n)
        assert find_produced_output(rd2) is None
        assert find_produced_output(d / "does_not_exist") is None

        # A NON-SQUARE FRAME, end to end in mock. Everything downstream of the
        # canvas used to assume a square: the mask, Y, the metrics and -- the
        # one that would have silently squashed a 4:3 photo -- the model's
        # --size, which build_edit_argv never passed at all.
        Wn, Hn = 96, 72                      # 4:3, both multiples of 8
        wsrc = (rng.random((Hn, Wn, 3)) * 0.4 + 0.4).astype(np.float32)
        wpat = np.zeros((Hn, Wn, 3), np.float32)
        wpat[24:48, 30:66] = 1.0
        wmask = np.zeros((Hn, Wn), np.float32)
        wmask[24:48, 30:66] = 1.0
        rn = handle_run({"source_png": array_to_b64_png(wsrc),
                         "patch_png": array_to_b64_png(wpat),
                         "patch_mask_png": array_to_b64_png(wmask),
                         "patch_rect": [24, 18, 48, 36],
                         "instruction": "x", "mock": True}, cfg)
        assert rn["ok"], rn
        assert rn["frame"] == [Wn, Hn], rn["frame"]
        assert 0.0 < rn["mask_frac"] < 1.0, rn["mask_frac"]
        # the saved inputs must all be the same non-square shape
        for name in ("src.png", "mask.png", "y.png", "out.png",
                     "derived_mask.png"):
            im = load_image(Path(rn["rundir"], name))
            assert im.shape[:2] == (Hn, Wn), (name, im.shape)

        ym = json.loads(Path(rn["rundir"], "y_meta.json").read_text())
        assert ym["frame"] == [Wn, Hn], ym["frame"]
        assert ym["params"]["size"] == f"{Wn}x{Hn}", ym["params"]["size"]
        # and the mask is not a stretched square: the derived region is wider
        # than it is tall, which it can only be if the geometry is per-axis
        m = load_image(Path(rn["rundir"], "derived_mask.png"))[..., 0] > 0.5
        ys, xs = np.nonzero(m)
        assert (xs.max() - xs.min()) > (ys.max() - ys.min()), \
            "the derived mask should be wider than tall for a 4:3 frame"

        # A REAL FAILING MODEL RUN, end to end, without a GPU: point the argv
        # builder at a command that exits 1 and prints a reason. This is the
        # exact shape of the failure the user hit ("exited 1 and no result PNG
        # was found"), whose cause used to be invisible in the error.
        real_argv = globals()["build_edit_argv"]
        try:
            globals()["build_edit_argv"] = lambda *a, **k: [
                sys.executable, "-c",
                "import sys; sys.stderr.write('BOOM: the diffusers fork is "
                "missing\\n'); sys.exit(1)"]
            rf = handle_run(dict(payload, mock=False), cfg)
        finally:
            globals()["build_edit_argv"] = real_argv
        assert rf["ok"] is False, rf
        assert "exited 1" in rf["error"], rf["error"]
        assert "BOOM: the diffusers fork is missing" in rf["error"], \
            "the reason must be IN the error, not only in the log panel"
        assert "edit_log.txt" in rf["error"], rf["error"]
        logf = Path(rf["rundir"]) / "edit_log.txt"
        assert logf.is_file(), logf
        assert "BOOM" in logf.read_text(), logf.read_text()[:300]
        assert "exit code: 1" in logf.read_text()
        # the error is valid JSON-encodable, like every other response
        assert json.loads(json.dumps(json_safe(rf), allow_nan=False))["ok"] is False

        # A NON-SQUARE PATCH, end to end: what the browser now sends (a panel
        # with the PATCH's aspect) must come back with the marked object
        # covered by the derived mask. This is the drift the user photographed:
        # the old square panel letterboxed a non-square patch, so the marking
        # and the content were in two coordinate systems -- 5 px of drift at
        # aspect 1.05, 30 px at 4:3, and ZERO for a square patch, which is why
        # every earlier test passed.
        pwp, php = 48, 30                     # 8:5, not 4:3 and not square
        panel = np.zeros((php, pwp, 3), np.float32)
        panel[..., 2] = 0.15
        panel[6:18, 10:40] = 1.0              # the object, off-centre on purpose
        mark = np.zeros((php, pwp), np.float32)
        mark[6:18, 10:40] = 1.0
        rq = handle_run({"source_png": array_to_b64_png(
                             np.full((Hn, Wn, 3), 0.5, np.float32)),
                         "patch_png": array_to_b64_png(panel),
                         "patch_mask_png": array_to_b64_png(mark),
                         "patch_rect": [20, 10, 50, 40],
                         "instruction": "x", "mock": True}, cfg)
        assert rq["ok"], rq
        yy = load_image(Path(rq["rundir"], "y.png"))
        mm = load_image(Path(rq["rundir"], "derived_mask.png"))[..., 0] > 0.5
        # Same rule as the mask, on purpose: inside the mask
        # Y_R = 0.5*(1-alpha) + 1.0*alpha, so Y_R > 0.75 IS alpha > 0.5. A
        # "brightness > 0.9" test instead shrinks the region by the blended
        # edge and reports a couple of pixels of "drift" that are only the
        # threshold's fault.
        bright = yy[..., 0] > 0.75
        assert bright.sum() > 0 and mm.sum() > 0, (bright.sum(), mm.sum())
        bys, bxs = np.nonzero(bright)
        mys, mxs = np.nonzero(mm)
        for nm, a, b in (("x0", bxs.min(), mxs.min()), ("x1", bxs.max(), mxs.max()),
                         ("y0", bys.min(), mys.min()), ("y1", bys.max(), mys.max())):
            assert abs(int(a) - int(b)) <= 1, (nm, a, b, pwp, php)


        # the argv must carry --size, or the model squashes the frame to square
        wav = build_edit_argv(sys.executable, Path(__file__).parent / "edit.py",
                              d / "src.png", d / "mask.png", d / "y.png", "x",
                              {**DEFAULTS, "size": "96x72",
                               "out": str(d / "o.png")})
        assert wav[wav.index("--size") + 1] == "96x72", wav
        r = subprocess.run(wav + ["--dry-run"], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-800:], r.stderr[-800:])
        assert "latent grid 12x9" in r.stdout, r.stdout[-400:]

        # THE INVARIANT THE USER ACTUALLY HIT: a real POST must come back as
        # text the browser's JSON.parse accepts. Driven over a real socket, so
        # the header, the body and the handler's own dump are all exercised --
        # a NaN in the metrics used to arrive as the literal `NaN` token and
        # turned a finished run into "SyntaxError: ... is not valid JSON".
        import http.server
        import threading
        import urllib.request
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/api/run"
            # all-white source and all-white patch: the outside-the-mask
            # "content" set is empty, which is exactly how a real photo produces
            # a NaN metric rather than a hypothetical one
            white = np.ones((S, S, 3), np.float32)
            hot = dict(payload, source_png=array_to_b64_png(white),
                       patch_png=array_to_b64_png(white),
                       patch_mask_png=array_to_b64_png(np.ones((S, S),
                                                              np.float32)))
            for tag, body in (("mock", hot), ("refined", dict(
                    hot, mask_refine_png=array_to_b64_png(abstain)))):
                req = urllib.request.Request(
                    url, json.dumps(body).encode("utf-8"),
                    {"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")
                assert resp.status == 200, resp.status
                # json.loads ACCEPTS the bare NaN token by default, which is why
                # a textual "NaN" not in raw would be the wrong check -- and
                # flaky, because base64 payloads contain that substring by
                # chance. parse_constant is called ONLY for NaN/Infinity, so
                # raising here is exactly the browser's JSON.parse.
                def _reject(tok):
                    raise AssertionError(
                        f"the body carries the non-JSON token {tok!r}, which "
                        "is what made the browser report a SyntaxError")
                j = json.loads(raw, parse_constant=_reject)
                assert j["ok"], (tag, j)
                assert j["record"] and j["record"].get("mode"), tag
                # the numeric metrics may legitimately be None (a NaN that was
                # sanitised) -- but never a bare NaN token, which parse_constant
                # above has already ruled out for the whole body
                def scalars(v):
                    if isinstance(v, dict):
                        return all(scalars(x) for x in v.values())
                    if isinstance(v, (list, tuple)):
                        return all(scalars(x) for x in v)
                    return v is None or isinstance(v, (str, bool, float, int))
                for k, v in j["metrics"].items():
                    assert scalars(v), (k, v, j["record"])
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)

        # proxy-path routing: every one of these must reach the API
        H = make_handler(cfg)
        for path, want in (("/api/run", True),
                           ("/proxy/8765/api/run", True),
                           ("/user/me/proxy/8765/api/run?x=1", True),
                           ("/api/other", False), ("/", False)):
            got = H._route(path).endswith("/api/run")
            assert got == want, (path, got, want)

        # the proxy URL derivation, pinned to the real seetacloud address bar
        assert proxy_url("https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443"
                         "/jupyter/lab/tree/RegionFlow/short_term", 8765) == \
            "https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443" \
            "/jupyter/proxy/8765/"
        for lab, want in (
                ("https://h/user/alice/lab/tree/x", "https://h/user/alice/proxy/9/"),
                ("https://h/lab", "https://h/proxy/9/"),
                ("https://h/jupyter/tree/a/b.ipynb", "https://h/jupyter/proxy/9/"),
                ("https://h/notebooks/x.ipynb", "https://h/proxy/9/"),
                ("http://h:8888/", "http://h:8888/proxy/9/"),
                ("h:8443/jupyter/lab", "https://h:8443/jupyter/proxy/9/")):
            got = proxy_url(lab, 9)
            assert got == want, (lab, got, want)

    # port guidance: the two platform-published ports change the advice, and
    # telling a user to open an unpublished port is the failure this prevents
    LAB = "https://h:8443/jupyter/lab/tree/RegionFlow/short_term"
    both = proxy_urls(LAB, 6006)
    assert both == ["https://h:8443/jupyter/proxy/6006/",
                    "https://h:8443/jupyter/proxy/absolute/6006/"], both
    unpub = " ".join(port_guidance(8765, (6006, 6008), LAB))
    assert "NOT one the platform publishes" in unpub, unpub
    assert "jupyter-server-proxy" in unpub, unpub
    assert "https://h:8443/jupyter/proxy/8765/" in unpub, unpub
    assert "https://h:8443/jupyter/proxy/absolute/8765/" in unpub, unpub
    assert "--port 6006" in unpub, unpub
    pub = " ".join(port_guidance(6006, (6006, 6008), LAB))
    assert "IS one the platform publishes" in pub, pub
    assert "platform console" in pub, pub
    assert "bypasses Jupyter" in pub, pub
    none = " ".join(port_guidance(8765, (), None))
    assert "(none)" in none and "lab.run_line1" in none, none

    # the ssh-tunnel route, pinned to the real connection string
    SSH = "root@connect.bjb1.seetacloud.com:20718"
    assert ssh_tunnel_command(SSH, 6006) == \
        "ssh -p 20718 -L 6006:127.0.0.1:6006 root@connect.bjb1.seetacloud.com"
    assert ssh_tunnel_command("root@h:22", 8765, local_port=9000) == \
        "ssh -p 22 -L 9000:127.0.0.1:8765 root@h"
    assert ssh_tunnel_command("h", 1) is None            # no port: unusable
    assert ssh_tunnel_command("root@h:abc", 1) is None   # non-numeric port
    assert ssh_tunnel_command("", 1) is None
    tun = " ".join(port_guidance(6006, (6006, 6008), LAB, ssh=SSH))
    assert "ssh -p 20718 -L 6006:127.0.0.1:6006" in tun, tun
    assert "needs NOTHING published and nothing installed" in tun, tun
    assert "http://127.0.0.1:6006/" in tun, tun
    assert port_in_use(1) in (True, False)     # callable, no socket needed here

    print("app self-test OK")
    print("  port guidance distinguishes a published port from 8765, prints BOTH")
    print("  proxy URL forms, the exact ssh -L line, and the no-port fallback")
    print("  the produced-output LOOKUP finds edit.py's tagged name (the bug that")
    print("  reported a successful, exited-0 run as 'no output was written')")
    print("  the mask is DERIVED from a marking on the patch: a mostly-black")
    print("  patch pastes only its marked object, an unmarked patch is refused,")
    print("  and no patch mask falls back to the footprint")
    print("  the touch-up layer is three-state (abstain/ON/OFF), travels as")
    print("  grey-128 because RGB conversion would turn transparency into a hole,")
    print("  and forcing a hole over the only marked object is refused")
    print("  json_safe maps NaN/inf/np.float32 to valid JSON, and a REAL POST over")
    print("  a socket parses with parse_constant raising (what JSON.parse does)")
    print("  a NON-SQUARE frame (4:3) runs end to end in mock: src/mask/y/out are")
    print("  all HxW, the derived mask is wider than tall, and --size reaches")
    print("  edit.py (it did not before, which would have squashed a 4:3 photo)")
    print("  a FAILED model run reports WHY: the stderr tail is in the error and")
    print("  in <rundir>/edit_log.txt, instead of only an exit code and a listing")
    print("  the page's own JavaScript is PARSED AND RUN (page_harness.js): a")
    print("  syntax error or a called-but-undefined function is caught here")
    print("  a non-square patch (8:5) keeps its marking ON the object: the")
    print("  derived mask covers the pasted marker to within 1 px (this used to")
    print("  drift 5-30 px, and 0 for a square patch, which hid it)")
    print("  base64 round trip; patch_rect validation incl. unplaced patch")
    print("  the POST is RELATIVE and routes under /proxy/PORT/ (JupyterLab)")
    print("  the argv for the real run is accepted by edit.py --dry-run")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 so the notebook server can proxy to it")
    ap.add_argument("--port", type=int, default=8765, help="0 picks a free one")
    ap.add_argument("--out-root", default="runs/st/ui")
    ap.add_argument("--mock", action="store_true",
                    help="Run pastes the patch into the mask instead of calling "
                         "the model: no GPU, no fork, validates the whole "
                         "upload/mask/placement path")
    ap.add_argument("--lab-url", default=None,
                    help="your JupyterLab address bar, e.g. "
                         "https://h:8443/jupyter/lab/tree/RegionFlow/short_term "
                         "-- the startup banner then prints the exact "
                         "--proxy-- URL to open instead of a template")
    ap.add_argument("--published-ports", default="6006,6008",
                    help="the HTTP ports the PLATFORM publishes (seetacloud and "
                         "AutoDL publish 6006 and 6008). Anything else is only "
                         "reachable through the Jupyter proxy. Comma-separated; "
                         "empty means 'none published'.")
    ap.add_argument("--ssh", default=None,
                    help="your ssh target, e.g. "
                         "root@connect.bjb1.seetacloud.com:20718 -- the banner "
                         "then prints the exact 'ssh -L' line that makes this "
                         "app reachable at http://127.0.0.1:PORT/ on your own "
                         "machine. This route needs no published port, no "
                         "Jupyter proxy and no install.")
    ap.add_argument("--open", action="store_true", help="open a local browser")
    args = ap.parse_args(argv)
    args.published_ports = tuple(
        int(p) for p in str(args.published_ports).replace(",", " ").split()
        if p.strip().isdigit())
    # Remember --lab-url. It is a 90-character paste and every later run needs
    # the same one; forgetting it drops back to a template link for no reason.
    cache = Path(args.out_root) / ".lab_url"
    if args.lab_url:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(args.lab_url.strip(), encoding="utf-8")
        except OSError as exc:
            print(f"[app] could not remember --lab-url ({exc}); ignoring.")
    elif cache.is_file():
        remembered = cache.read_text(encoding="utf-8").strip()
        if remembered:
            args.lab_url = remembered
            print(f"[app] using the remembered --lab-url ({cache}); pass "
                  "--lab-url to change it, or delete that file.")
    return serve(args)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
