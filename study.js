"use strict";

const data = STIMULI;
const $ = id => document.getElementById(id);
const faces = Uint8Array.from(atob(data.faces), c => c.charCodeAt(0));
const keys = data.names.map(name => name.replaceAll(" ", "_"));
const files = new Map();
const rows = [];
let parentDir, sessionDir, session, folderName, pending, started, busy = false, finished = false;
let startAttempt, currentEvidence;

function bit(bytes, i) { return (bytes[i >> 3] >> (7 - (i & 7))) & 1; }
function encodeBits(values) {
  const packed = new Uint8Array(Math.ceil(values.length / 8));
  values.forEach((v,i) => { packed[i >> 3] |= v << (7 - (i & 7)); });
  return btoa(String.fromCharCode(...packed));
}
function decodeBits(s) { return Uint8Array.from(atob(s), c => c.charCodeAt(0)); }
function recipe(id) {
  return Array.from({length:6}, (_,j) => Math.floor(id / 3**(5-j)) % 3);
}
function stateId(values) { return values.reduce((id,k) => 3*id+k, 0); }
function faceBit(id,i) { return bit(faces, id*1600+i); }
function drawFace(canvas, id, evidence) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(40,40);
  for (let i=0; i<1600; i++) {
    const value = evidence ? evidence.observed[i] : faceBit(id,i);
    const shade = value ? 0 : 255;
    const rgba = evidence && !evidence.mask[i] ? [184,204,224,255] : [shade,shade,shade,255];
    img.data.set(rgba,4*i);
  }
  ctx.putImageData(img,0,0);
}
function faceCanvas(id) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 40;
  drawFace(canvas,id);
  return canvas;
}
// Mulberry32. Independent streams keep the images paired when hint settings change.
function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state+0x6d2b79f5) >>> 0;
    let n = Math.imul(state ^ (state >>> 15), state | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffled(values,rng) {
  const a = [...values];
  for (let i=a.length-1; i>0; i--) {
    const j = Math.floor(rng()*(i+1));
    [a[i],a[j]] = [a[j],a[i]];
  }
  return a;
}
function evidenceFor(t) {
  const blockBits = decodeBits(t.visible_blocks), observedBits = decodeBits(t.observed_bits);
  const mask = new Uint8Array(1600), observed = new Uint8Array(1600);
  let visible=0, flips=0;
  for (let i=0;i<1600;i++) {
    const block = Math.floor(i/200)*8 + Math.floor((i%40)/5);
    mask[i] = bit(blockBits,block);
    observed[i] = bit(observedBits,i);
    visible += mask[i];
    if (mask[i] && observed[i] !== faceBit(t.state_id,i)) flips++;
  }
  return {mask,observed,visible,flips};
}
function makePlan(config) {
  const rng=random(config.seed), hintRng=random(config.seed ^ 0x861f93aa);
  const countRng=random(config.seed ^ 0xd257a114), ids=shuffled(data.heldOut,rng);
  const [low,high]=config.hint_range;
  const totals=Array.from({length:high-low+1},(_,i)=>i+low);
  const conditions=[];
  for (const visibility of data.visibility) {
    const noises=shuffled([...data.noise,...data.noise],rng);
    const counts=[];
    while (counts.length<noises.length) counts.push(...shuffled(totals,hintRng));
    const hintTotals=shuffled(counts.slice(0,noises.length),hintRng);
    noises.forEach((noise,i)=>conditions.push({visibility,noise,total:hintTotals[i]}));
  }
  return shuffled(conditions,rng).map(({visibility,noise,total},i)=>{
    const id=ids[i], state=recipe(id);
    const stimulusSeed=Math.floor(rng()*2**32), cueSeed=Math.floor(rng()*2**32);
    const imgRng=random(stimulusSeed), cueRng=random(cueSeed);
    const blocks=Array.from({length:64},()=>Number(imgRng()<visibility));
    const observed=Array.from({length:1600},(_,p)=>{
      const flip=Number(imgRng()<noise), block=Math.floor(p/200)*8+Math.floor((p%40)/5);
      return blocks[block] ? faceBit(id,p)^flip : 0;
    });
    const correct=config.hint_accuracy===null ? Math.floor(countRng()*(total+1)) :
      Array.from({length:total},()=>Number(countRng()<config.hint_accuracy)).reduce((a,b)=>a+b,0);
    const wrong=state.map(v=>(v+1+Math.floor(cueRng()*2))%3);
    const cues=Array(6).fill(-1);
    shuffled([0,1,2,3,4,5],cueRng).slice(0,total).forEach((j,k)=>{cues[j]=k<correct?state[j]:wrong[j];});
    return {trial:i+1,state_id:id,visibility,noise,cues_given:total,correct_cues:correct,
      hint_counts_shown:config.show_hint_counts,stimulus_seed:stimulusSeed,cue_seed:cueSeed,
      visible_blocks:encodeBits(blocks),observed_bits:encodeBits(observed),cue_values:cues};
  });
}

function csv(records) {
  const cols=Object.keys(records[0]);
  const quote=v=>'"'+String(v ?? "").replaceAll('"','""')+'"';
  return [cols,...records.map(r=>cols.map(k=>r[k]))].map(row=>row.map(quote).join(",")).join("\r\n")+"\r\n";
}
function score(t,choice,ev,seconds) {
  const truth=recipe(t.state_id);
  const row={trial:t.trial,state_id:t.state_id,renderer_version:data.version,
    stimulus_seed:t.stimulus_seed,cue_seed:t.cue_seed,visibility_pct:Math.round(100*t.visibility),
    actual_visibility_pct:ev.visible/16,noise_pct:Math.round(100*t.noise),visible_pixels:ev.visible,flipped_pixels:ev.flips,
    flipped_pct:ev.visible ? 100*ev.flips/ev.visible : "",cues_given:t.cues_given,correct_cues:t.correct_cues,
    incorrect_cues:t.cues_given-t.correct_cues,hint_counts_shown:Number(t.hint_counts_shown),
    score:choice.filter((k,j)=>k===truth[j]).length};
  keys.forEach((key,j)=>{
    row["true_"+key]=data.values[j][truth[j]];
    row["chosen_"+key]=data.values[j][choice[j]];
    row["correct_"+key]=Number(truth[j]===choice[j]);
    row["cue_"+key]=t.cue_values[j]<0 ? "" : data.values[j][t.cue_values[j]];
  });
  return Object.assign(row,{source:"participant",subject:session.subject,session:session.session,
    response_seconds:Math.round(seconds*1000)/1000, prior_weight:"",cue_weight:"",
    feedback_png:"trial_"+String(t.trial).padStart(2,"0")+".png"});
}
function makeFeedback(t,choice,row) {
  const ctx=$("comparison").getContext("2d");
  ctx.fillStyle="white"; ctx.fillRect(0,0,1080,480);
  ctx.fillStyle="#233c43"; ctx.textAlign="center"; ctx.font="bold 30px Arial";
  ctx.fillText("Trial "+t.trial+"     |     "+row.score+" of 6 features correct",540,55);
  ctx.imageSmoothingEnabled=false;
  const tiles=[$("observed"),faceCanvas(stateId(choice)),faceCanvas(t.state_id)];
  const labels=["What you saw","Your choice","Original face"];
  tiles.forEach((tile,j)=>{
    ctx.drawImage(tile,40+j*360,92,280,280);
    ctx.font="23px Arial"; ctx.fillText(labels[j],180+j*360,420);
  });
}
function makeSummary(records) {
  const c=$("summary").getContext("2d");
  c.fillStyle="white";c.fillRect(0,0,1000,500);
  c.fillStyle="#233c43";c.font="bold 25px Arial";c.fillText("Average features recovered",60,40);
  c.font="18px Arial";c.fillText("Requested visibility",20,76);
  const levels=["#873b18","#b86632","#da8443","#eca15d","#f7c797","#50bfc8","#006b67"];
  data.visibility.forEach((v,i)=>{
    const y=90+i*65;
    c.fillStyle="#233c43";c.fillText(Math.round(100*v)+"%",55,y+32);
    for(let k=0;k<7;k++){
      const group=records.filter(r=>r.visibility_pct===Math.round(100*v) && r.correct_cues===k);
      const mean=group.length ? group.reduce((sum,r)=>sum+r.score,0)/group.length : null;
      c.fillStyle=mean===null ? "#f0f2f0" : levels[Math.round(mean)]; c.fillRect(150+k*115,y,108,56);
      c.fillStyle=mean!==null && (mean<1.5 || mean>=5.5) ? "white" : "#233c43";
      c.fillText(mean===null ? "–" : mean.toFixed(1),190+k*115,y+34);
    }
  });
  c.fillStyle="#233c43";
  for(let k=0;k<7;k++) c.fillText(k,194+k*115,438);
  c.fillText("Correct hints supplied",470,472);
}
const png=canvas=>new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error("PNG creation failed.")),"image/png"));
function show(id,stage) {
  for(const s of ["setup","guide","trial","feedback","complete"]) $(s).hidden=s!==id;
  $("stage").textContent=stage;
  $("error").textContent="";
  $("save-actions").hidden=!session || (Boolean(sessionDir) && !finished);
  $(id).querySelector("h1")?.focus({preventScroll:true});
  window.scrollTo(0,0);
}
function report(e) { $("error").textContent=e.message || String(e); }
function showGuide(typical=false) {
  show("guide","Getting ready");
  const title=typical?"Typical faces":"Learn the six features";
  $("guide").querySelector("h1").textContent=title;
  $("guide").querySelector("p").textContent=typical ?
    "These are common combinations. Any combination is possible. Choose the features you think were originally shown." :
    "Each row shows the three choices for one feature. Eye spacing changes the distance between the eyes. Layout changes where the features sit.";
  const im=$("guide").querySelector("img");
  im.src=data.slides[typical?"typical_faces.png":"feature_guide.png"];
  im.alt=typical ? data.categories.join(", ")+" example faces" : data.order.map(j=>data.names[j]+": "+data.values[j].join(", ")).join(". ");
  $("guide-next").textContent=typical?"Begin 50 trials":"Next";
  $("guide-next").onclick=typical?showTrial:()=>showGuide(true);
}
function showTrial() {
  const t=session.trials[rows.length];
  show("trial","Trial "+t.trial+" of 50");
  $("trial").querySelector("h1").textContent="Trial "+t.trial+" of 50";
  $("instructions").textContent="Choose one option in every row to recover the original face. "+
    (session.cued?"Refer to the hints, but note that some can be incorrect. ":"")+
    "Blue-gray blocks are hidden. Visible black and white pixels can contain noise. Feel free to take an educated guess when too little of the face is visible to decide. Keep in mind the typical combinations you saw at the beginning.";
  currentEvidence=evidenceFor(t);
  drawFace($("observed"),t.state_id,currentEvidence);
  $("visible").textContent=(currentEvidence.visible/16).toFixed(1)+"% visible";
  const phrases=data.order.filter(j=>t.cue_values[j]>=0).map(j=>data.hints[j][t.cue_values[j]]);
  $("hints").textContent=session.cued ? (phrases.length?"Hints: "+phrases.join(", ")+".":"No hints this trial.")+
    (t.hint_counts_shown?"\n"+t.correct_cues+" correct · "+(t.cues_given-t.correct_cues)+" incorrect":"") : "";
  $("recipe").reset();
  $("choices").querySelectorAll("input").forEach(input=>{input.disabled=false;});
  $("confirm").disabled=true; $("confirm").textContent="Confirm recipe";
  started=performance.now();
}
function setupChoices() {
  for(const j of data.order){
    const group=document.createElement("div"), label=document.createElement("h2");
    group.className="feature"; label.id="group-"+j;
    label.textContent=data.names[j][0].toUpperCase()+data.names[j].slice(1);
    group.setAttribute("role","radiogroup");group.setAttribute("aria-labelledby",label.id);
    group.append(label);
    data.values[j].forEach((value,k)=>{
      const card=document.createElement("label"), img=new Image(), input=document.createElement("input");
      card.className="option"; input.type="radio"; input.name="factor"+j; input.value=String(k);
      img.alt=""; img.src=data.icons[j][k];
      const text=value==="oval pupils"?"Oval + vertical pupil":value[0].toUpperCase()+value.slice(1);
      card.append(input,img,document.createTextNode(text));group.append(card);
    });
    $("choices").append(group);
  }
}
function dataBlob(uri) {
  const bytes=Uint8Array.from(atob(uri.split(",")[1]),c=>c.charCodeAt(0));
  return new Blob([bytes],{type:"image/png"});
}
async function writeFile(dir,name,blob) {
  const file=await dir.getFileHandle(name,{create:true});
  const stream=await file.createWritable();
  try {await stream.write(blob); await stream.close();}
  catch(e){await stream.abort().catch(()=>{});throw e;}
}
async function newFolder(parent,name) {
  const create=async()=>{
    try {await parent.getDirectoryHandle(name);}
    catch(e){if(e.name==="NotFoundError") return parent.getDirectoryHandle(name,{create:true});throw e;}
    throw new Error("This session folder already exists. Choose a new session number or destination.");
  };
  return navigator.locks ? navigator.locks.request("face-study-"+name,create) : create();
}
async function start(e) {
  e.preventDefault(); if(busy)return;
  const subject=$("subject").value.trim(), number=Number($("number").value);
  const cued=$("hints-on").checked, low=Number($("min").value), high=Number($("max").value);
  if(!/^[\p{L}\p{N}_ -]{1,60}$/u.test(subject) || !/[\p{L}\p{N}]/u.test(subject)){
    report(new Error("Use letters, numbers, spaces, underscores or hyphens for the participant ID."));return;
  }
  if(cued && low>high){report(new Error("Minimum hints must be at most the maximum."));return;}
  const fallback=$("zip-mode").checked;
  if(!fallback && !parentDir){report(new Error("Choose a local folder or select the ZIP option."));return;}
  busy=true; $("start").disabled=true; $("error").textContent="";
  try{
    const seed=$("seed").value==="" ? crypto.getRandomValues(new Uint32Array(1))[0] : Number($("seed").value);
    const cfg={subject,session:number,seed,cued,renderer_version:data.version,protocol:"browser-v1",
      hint_range:cued?[low,high]:[0,0],hint_accuracy:$("chance").value===""?null:Number($("chance").value),
      show_hint_counts:cued && $("disclose").checked};
    folderName=subject+"_session_"+String(number).padStart(3,"0");
    if(!fallback && (startAttempt?.parent!==parentDir || startAttempt?.name!==folderName)){
      const dir=await newFolder(parentDir,folderName);
      startAttempt={parent:parentDir,name:folderName,dir};
    }
    sessionDir=fallback?null:startAttempt.dir;
    cfg.trials=makePlan(cfg);
    files.set("session.json",new Blob([JSON.stringify(cfg,null,2)],{type:"application/json"}));
    for(const [name,uri] of Object.entries(data.slides))files.set(name,dataBlob(uri));
    if(sessionDir)for(const [name,blob] of files)await writeFile(sessionDir,name,blob);
    session=cfg;showGuide();
  } catch(err){report(err);} finally{busy=false;$("start").disabled=false;}
}
async function confirmTrial(e) {
  e?.preventDefault(); if(busy)return;
  const t=session.trials[rows.length];
  busy=true; $("confirm").disabled=true;
  try{
    if(!pending){
      const form=new FormData($("recipe"));
      if(keys.some((_,j)=>!form.has("factor"+j)))return;
      const choice=keys.map((_,j)=>Number(form.get("factor"+j)));
      const row=score(t,choice,currentEvidence,(performance.now()-started)/1000);
      makeFeedback(t,choice,row);
      const updates=new Map([[row.feedback_png,await png($("comparison"))]]);
      if(t.trial===50){makeSummary([...rows,row]);updates.set("session_accuracy.png",await png($("summary")));}
      updates.set("responses.csv",new Blob([csv([...rows,row])],{type:"text/csv;charset=utf-8"}));
      pending={row,updates};
      $("choices").querySelectorAll("input").forEach(input=>{input.disabled=true;});
    }
    if(sessionDir){
      if(sessionDir.requestPermission && await sessionDir.requestPermission({mode:"readwrite"})!=="granted")
        throw new Error("Folder write permission was denied.");
      for(const [name,blob] of pending.updates) await writeFile(sessionDir,name,blob);
    }
    for(const [name,blob] of pending.updates)files.set(name,blob);
    rows.push(pending.row);pending=null;
    show("feedback","Trial "+t.trial+" of 50");
    $("next").textContent=t.trial===50?"Finish session":"Next trial";
  }catch(err){
    report(new Error("Could not save this response. "+err.message+" Retry saving, or download your session and continue in ZIP mode."));
    $("confirm").textContent="Retry saving";$("confirm").disabled=false;
    $("save-actions").hidden=false;
  }finally{busy=false;}
}
function finish() {
  finished=true;show("complete","Session complete");
  const mean=rows.reduce((sum,r)=>sum+r.score,0)/rows.length;
  $("result").textContent="50 trials completed. Average score: "+mean.toFixed(2)+" of 6 features. "+
    (sessionDir?"Files saved in "+parentDir.name+" / "+folderName+".":"Download your ZIP to keep the CSV and feedback images.");
}

// Uncompressed ZIP records. PNG files are already compressed.
const crcTable=Uint32Array.from({length:256},(_,v)=>{
  for(let i=0;i<8;i++)v=(v>>>1)^((v&1)?0xedb88320:0);
  return v>>>0;
});
async function sessionZip(entries) {
  const parts=[],central=[];
  let offset=0, centralSize=0;
  const header=n=>new DataView(new ArrayBuffer(n));
  for(const [filename,blob] of entries){
    const name=new TextEncoder().encode(folderName+"/"+filename), bytes=new Uint8Array(await blob.arrayBuffer());
    let crc=0xffffffff; for(const byte of bytes)crc=(crc>>>8)^crcTable[(crc^byte)&255]; crc=(crc^0xffffffff)>>>0;
    const local=header(30), dir=header(46);
    local.setUint32(0,0x04034b50,true);local.setUint16(4,20,true);
    dir.setUint32(0,0x02014b50,true);dir.setUint16(4,20,true);dir.setUint16(6,20,true);
    for(const [record,base] of [[local,6],[dir,8]]){
      record.setUint16(base,0x800,true);record.setUint16(base+6,33,true);
      record.setUint32(base+8,crc,true);record.setUint32(base+12,bytes.length,true);
      record.setUint32(base+16,bytes.length,true);record.setUint16(base+20,name.length,true);
    }
    dir.setUint32(42,offset,true);
    parts.push(local.buffer,name,bytes);central.push(dir.buffer,name);
    offset+=30+name.length+bytes.length;centralSize+=46+name.length;
  }
  const end=header(22);end.setUint32(0,0x06054b50,true);end.setUint16(8,entries.size,true);
  end.setUint16(10,entries.size,true);end.setUint32(12,centralSize,true);end.setUint32(16,offset,true);
  return new Blob([...parts,...central,end.buffer],{type:"application/zip"});
}
async function download(){
  if(busy)return;
  if(pending){sessionDir=null;await confirmTrial();}
  try{
    const url=URL.createObjectURL(await sessionZip(files)), a=document.createElement("a");
    a.href=url;a.download=folderName+".zip";document.body.append(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),60000);
    if(finished)window.removeEventListener("beforeunload",guard);
  }catch(e){report(e);}
}
function guard(e){if(session && (!finished || !sessionDir)){e.preventDefault();e.returnValue="";}}
window.addEventListener("beforeunload",guard);
$("pick").onclick=async()=>{
  try{
    parentDir=await window.showDirectoryPicker({mode:"readwrite",id:"face-study"});
    $("zip-mode").checked=false;
    $("destination").textContent="Selected folder: "+parentDir.name+". A new session subfolder will be created.";
    $("error").textContent="";
  }catch(e){if(e.name!=="AbortError")report(e);}
};
$("zip-mode").onchange=()=>{
  $("destination").textContent=$("zip-mode").checked?"Download your ZIP from a feedback slide or the final screen. Keep this tab open until you save it.":
    (parentDir?"Selected folder: "+parentDir.name:"Choose a folder for the trial images and CSV.");
};
if(typeof window.showDirectoryPicker!=="function"){
  $("pick").disabled=true;$("zip-mode").checked=true;$("zip-mode").disabled=true;
  $("destination").textContent="This browser can download a ZIP. For direct folder saving, open the study in desktop Chrome or Edge.";
}
$("hints-on").onchange=()=>{for(const id of ["min","max","chance","disclose"])$(id).disabled=!$("hints-on").checked;};
$("settings").onsubmit=start;
$("recipe").onchange=()=>{$("confirm").disabled=$("choices").querySelectorAll("input:checked").length!==6;};
$("recipe").onsubmit=confirmTrial;
$("next").onclick=()=>rows.length===50?finish():showTrial();
$("download").onclick=download;
data.prototypes.forEach((id,i)=>{const c=faceCanvas(id);c.setAttribute("aria-label",data.categories[i]);$("sample-faces").append(c);});
setupChoices();
