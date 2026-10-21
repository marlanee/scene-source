"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const safeUrl = (s) => {
  try { const u = new URL(s, location.origin); return ["https:","http:"].includes(u.protocol) && (!s.startsWith("/") || s.startsWith("/api/pdf/")) ? u.href : "#"; } catch { return "#"; }
};
const titles = {chat:"작품과 대화",discover:"영화 탐색",compare:"검색 실험실",evaluation:"답변 비교·평가",library:"자료실"};
const state = {movies:[], movie:null, view:"chat", task:"qa", status:{}, busy:false, history:{}, comparisons:null, runs:[], currentSources:[]};
try { state.history = JSON.parse(localStorage.getItem("scene-history-v2") || "{}"); } catch { state.history = {}; }
const key = () => state.movie.movie_id + ":" + $("response-mode").value;
const persist = () => {
  try { localStorage.setItem("scene-history-v2", JSON.stringify(state.history)); }
  catch { /* A full browser store should not break a completed answer. */ }
};
async function api(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(()=>({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "입력과 검색 설정을 확인해 주세요.");
  }
  return response.json();
}
function failure(error) {
  $("fatal").textContent = error.message || "요청을 처리하지 못했습니다.";
  $("fatal").hidden = false;
  $("fatal").scrollIntoView({behavior:"smooth", block:"center"});
}
function noticeOff() { $("fatal").hidden = true; }
function view(name) {
  state.view = name;
  document.querySelectorAll(".view").forEach(el => el.hidden = el.id !== "view-" + name);
  document.querySelectorAll(".nav").forEach(el => {
    const active = el.dataset.view === name;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current","page"); else el.removeAttribute("aria-current");
  });
  $("breadcrumb").innerHTML = "WORKSPACE <b>/ " + esc(titles[name]) + "</b>";
  if (name === "evaluation") loadEvaluation().catch(failure);
  if (name === "library") loadLibrary().catch(failure);
  noticeOff();
}
function movieOptions(query="") {
  const q = query.trim().toLowerCase();
  const filtered = state.movies.filter(m => [m.title,m.original_title,...(m.director||[])].join(" ").toLowerCase().includes(q));
  const options = filtered.length ? filtered : [];
  $("movie-select").innerHTML = options.map(m => '<option value="'+esc(m.movie_id)+'">'+esc(m.title)+" ("+esc(m.year)+")</option>").join("");
  if (!options.length) { $("movie-select").innerHTML = '<option value="">검색 결과 없음</option>'; return; }
  if (options.some(m=>String(m.movie_id) === String(state.movie?.movie_id))) $("movie-select").value = state.movie.movie_id;
  else if (state.movie) $("movie-select").insertAdjacentHTML("afterbegin", '<option value="" selected>검색 결과 '+options.length+'편 · 작품을 선택하세요</option>');
}
function setMovie(id) {
  if (state.busy) return;
  const movie = state.movies.find(m=>String(m.movie_id) === String(id));
  if (!movie) return;
  state.movie = movie;
  $("movie-select").value = movie.movie_id;
  $("film-title").textContent = movie.title;
  $("film-original").textContent = movie.original_title || "";
  $("film-meta").textContent = [movie.year,(movie.genres||[]).join(" · "),movie.runtime ? movie.runtime+"분" : "상영시간 미상"].filter(Boolean).join("   /   ");
  $("film-overview").textContent = movie.overview || "소개 정보가 없습니다.";
  $("film-credits").textContent = "감독  "+((movie.director||[]).join(", ")||"정보 없음")+"     ·     출연  "+(movie.cast||[]).slice(0,3).join(", ");
  for (const id of ["compare-movie","library-movie"]) $(id).innerHTML = "선택한 작품 <strong>"+esc(movie.title)+"</strong> · "+esc(movie.year)+' <button class="quiet" data-switch-chat>작품 변경 ↗</button>';
  $("compare-results").innerHTML = "";
  state.comparisons = null;
  renderChat();
  if (state.view === "library") loadLibrary().catch(failure);
}
function formatAnswer(text) {
  return esc(text).replace(/\*\*([^*\n]+)\*\*/g,"<strong>$1</strong>").replace(/\[근거\s*([\d,\s]+)\]/g,'<span class="citation">[근거 $1]</span>');
}
function sourceHTML(e, number, full=false) {
  return '<article class="source"><div class="source-tag"><span>근거 '+number+'</span> / '+esc(e.source_type)+(e.evidence_scope==="film"?" · 작품 소개":" · 배경 자료")+(e.page_number ? " · p."+e.page_number : "")+'</div><a class="source-title" href="'+esc(safeUrl(e.url))+'" target="_blank" rel="noreferrer">'+esc(e.title)+' ↗</a>'+(e.publisher ? '<small>'+esc(e.publisher)+'</small>' : "")+'<details'+(full?" open":"")+'><summary>원문 발췌 보기'+(Number.isFinite(e.score) && e.source_type !== "영화 메타데이터" ? " · 유사도 "+e.score.toFixed(3) : "")+'</summary><p>'+esc(e.text)+'</p></details>'+(e.collected_at ? '<small>수집 '+esc(e.collected_at.slice(0,10))+'</small>' : "")+"</article>";
}
function renderSources(sources) {
  state.currentSources = sources;
  $("source-list").innerHTML = sources.length ? sources.map((e,i)=>sourceHTML(e,e.number||i+1)).join("") : '<div class="empty"><span>▱</span><h3>근거는 여기에 모입니다</h3><p>질문하면 영화 정보와 외부 자료를<br>함께 확인할 수 있어요.</p></div>';
}
function renderChat() {
  const history = state.history[key()] || [];
  if (!history.length) {
    const prompts = state.movie.title === "인터스텔라" ? ["블랙홀의 사건의 지평선은 무엇인가요?","웜홀은 실제로 관측되었나요?"] : [state.movie.title+"의 배경을 설명해 주세요.", "영화에 등장하는 실존 인물이나 사건은 무엇인가요?"];
    $("chat-log").innerHTML = '<div class="welcome"><h3>스크린 밖의 이야기가 궁금한가요?</h3><p>장면 속 과학, 역사, 실존 인물에 대해 물어보세요.<br>'+($("response-mode").value==="gms"?"검색한 근거를 바탕으로 GMS가 답변을 생성합니다.":"현재 로컬 모드에서는 검색한 원문을 그대로 보여드립니다.")+'</p><div class="suggestions">'+prompts.map(q=>'<button class="suggestion" data-question="'+esc(q)+'">'+esc(q)+' ↗</button>').join("")+'</div></div>';
    renderSources([]);
  } else {
    $("chat-log").innerHTML = history.map(m=> m.role === "user" ? '<div class="message user">'+esc(m.content)+'</div>' : '<article class="message"><div class="message-label">SCENE & SOURCE / '+(m.generation_mode==="gms"?"GMS ANSWER":"SOURCE EXCERPTS")+'</div><div class="answer">'+formatAnswer(m.content)+'</div><small>'+esc(m.total_seconds)+"초 · "+(m.generation_mode==="gms"?"GMS 실제 답변":"원문 발췌 · AI 판정 전")+'</small></article>').join("");
    const last = [...history].reverse().find(m=>m.role==="assistant");
    renderSources(last?.sources || []);
    if (last && !last.sources?.length) $("source-list").innerHTML = '<div class="notice">이번 답변에는 인용된 근거가 없습니다. 답변의 확인 불가 안내를 살펴보고, 자료가 부족하면 추가 원문을 연결해 주세요.</div>';
  }
}
function options(action, question="") {
  return {action, movie_id:String(state.movie.movie_id), question,
    source_mode:$("source-mode").value, mode:$("response-mode").value,
    method:$("method").value, task:state.task, top_k:Number($("top-k").value),
    fetch_k:Number($("fetch-k").value), mmr_lambda:Number($("mmr-lambda").value), hybrid:$("hybrid").checked};
}
async function work(body) {
  if (state.busy) throw new Error("진행 중인 작업이 완료된 뒤 다시 실행해 주세요.");
  if (body.top_k > body.fetch_k) throw new Error("후보 수는 외부 근거 수(Top-k) 이상이어야 합니다.");
  state.busy = true; noticeOff();
  $("progress").hidden = false;
  const locked = [...document.querySelectorAll("button,input,select,textarea")].filter(el=>!el.disabled && el.id!=="connection-help" && el.id!=="close-help");
  locked.forEach(el=>el.disabled=true);
  const start = Date.now();
  const timer = setInterval(()=>{ $("elapsed").textContent = Math.floor((Date.now()-start)/1000)+"초"; },1000);
  try {
    $("progress-text").textContent = "작업 요청 중";
    const {id} = await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    while (true) {
      const job = await api("/api/jobs/"+id);
      $("progress-text").textContent = job.message;
      if (job.status === "completed") return job.result;
      if (job.status === "failed") throw new Error(job.message);
      await new Promise(r=>setTimeout(r,700));
    }
  } finally {
    clearInterval(timer); $("elapsed").textContent=""; $("progress").hidden = true; state.busy=false;
    locked.forEach(el=>el.disabled=false);
    updateGate();
  }
}
async function ask(question) {
  const savedKey = key();
  const previous = state.history[savedKey] || [];
  const result = await work({...options("chat",question), history:previous.slice(-10).map(m=>({role:m.role,content:m.content.slice(0,8000)}))});
  state.history[savedKey] = [...previous,{role:"user",content:question},{role:"assistant",content:result.answer,generation_mode:result.generation_mode,sources:result.used_sources,total_seconds:result.total_seconds}].slice(-30);
  persist(); renderChat(); $("question").value="";
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}
function download(name, content, type="application/json") {
  const url = URL.createObjectURL(new Blob([content],{type}));
  const a = document.createElement("a"); a.href=url; a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderComparison(result) {
  state.comparisons = result;
  const metrics = (m)=>'<div class="metrics"><div class="metric"><strong>'+m.unique_documents+'</strong><span>고유 문서</span></div><div class="metric"><strong>'+m.source_kinds+'</strong><span>출처 종류</span></div><div class="metric"><strong>'+(m.mean_pairwise_similarity??"—")+'</strong><span>평균 문서 간 유사도</span></div></div><p class="explain">질문 유사도 평균 '+(m.mean_query_similarity??"—")+" · 외부 청크 "+m.external_chunks+"개 · "+m.seconds+'초</p>';
  $("compare-results").innerHTML = '<div class="export-row"><span>공통 외부 청크 '+result.shared_chunk_ids.length+'개</span><button id="export-comparison" class="quiet">실험 결과 JSON 저장 ↓</button></div><div class="comparison-grid">'+["similarity","mmr"].map(method=>'<section class="panel"><span class="section-no">'+(method==="mmr"?"DIVERSITY-AWARE":"RELEVANCE-FIRST")+'</span><h2>'+(method==="mmr"?"MMR 검색":"유사도 검색")+'</h2>'+metrics(result.methods[method].metrics)+result.methods[method].evidence.map((e,i)=>sourceHTML(e,i+1)).join("")+'</section>').join("")+'</div><p class="explain">'+esc(result.note)+' 평균 문서 간 유사도가 낮을수록 내용이 서로 다릅니다. 이는 정답률 점수가 아닙니다.</p>';
}
async function loadLibrary() {
  const {sources} = await api("/api/library?movie_id="+state.movie.movie_id);
  $("library-sources").innerHTML = '<section class="panel"><h2>이 영화의 PDF 자료 · '+sources.length+'</h2>'+(sources.length ? sources.map(s=>'<article class="library-item"><div class="source-tag">'+esc(s.source)+' · 본문 '+s.pages+'쪽 / 전체 '+(s.total_pages||s.pages)+'쪽</div><h3>'+esc(s.title)+'</h3><p>'+esc(s.publisher)+" · "+esc(s.filename)+'</p><a class="text-link" href="'+esc(safeUrl(s.url))+'" target="_blank" rel="noreferrer">원문 열기 ↗</a>'+ (state.status.admin_features===false?'':' <button class="quiet" data-delete-source="'+s.id+'">자료 삭제</button>')+'</article>').join(""):'<p class="muted">아직 연결한 PDF가 없습니다. 위키백과 자료는 질문할 때 수집·검색합니다.</p>')+'</section>';
  $("import-nasa").textContent = sources.some(s=>s.catalog_id==="nasa-black-holes")?"공식 PDF 연결됨 · 다시 확인":"인터스텔라에 공식 PDF 연결";
}
function updateGate() {
  const ready = state.status.gms_available;
  const admin = state.status.admin_features !== false;
  $("run-evaluation").disabled = !ready || state.busy || !admin;
  $("import-nasa").disabled = !admin || state.busy;
  $("upload-form").hidden = !admin;
  $("evaluation-form").hidden = !admin;
  if (!admin) { $("source-mode").value="live"; $("source-mode").disabled=true; }
  $("response-mode").querySelector('[value="gms"]').disabled = !ready;
}
function scoreField(key,label,saved) {
  return '<label>'+label+'<select name="'+key+'" required><option value="">미평가</option>'+[0,1,2].map(n=>'<option value="'+n+'"'+(saved?.[key]===n?" selected":"")+'>'+n+"점</option>").join("")+'</select></label>';
}
function runNotice(run) {
  const notices=[run.protocol?.public_copy_notice,run.protocol?.limitation];
  return notices.filter(Boolean).map(n=>'<p class="notice">'+esc(n)+'</p>').join("");
}
function renderRuns(runs) {
  state.runs = runs;
  $("evaluation-runs").innerHTML = runs.length ? runs.map(run=>{
    const reviewed = run.cases.filter(c=>c.status==="completed" && c.review);
    const average = (key)=>reviewed.length ? (reviewed.reduce((s,c)=>s+c.review[key],0)/reviewed.length).toFixed(2) : "미평가";
    return '<section class="panel run"><span class="section-no">ACTUAL RUN / '+esc(run.created_at.slice(0,19))+'</span><h2>'+esc(run.model)+' · '+run.cases.filter(c=>c.status==="completed").length+"/"+run.cases.length+'개 완료</h2>'+runNotice(run)+'<div class="summary-score">사람 평가 '+reviewed.length+'개 · 정확성 평균: 미사용 '+average("baseline_accuracy")+" / RAG "+average("rag_accuracy")+' (2점 만점)</div><div class="export-row">'+["md","json","csv"].map(f=>'<a class="text-link" href="/api/evaluation/'+run.id+'/export?format='+f+'">'+f.toUpperCase()+' 저장 ↓</a>').join("")+(state.status.admin_features!==false && run.cases.some(c=>c.status!=="completed")?'<button class="quiet" data-resume="'+run.id+'"'+(!state.status.gms_available?" disabled":"")+'>실패·미완료 재개 ↻</button>':"")+'</div>'+run.cases.map(row=>'<details class="run-case"><summary>'+esc(row.id)+" · "+esc(row.title)+" — "+esc(row.question)+" · "+(row.status==="completed"?"실행 완료":esc(row.status))+'</summary>'+(row.error?'<p class="notice error">'+esc(row.error)+'</p>':"")+'<div class="comparison-grid">'+["without_rag","with_rag"].map(arm=>'<div><h3>'+(arm==="without_rag"?"RAG 미사용":"RAG 사용")+'</h3><div class="answer">'+formatAnswer(row[arm]?.answer||"실행 결과 없음")+'</div><p class="muted">생성 시간 '+(row[arm]?.generate_seconds??"—")+'초</p></div>').join("")+'</div><details><summary class="text-link">검색한 근거 원문</summary>'+(row.evidence||[]).map((e,i)=>sourceHTML(e,i+1)).join("")+'</details>'+(row.status==="completed" && state.status.admin_features!==false?'<form class="review-form" data-run="'+run.id+'" data-case="'+row.id+'">'+scoreField("baseline_accuracy","미사용 · 정확성",row.review)+scoreField("rag_accuracy","RAG · 정확성",row.review)+scoreField("support","RAG · 근거 지지",row.review)+scoreField("baseline_abstention","미사용 · 불확실성",row.review)+scoreField("rag_abstention","RAG · 불확실성",row.review)+'<label class="sr-only" for="notes-'+run.id+"-"+row.id+'">원문 대조 평가 메모</label><textarea id="notes-'+run.id+"-"+row.id+'" name="notes" maxlength="2000" placeholder="원문과 대조한 평가 근거를 적어주세요.">'+esc(row.review?.notes||"")+'</textarea><button class="primary">평가 저장</button></form>':"")+'</details>').join("")+'</section>';
  }).join(""):'<div class="empty"><span>⇄</span><h3>아직 실제 평가 기록이 없습니다</h3><p>키 연결 후 실행한 답변만 기록합니다.</p></div>';
}
async function loadEvaluation() {
  const data = await api("/api/evaluation");
  state.status.gms_available = data.gms_available;
  $("evaluation-gate").innerHTML = '<div class="notice">'+(state.status.public_demo?"공개 체험에서는 실제 비교 기록을 열람할 수 있습니다.":data.gms_available?"GMS 연결됨 · 선택한 질문을 실제 모델로 실행합니다.":"GMS 연결 대기 · 질문·평가 기준은 준비되어 있습니다. 키 연결 후 실제 비교를 실행할 수 있습니다.")+'</div>';
  $("evaluation-cases").innerHTML = data.cases.map(c=>'<label class="check-row"><input type="checkbox" name="case" value="'+c.id+'" checked><span><b>'+c.id+'</b>'+esc(c.title)+'<small>'+esc(c.question)+'</small></span></label>').join("");
  renderRuns(data.runs); updateGate();
}
document.addEventListener("click", async (event)=>{
  const el = event.target.closest("button");
  if (!el) return;
  try {
    if (el.dataset.view) view(el.dataset.view);
    if (el.hasAttribute("data-switch-chat")) { view("chat"); $("movie-search").focus(); }
    if (el.dataset.task) {
      state.task=el.dataset.task;
      document.querySelectorAll("[data-task]").forEach(b=>b.classList.toggle("selected", b.dataset.task===state.task));
    }
    if (el.dataset.question) { $("question").value=el.dataset.question; $("question").focus(); }
    if (el.dataset.movie) { $("movie-search").value=""; movieOptions(); setMovie(el.dataset.movie); view("chat"); window.scrollTo(0,0); }
    if (el.id==="export-comparison") download("retrieval_comparison.json",JSON.stringify(state.comparisons,null,2));
    if (el.dataset.deleteSource) {
      await api("/api/pdf/"+el.dataset.deleteSource,{method:"DELETE"}); await loadLibrary();
    }
    if (el.dataset.resume) {
      const run = state.runs.find(r=>r.id===el.dataset.resume);
      const r = run.protocol.retrieval;
      await work({...options("evaluate"),resume_id:run.id,mode:"gms",source_mode:r.source_mode,top_k:r.top_k,fetch_k:r.fetch_k,mmr_lambda:r.lambda,hybrid:r.hybrid??false});
      await loadEvaluation();
    }
  } catch(error) { failure(error); }
});
$("movie-search").addEventListener("input",e=>movieOptions(e.target.value));
$("movie-select").addEventListener("change",e=>setMovie(e.target.value));
$("response-mode").addEventListener("change",renderChat);
$("hybrid").addEventListener("change",e=>$("compare-hybrid").checked=e.target.checked);
$("compare-hybrid").addEventListener("change",e=>$("hybrid").checked=e.target.checked);
$("mmr-lambda").addEventListener("input",e=>$("lambda-value").textContent=Number(e.target.value).toFixed(2));
$("connection-help").onclick=()=>$("help").showModal();
$("close-help").onclick=()=>$("help").close();
$("clear-history").onclick=()=>{ delete state.history[key()]; persist(); renderChat(); };
$("chat-form").addEventListener("submit",async e=>{ e.preventDefault(); try { await ask($("question").value.trim()); } catch(error){ failure(error); } });
$("discover-form").addEventListener("submit",async e=>{
  e.preventDefault();
  try {
    const result = await work(options("discover",$("discover-query").value.trim()));
    const f = result.filters;
    const labels = [["min_year","최소 연도"],["max_year","최대 연도"],["min_runtime","최소 시간"],["max_runtime","최대 시간"],["genres","장르"],["language","언어"]].filter(([k])=>f[k]!=null && String(f[k])!=="").map(([k,label])=>label+": "+(Array.isArray(f[k])?f[k].join("·"):f[k]));
    $("discover-results").innerHTML = '<div class="filters">적용 조건 / '+esc(labels.join(" / ") || "의미 검색")+' · '+(result.verified?"모델 후보 검증 실행":"로컬 후보 · 주제 적합성 최종 검증 전")+'</div><div class="cards">'+result.items.map(({movie:m,reason},i)=>'<article class="film-card"><span class="number">0'+(i+1)+'</span><h2>'+esc(m.title)+'</h2><p>'+esc(m.year)+" · "+esc((m.genres||[]).join(" / "))+" · "+esc(m.runtime||"미상")+'분</p><p>'+esc(reason)+'</p><button class="quiet" data-movie="'+esc(m.movie_id)+'">이 작품과 대화하기 ↗</button></article>').join("")+'</div>'+(result.items.length?"":'<div class="empty"><h3>조건에 맞는 영화를 찾지 못했습니다</h3><p>연도나 장르 조건을 조금 넓혀 보세요.</p></div>');
  } catch(error) { failure(error); }
});
$("compare-form").addEventListener("submit",async e=>{
  e.preventDefault(); try { renderComparison(await work(options("compare",$("compare-query").value.trim()))); } catch(error){ failure(error); }
});
$("evaluation-form").addEventListener("submit",async e=>{
  e.preventDefault();
  try {
    const case_ids = [...document.querySelectorAll('input[name="case"]:checked')].map(el=>el.value);
    if (!case_ids.length) throw new Error("비교할 질문을 한 개 이상 선택해 주세요.");
    await work({...options("evaluate"),mode:"gms",case_ids});
    await loadEvaluation();
  } catch(error){ failure(error); }
});
document.addEventListener("submit",async e=>{
  if (!e.target.classList.contains("review-form")) return;
  e.preventDefault();
  try {
    const data = Object.fromEntries(new FormData(e.target));
    for (const k of ["baseline_accuracy","rag_accuracy","support","baseline_abstention","rag_abstention"]) data[k]=Number(data[k]);
    const report = await api("/api/evaluation/"+e.target.dataset.run+"/"+e.target.dataset.case+"/review",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const button = e.target.querySelector("button"); button.textContent="저장 완료 ✓";
    state.runs = state.runs.map(r=>r.id===report.id?report:r);
    // Preserve the expanded case and keyboard focus after saving.
    const reviewed=report.cases.filter(c=>c.status==="completed"&&c.review);
    const avg=k=>(reviewed.reduce((s,c)=>s+c.review[k],0)/reviewed.length).toFixed(2);
    e.target.closest(".run").querySelector(".summary-score").textContent="사람 평가 "+reviewed.length+"개 · 정확성 평균: 미사용 "+avg("baseline_accuracy")+" / RAG "+avg("rag_accuracy")+" (2점 만점)";
  } catch(error){ failure(error); }
});
$("import-nasa").onclick=async()=>{
  try {
    await work({...options("official"),movie_id:"157336"});
    if (String(state.movie.movie_id)!=="157336") { $("movie-search").value=""; movieOptions(); setMovie("157336"); }
    await loadLibrary(); state.status=await api("/api/status");
  } catch(error){ failure(error); }
};
$("upload-form").addEventListener("submit",async e=>{
  e.preventDefault();
  const button=e.target.querySelector("button");
  try {
    const file=$("pdf-file").files[0];
    if (!file || file.size>10*1024*1024) throw new Error("10MB 이하의 PDF를 선택해 주세요.");
    button.disabled=true; button.textContent="PDF 본문 확인 중";
    const params=new URLSearchParams({title:$("pdf-title").value,publisher:$("pdf-publisher").value,filename:file.name});
    await api("/api/pdf/"+state.movie.movie_id+"?"+params,{method:"POST",headers:{"Content-Type":"application/pdf"},body:file});
    e.target.reset(); await loadLibrary();
  } catch(error){ failure(error); }
  finally { button.disabled=false; button.textContent="이 영화의 자료로 저장 ↗"; }
});
async function init() {
  try {
    [state.status,state.movies]=await Promise.all([api("/api/status"),api("/api/movies")]);
    $("movie-count").textContent=state.status.movie_count;
    $("workspace-label").textContent=state.status.public_demo?"PUBLIC DEMO":"LOCAL WORKSPACE";
    $("key-status").textContent=state.status.gms_available?"GMS 연결됨":"로컬 검색 모드 · GMS 연결 대기";
    $("response-mode").value=state.status.default_mode || "local";
    $("source-mode").value=state.status.source_mode || "bundle";
    const help=$("help");
    if (state.status.public_demo) {
      $("response-mode").querySelector('[value="local"]').textContent="원문 발췌 · 검색 API 사용";
      help.querySelector("h2").textContent="공개 체험 안내";
      const paragraphs=help.querySelectorAll("p");
      paragraphs[0].textContent="검색·답변은 접속 환경별 시간당 3회까지 체험할 수 있습니다. 의미 검색과 답변 생성은 함께 하루 사용량 한도를 적용합니다. 한도 이후에는 영화 목록·비교 기록·PDF 자료를 열람할 수 있습니다.";
      paragraphs[1].textContent="공개 서버의 의미 검색을 위해 검색어·선별 문서를 GMS 임베딩 API로 전송하며, 답변 생성에는 질문·최근 대화·근거를 전송합니다. 개인 정보나 비밀은 질문에 넣지 마세요. 자료 변경과 사람 채점은 로컬 버전에서 제공합니다.";
    } else if (state.status.gms_available) {
      help.querySelector("p").textContent="GMS가 연결되어 있습니다. 검색한 근거를 바탕으로 AI 답변과 사실 확인을 사용할 수 있습니다.";
    }
    movieOptions(); setMovie("157336"); updateGate();
  } catch(error){ failure(error); }
}
init();
