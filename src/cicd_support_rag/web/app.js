/* PipelineHub chat. All user/model/source content is rendered as text nodes. */
const $ = id => document.getElementById(id);
const storeKey = 'pipelinehub-chat-v1';
const pendingPolls = new Set();
const stageLabels = {queued:'Waiting to prepare your answer…',retrieving:'Finding relevant documentation…',drafting:'Preparing your answer…',checking:'Checking the answer against its sources…',finalizing:'Finishing up…'};
let chats = [], activeId, toastTimer, storageWarning = false;

function node(tag, className, text) { const n = document.createElement(tag); if(className) n.className=className; if(text!==undefined) n.textContent=text; return n; }
function icon(name) { const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'); const use=document.createElementNS('http://www.w3.org/2000/svg','use'); use.setAttribute('href','#i-'+name); svg.append(use); svg.setAttribute('aria-hidden','true'); return svg; }
function iconButton(name,label,callback) { const b=node('button','icon-button'); b.append(icon(name)); b.type='button'; b.title=label; b.setAttribute('aria-label',label); b.addEventListener('click',callback); return b; }
function toast(text) { $('toast').textContent=text; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,4000); }
function save() { try { localStorage.setItem(storeKey,JSON.stringify({activeId,chats})); } catch { if(!storageWarning){storageWarning=true;toast('Browser storage is full. Export this chat to keep a copy.');} } }
function activeChat() { return chats.find(c=>c.id===activeId); }
function newChat() { const empty=chats.find(c=>!c.messages.length); if(empty) activeId=empty.id; else {const chat={id:crypto.randomUUID(),title:'New chat',created:Date.now(),messages:[]};chats.unshift(chat);activeId=chat.id;} save(); closeMobileSidebar(); render(); $('question').focus(); }
function scrollDown() { requestAnimationFrame(()=>$('chatScroll').scrollTop=$('chatScroll').scrollHeight); }

function renderHistory() {
  const host=$('chatHistory');host.replaceChildren();
  const populated=chats.filter(c=>c.messages.length);
  if(!populated.length) host.append(node('p','history-empty','A fresh start. Your conversations will appear here.'));
  for(const chat of populated) {
    const row=node('div','history-item'+(chat.id===activeId?' active':''));
    const select=node('button','history-select',chat.title);select.title=chat.title;select.type='button';
    if(chat.id===activeId) select.setAttribute('aria-current','page');
    select.addEventListener('click',()=>{activeId=chat.id;save();closeMobileSidebar();render();scrollDown();});
    const remove=iconButton('trash','Delete '+chat.title,()=>{
      if(!confirm('Delete this conversation from this browser?'))return;
      chats=chats.filter(c=>c.id!==chat.id);
      if(activeId===chat.id){if(chats.length)activeId=chats[0].id;else return newChat();}
      save();render();
    });remove.classList.add('history-delete');
    remove.disabled=chat.messages.some(m=>m.pending);if(remove.disabled)remove.title='Wait for this answer before deleting the chat';
    row.append(select,remove);host.append(row);
  }
}

function inline(text, result) {
  const fragment=document.createDocumentFragment();
  const pattern=/(\*\*[^*]+\*\*|`[^`]+`|\[doc-\d+\])/g;
  let offset=0;
  for(const match of text.matchAll(pattern)) {
    fragment.append(document.createTextNode(text.slice(offset,match.index)));
    const value=match[0];
    if(value.startsWith('**'))fragment.append(node('strong','',value.slice(2,-2)));
    else if(value.startsWith('`'))fragment.append(node('code','',value.slice(1,-1)));
    else { const id=value.slice(1,-1);const b=node('button','citation',id);b.type='button';b.title='View source '+id;b.addEventListener('click',()=>openSource(result,id));fragment.append(b); }
    offset=match.index+value.length;
  }
  fragment.append(document.createTextNode(text.slice(offset)));return fragment;
}

function renderAnswer(text,result) {
  const body=node('div','answer-body');let list=null, inCode=false, codeLines=[];
  // The source list is presented as inspectable source chips below the answer.
  const mainText=text.split('\nSources:')[0];
  for(const raw of mainText.split('\n')) {
    let line=raw.trim();
    if(line.startsWith('```')){if(inCode){const pre=node('pre');pre.append(node('code','',codeLines.join('\n')));body.append(pre);codeLines=[];}inCode=!inCode;list=null;continue;}
    if(inCode){codeLines.push(raw);continue;}
    if(!line){list=null;continue;}
    if(line.startsWith('- ')){if(!list){list=node('ul');body.append(list);}const item=node('li');item.append(inline(line.slice(2),result));list.append(item);}
    else {list=null;const quote=line.startsWith('>');if(quote)line=line.slice(1).trim().replace(/^[*+-]\s+/,'');const p=node(quote?'blockquote':'p');p.append(inline(line,result));body.append(p);}
  }
  if(inCode){const pre=node('pre');pre.append(node('code','',codeLines.join('\n')));body.append(pre);}
  return body;
}

function sourceDocuments(result) {
  const all=result?.retrieved_sources||[];
  let ids=result?.status==='answered' ? new Set([...(result.supporting_quotes||[]),...(result.source_cautions||[])].map(s=>s.document_id)) : new Set(all.map(s=>s.document_id));
  return [...new Map(all.filter(s=>ids.has(s.document_id)).map(s=>[s.document_id,s])).values()];
}

function openSource(result,id) {
  const source=(result?.retrieved_sources||[]).find(s=>s.document_id===id);
  if(!source){toast('That source was not included in this response.');return;}
  const host=$('sourceContent');host.replaceChildren();
  const title=node('h2','',source.title);title.id='sourceTitle';host.append(title);
  host.append(node('div','source-meta',id+(source.parent_id?' · Parent: '+source.parent_id:'')));
  host.append(node('p','source-intro','Inspect the text behind this response. These are adapted documentation excerpts; links, commands, and configurations are illustrative.'));
  const quotes=[...(result.supporting_quotes||[]),...(result.source_cautions||[])].filter(q=>q.document_id===id);
  const seen=new Set();
  for(const quote of quotes){if(seen.has(quote.quote))continue;seen.add(quote.quote);const card=node('div','source-excerpt');card.append(node('strong','','SOURCE EXCERPT'));const p=node('p');p.append(inline(quote.quote,result));card.append(p);host.append(card);}
  const details=node('details','source-passages');details.open=!quotes.length;
  details.append(node('summary','','View retrieved passages'));
  for(const passage of (result.retrieved_sources||[]).filter(s=>s.document_id===id)){
    const card=node('div','source-excerpt full');card.append(node('strong','',passage.chunk_id));card.append(node('p','',passage.text));details.append(card);
  }
  host.append(details);if(!$('sourceDialog').open)$('sourceDialog').showModal();
}

function download(name, value) { const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); }
async function copyText(text) { try{await navigator.clipboard.writeText(text);toast('Answer copied.');}catch{toast('Copy is unavailable here. You can select the answer text or export the chat.');} }

function renderAssistant(message,chat) {
  const article=node('article','message assistant');
  const heading=node('div','assistant-heading');const avatar=node('span','assistant-avatar');avatar.append(icon('logo'));heading.append(avatar,node('span','assistant-name','PipelineHub'));
  const result=message.result;
  if(result){const labels={answered:'From your docs',needs_clarification:'A little more detail',needs_human_review:'Human review'};const kind=result.status==='needs_human_review'?' review':result.status==='needs_clarification'?' clarification':'';heading.append(node('span','answer-badge'+kind,labels[result.status]||'Response'));}
  article.append(heading);
  if(message.pending){
    const pending=node('div','pending-body');pending.setAttribute('role','status');const dots=node('span','loading-dots');dots.setAttribute('aria-hidden','true');for(let i=0;i<3;i++)dots.append(node('i'));pending.append(dots,node('span','',stageLabels[message.stage]||stageLabels.queued));article.append(pending);
    const note=node('p','pending-note','A local answer can take about a minute. The answer will appear after its source checks.');article.append(note);return article;
  }
  if(message.error){const error=node('div','error-body',message.error);const retry=node('button','retry-button','Try again');retry.addEventListener('click',()=>retryMessage(chat,message));error.append(retry);article.append(error);return article;}
  if(!result)return article;
  article.append(renderAnswer(result.answer,result));
  if(result.status==='answered')article.append(node('p','answer-note','Documentation links, commands, and configurations are illustrative.'));
  const docs=sourceDocuments(result);
  if(docs.length){const sources=node('div','sources-bar');sources.append(node('div','sources-label',(result.status==='answered'?'Sources':'Retrieved documents')+' · '+docs.length));const chips=node('div','source-chips');for(const source of docs){const chip=node('button','source-chip');chip.append(icon('file'),node('span','',source.title));chip.title=source.document_id+' · '+source.title;chip.addEventListener('click',()=>openSource(result,source.document_id));chips.append(chip);}sources.append(chips);article.append(sources);}
  const actions=node('div','message-actions');actions.append(iconButton('copy','Copy answer',()=>copyText(result.answer)));
  if(result.handoff){const b=node('button','action-link','Save review packet');b.prepend(icon('download'));b.addEventListener('click',()=>download('pipelinehub-human-review.json',result.handoff));actions.append(b);}
  if(result.status==='needs_human_review'){
    const reason=node('div','reason-detail',result.evidence_reason);reason.hidden=true;
    const b=node('button','action-link','Why this response?');b.setAttribute('aria-expanded','false');b.addEventListener('click',()=>{reason.hidden=!reason.hidden;b.setAttribute('aria-expanded',String(!reason.hidden));});actions.append(b);article.append(reason);
  }
  if(Number.isFinite(result.elapsed_seconds))actions.append(node('span','message-timing',Math.round(result.elapsed_seconds)+'s'));
  article.append(actions);return article;
}

function updateComposer() {
  const busy=activeChat()?.messages.some(m=>m.pending);
  $('sendButton').disabled=!$('question').value.trim()||busy;
  $('question').placeholder=busy?'Preparing your answer…':'Ask anything about PipelineHub…';
  $('question').style.height='auto';$('question').style.height=Math.min($('question').scrollHeight,160)+'px';
}
function render() {
  renderHistory();const chat=activeChat();if(!chat)return;
  const empty=!chat.messages.length;$('welcome').hidden=!empty;$('messages').hidden=empty;
  $('messages').replaceChildren();
  for(const message of chat.messages){if(message.role==='user')$('messages').append(node('div','message user',message.content));else $('messages').append(renderAssistant(message,chat));}
  $('exportChat').disabled=empty;updateComposer();
}

function contextFor(chat,message) {
  const index=chat.messages.indexOf(message);
  return chat.messages.slice(0,Math.max(0,index-1)).filter(m=>!m.pending&&!m.error).slice(-6).map(m=>({role:m.role,content:(m.role==='user'?m.content:m.result?.answer||'').slice(0,4000),status:m.result?.status}));
}

async function requestAnswer(chat,message,question) {
  try {
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-PipelineHub-Client':'web'},body:JSON.stringify({question,history:contextFor(chat,message)})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'The request could not be started.');
    message.jobId=data.job_id;save();poll(chat,message);
  }catch(error){message.pending=false;message.error=error.message||'The local chat server is unavailable.';save();if(chat.id===activeId){render();scrollDown();}}
}

async function poll(chat,message) {
  if(pendingPolls.has(message.jobId))return;pendingPolls.add(message.jobId);
  const started=Date.now();
  try{
    while(message.pending){
      const response=await fetch('/api/jobs/'+encodeURIComponent(message.jobId));const data=await response.json();if(!response.ok)throw new Error(data.error||'This request is unavailable. Please try again.');
      if(data.status==='done'){message.pending=false;message.result=data.result;message.stage='done';save();renderHistory();if(chat.id===activeId){render();scrollDown();}break;}
      if(data.status==='error')throw new Error(data.error);
      if(message.stage!==data.stage){message.stage=data.stage;save();if(chat.id===activeId){render();scrollDown();}}
      if(Date.now()-started>360000)throw new Error('This request is taking longer than expected. Please try again shortly.');
      await new Promise(resolve=>setTimeout(resolve,1100));
    }
  }catch(error){message.pending=false;message.error=error.message||'The local chat server is unavailable. Try reopening the interface.';save();if(chat.id===activeId){render();scrollDown();}}
  finally{pendingPolls.delete(message.jobId);updateComposer();}
}

function sendQuestion(question) {
  question=question.trim();if(!question)return;
  const chat=activeChat();if(chat.messages.some(m=>m.pending)){toast('Please wait for this answer before sending another question.');return;}
  if(question.length>4000){toast('Please keep your question under 4,000 characters.');return;}
  if(!chat.messages.length)chat.title=question.length>48?question.slice(0,48)+'…':question;
  chat.messages.push({id:crypto.randomUUID(),role:'user',content:question});
  const assistant={id:crypto.randomUUID(),role:'assistant',pending:true,stage:'queued',started:Date.now()};chat.messages.push(assistant);
  $('question').value='';save();render();scrollDown();requestAnswer(chat,assistant,question);
}
function retryMessage(chat,message) {
  if(chat.messages.some(m=>m.pending))return;
  const index=chat.messages.indexOf(message);const user=[...chat.messages.slice(0,index)].reverse().find(m=>m.role==='user');if(!user)return;
  delete message.error;delete message.result;delete message.jobId;message.pending=true;message.stage='queued';save();render();scrollDown();requestAnswer(chat,message,user.content);
}

function closeMobileSidebar(){document.body.classList.remove('mobile-sidebar-open');$('sidebarBackdrop').hidden=true;}
function syncSidebar(){const mobile=matchMedia('(max-width:760px)').matches;$('openSidebar').hidden=!(mobile||document.body.classList.contains('sidebar-collapsed'));if(!mobile)closeMobileSidebar();}
$('closeSidebar').addEventListener('click',()=>{if(matchMedia('(max-width:760px)').matches)closeMobileSidebar();else document.body.classList.add('sidebar-collapsed');syncSidebar();});
$('openSidebar').addEventListener('click',()=>{if(matchMedia('(max-width:760px)').matches){document.body.classList.add('mobile-sidebar-open');$('sidebarBackdrop').hidden=false;}else document.body.classList.remove('sidebar-collapsed');syncSidebar();});
$('sidebarBackdrop').addEventListener('click',closeMobileSidebar);window.addEventListener('resize',syncSidebar);
$('newChat').addEventListener('click',newChat);
$('chatForm').addEventListener('submit',event=>{event.preventDefault();sendQuestion($('question').value);});
$('question').addEventListener('input',updateComposer);
$('question').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('sendButton').disabled)sendQuestion($('question').value);}});
document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>sendQuestion(button.dataset.prompt)));
$('exportChat').addEventListener('click',()=>{const chat=activeChat();download('pipelinehub-chat.json',{title:chat.title,exported_at:new Date().toISOString(),messages:chat.messages.filter(m=>!m.pending)});});
$('closeSource').addEventListener('click',()=>$('sourceDialog').close());$('aboutButton').addEventListener('click',()=>$('aboutDialog').showModal());$('closeAbout').addEventListener('click',()=>$('aboutDialog').close());
for(const dialog of [$('sourceDialog'),$('aboutDialog')])dialog.addEventListener('click',event=>{const box=dialog.getBoundingClientRect();if(event.target===dialog&&(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom))dialog.close();});
document.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();newChat();}if(event.key==='Escape')closeMobileSidebar();});

try {const saved=JSON.parse(localStorage.getItem(storeKey)||'null');if(saved&&Array.isArray(saved.chats)){chats=saved.chats.filter(c=>c&&typeof c.id==='string'&&Array.isArray(c.messages));activeId=saved.activeId;}}catch{}
if(!activeChat()){if(chats.length)activeId=chats[0].id;else newChat();}
for(const chat of chats)for(const message of chat.messages)if(message.pending){if(message.jobId)poll(chat,message);else{message.pending=false;message.error='This request was interrupted. Try sending it again.';}}
render();syncSidebar();
fetch('/api/health').then(r=>r.json()).then(health=>{if(!health.model_ready||!health.knowledge_configured){$('connectionNotice').textContent=!health.model_ready?'The answer provider is not ready. Check its connection and model settings, then reload this page.':'The knowledge base needs its Pinecone configuration. Check the project setup before sending a question.';$('connectionNotice').hidden=false;}}).catch(()=>{$('connectionNotice').textContent='The local server is unavailable. Reopen the chat launcher to reconnect.';$('connectionNotice').hidden=false;});
