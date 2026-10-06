var TW=(function(){
var D={"pts": [[187, 156.6], [202, 156.6], [217, 156.6], [232, 158.4], [247, 153.0], [262, 145.8], [277, 144.0], [292, 142.2], [307, 147.6], [322, 151.2], [337, 151.2], [352, 158.4], [367, 162.0], [382, 163.8], [397, 167.4], [412, 169.2], [427, 172.8], [442, 178.2], [457, 183.6], [472, 185.4], [487, 190.8], [502, 194.4], [517, 203.4], [532, 212.4], [547, 203.4], [562, 178.2], [577, 153.0], [592, 126.0], [607, 97.2], [622, 75.6], [637, 61.2], [652, 55.8], [667, 61.2], [682, 75.6], [697, 77.4], [712, 75.6], [727, 68.4], [742, 64.8], [757, 68.4], [772, 75.6], [787, 81.0], [802, 88.2]], "ev": [["meal", 412, "Cereal 100 g and more"], ["insulin", 412, "insulin glargine, 14 IU, Humulin 70/30, 12 IU"], ["meal", 637, "Water bamboo 60 g and more"], ["insulin", 637, "Humulin 70/30, 3 IU"]]},HEART=true,state={now:592,reviewed:false,snooze:false,report:false,log:[]};
function fmt(m){var h=Math.floor(m/60),mm=m%60;return (h<10?'0':'')+h+':'+(mm<10?'0':'')+mm}
function at(m){var b=null;for(var i=0;i<D.pts.length;i++){if(D.pts[i][0]<=m)b=D.pts[i][1]}return b}
function sinceEv(kind,now){var b=null;D.ev.forEach(function(e){if(e[0]==kind&&e[1]<=now&&(b==null||e[1]>b[1]))b=e});return b}
function forecast(now){var v=at(now),tr=(v-at(now-30))/2,out=[],s=0,half={15:8,30:15,60:25,120:37};
for(var k=1;k<=8;k++){s+=Math.pow(.8,k);var h=k*15,hw;if(h<=15)hw=half[15];else if(h<=30)hw=half[15]+(h-15)/15*(half[30]-half[15]);else if(h<=60)hw=half[30]+(h-30)/30*(half[60]-half[30]);else hw=half[60]+(h-60)/60*(half[120]-half[60]);var p=Math.max(45,v+tr*s);out.push({m:now+h,h:h,p:p,lo:p-hw,hi:p+hw})}return out}
function reasons(list){return list}
function compute(now){var v=at(now),ch=v-at(now-30),f=forecast(now),ins=sinceEv('insulin',now),meal=sinceEv('meal',now);
var alertT=HEART?100:90,watchT=alertT+20,low=f.slice(0,4).reduce(function(a,b){return b.p<a.p?b:a}),tier=low.p<=alertT?'alert':(low.p<=watchT?'watch':'calm');
var cells=f.map(function(q){return {h:q.h,p:q.p,lo:q.lo,hi:q.hi,tier:q.p<=alertT?'alert':(q.p<=watchT?'watch':(q.p>=180?'high':'calm'))}});
var rs=[];if(ch<-6)rs.push({t:'Sugar fell '+Math.round(-ch)+' mg/dL in the last 30 minutes',up:true,w:Math.min(100,-ch*3)});else if(ch>6)rs.push({t:'Sugar rose '+Math.round(ch)+' mg/dL in the last 30 minutes',up:false,w:Math.min(100,ch*3)});
if(v<140)rs.push({t:'Sugar is already down at '+Math.round(v)+' mg/dL',up:true,w:Math.min(100,(140-v)*1.2)});
if(ins&&now-ins[1]<=300)rs.push({t:'Insulin given '+(now-ins[1])+' minutes ago',up:true,w:Math.max(15,70-(now-ins[1])/5)});
if(meal&&now-meal[1]<=120)rs.push({t:'A meal was logged '+(now-meal[1])+' minutes ago, which pushes sugar up',up:false,w:Math.max(15,60-(now-meal[1])/2)});
if(!rs.length)rs.push({t:'Sugar is steady and well above the low range',up:false,w:30});
var mx=Math.max(f[3].p,f[7].p),hp=Math.round(100/(1+Math.exp(-(mx-190)/22))),ht=hp<15?'calm':(hp<40?'watch':'alert');
var head=tier=='alert'?'Possible low in about '+low.h+' minutes':(tier=='watch'?'Sugar is drifting down. Keep an eye on the next hour':'No low expected in the next hour');
var sub=tier=='calm'?'Sugar is '+Math.round(v)+' mg/dL and steady.':'Sugar is '+Math.round(v)+' mg/dL and the forecast reaches about '+Math.round(low.p)+' mg/dL.';
var checks=[];if(ins&&now-ins[1]<=360)checks.push('Last insulin: '+ins[2].split(',').slice(-2).join(',').trim()+', '+(now-ins[1])+' minutes ago');if(meal)checks.push('Last meal logged '+(now-meal[1])+' minutes ago');
checks.push('This patient had 18 lows in 2 weeks of readings');checks.push('The sensor reads about 10 mg/dL above this patient\'s finger-prick checks, so real sugar may be lower');
return {now:now,v:v,ch:ch,f:f,low:low,tier:tier,alertT:alertT,watchT:watchT,cells:cells,lowReasons:rs,highP:hp,highTier:ht,highText:'About '+hp+' in 100 chance of going above 180 mg/dL in the next 2 hours.',headline:head,headSub:sub,checks:checks,heart:HEART&&tier!='calm'}}
function stamp(){return fmt(state.now)}
function act(name,extra){var c=compute(state.now),msg='';
if(name=='note'){msg='Note saved to the clinical notes';state.log.push({t:stamp(),a:'Note added: '+(extra||'')})}
if(name=='reviewed'){state.reviewed=true;msg='Marked as reviewed. The twin re-checks in 15 minutes.';state.log.push({t:stamp(),a:'Marked as reviewed by Dr. Rao'})}
if(name=='snooze'){state.snooze=true;msg='Snoozed for 1 hour.';state.log.push({t:stamp(),a:'Alert snoozed for 1 hour'})}
if(name=='report'){state.report=!state.report;msg=state.report?'Added to the next report.':'Removed from the next report.';state.log.push({t:stamp(),a:state.report?'Added to the next report':'Removed from the next report'})}
if(name=='undo'){state.reviewed=false;state.snooze=false;msg='Back to open.';state.log.push({t:stamp(),a:'Status reset to open'})}
toast(msg);return msg}
function draft(){var c=compute(state.now);return fmt(state.now)+', 10 Nov 2020. Digital Twin flagged: '+c.headline.toLowerCase()+'. '+c.headSub+' '+(c.checks[0]||'')+'. Reviewed by Dr. Rao. Plan: '}
function logHTML(){if(!state.log.length)return '<p class="tw-foot">Nothing yet. Actions you take are recorded here, and also appear on the patient timeline and in the PDF report.</p>';return state.log.slice().reverse().map(function(e){return '<div class="tw-logrow"><b>'+e.t+'</b><span>'+e.a+'</span></div>'}).join('')+'<p class="tw-foot">Recorded on the patient timeline and in the PDF report.</p>'}
function reasonsHTML(list){return list.map(function(r){return '<div class="tw-reason"><span>'+r.t+'</span><span class="tw-bar" title="'+(r.up?'raises':'lowers')+' the risk"><i style="width:'+Math.max(8,Math.min(100,r.w))+'%;background:'+(r.up?'#D3402A':'#12876F')+'"></i></span></div>'}).join('')+'<p class="tw-foot">Red raises the risk, green lowers it. Longer bar means a bigger push.</p>'}
var tt=null;function toast(m){var el=document.getElementById('toast');if(!el)return;el.textContent=m;el.style.opacity=1;clearTimeout(tt);tt=setTimeout(function(){el.style.opacity=0},2600)}
function copy(t,b){try{navigator.clipboard.writeText(t)}catch(e){}var o=b.textContent;b.textContent='Copied';setTimeout(function(){b.textContent=o},1200)}
var timer=null,onTick=null;function play(btn,cb){onTick=cb;if(timer){clearInterval(timer);timer=null;btn.textContent='Play replay';return}btn.textContent='Pause replay';timer=setInterval(function(){state.now=state.now>=660?420:state.now+15;state.reviewed=false;state.snooze=false;onTick()},1100)}
var onChange=function(){};
function afterHTML(c){var s=state,h='';
if(s.reviewed)h+='<div class="tw-row" style="margin-bottom:6px"><span class="tw-chip reviewed">Reviewed by Dr. Rao at '+fmt(s.now)+'</span><button class="ghost small" data-act="undo">Reopen</button></div><p class="tw-sub">The twin re-checks in 15 minutes and flags this patient again if the forecast changes.</p>';
else if(s.snooze)h+='<div class="tw-row" style="margin-bottom:6px"><span class="tw-chip">Snoozed for 1 hour</span><button class="ghost small" data-act="undo">Reopen</button></div>';
else if(c.tier=='calm')h+='<p class="tw-sub">Nothing to do right now. The twin re-checks every 15 minutes.</p>';
else{h+='<div style="font-weight:600;margin-bottom:2px">Things doctors often check</div><ul class="tw-checks">'+c.checks.map(function(x){return '<li>'+x+'</li>'}).join('')+'</ul><div style="font-weight:600;margin:12px 0 4px">What would you like to do?</div><div class="tw-actions"><button class="primary" data-act="note-open">Write a note</button><button class="ghost" data-act="timeline">Look at the medicine timeline</button><button class="ghost" data-act="report">'+(s.report?'Remove from the next report':'Add to the next report')+'</button><button class="ghost" data-act="reviewed">Mark as reviewed</button><button class="ghost" data-act="snooze">Remind me in 1 hour</button></div><p class="tw-foot">The twin only informs. You decide about any change to treatment.</p>'}
return h}
function bind(){if(!document.getElementById('noteDlg')){var d=document.createElement('dialog');d.id='noteDlg';d.innerHTML='<form method="dialog"><b style="font-size:17px">Clinical note</b><p class="tw-sub">A draft is filled in from what the twin saw. Edit it, then save.</p><textarea id="noteTxt"></textarea><div class="tw-actions"><button class="primary" value="save">Save note</button><button class="ghost" value="cancel">Cancel</button></div></form>';document.body.appendChild(d);var t=document.createElement('div');t.id='toast';t.setAttribute('role','status');document.body.appendChild(t);
d.addEventListener('close',function(){if(d.returnValue=='save'){var v=document.getElementById('noteTxt').value;act('note',v.length>60?v.slice(0,60)+'...':v);onChange()}})}
document.addEventListener('click',function(e){var b=e.target.closest('[data-act]');if(!b)return;var a=b.getAttribute('data-act');
if(a=='note-open'){document.getElementById('noteTxt').value=draft();document.getElementById('noteDlg').showModal();return}
if(a=='timeline'){var el=document.getElementById('timeline');if(el){el.scrollIntoView({behavior:'smooth',block:'center'});toast('This is the medicine timeline.')}else toast('Opens the medicine timeline for this patient.');state.log.push({t:stamp(),a:'Opened the medicine timeline'});onChange();return}
act(a);onChange()})}
return {afterHTML:afterHTML,bind:bind,setOnChange:function(f){onChange=f},D:D,state:state,fmt:fmt,at:at,compute:compute,act:act,draft:draft,logHTML:logHTML,reasonsHTML:reasonsHTML,toast:toast,copy:copy,play:play,forecast:forecast}
})();
