const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const ts=require('typescript');
const source=ts.transpileModule(fs.readFileSync(__dirname+'/soloDemoClient.ts','utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.CommonJS}}).outputText;
const tick=()=>new Promise(r=>setImmediate(r));
const reply=(id='one')=>({ok:true,status:201,json:async()=>({contract_version:'solo.v1',mode:'solo',session_id:id,livekit_url:'wss://voice.test',token:'room-token-'+id,control_token:'control-token-'+id})});
function setup(t,{request,connectError=false,supportError=null,prepare,connect,timerDurations=[]}={}){
 const calls=[],transports=[],exports={};let count=0;
 class Transport{
  constructor(){transports.push(this);}onData(fn){this.data=fn;return()=>{};}onStatus(fn){this.status=fn;return()=>{};}
  async prepareMicrophone(){if(prepare)await prepare(this);}
  async connect(url,token){this.args={url,token};if(connectError)throw Error('media failed');if(connect)await connect(this);}
  async disconnect(){this.disconnected=true;this.cancelPreparation?.();}
 }
 vm.runInNewContext(source,{exports,require:()=>({LiveKitService:Transport}),AbortController,setTimeout:(fn,ms)=>{timerDurations.push(ms);return setTimeout(fn,ms);},clearTimeout,performance,console,globalThis:{fetch:()=>{}}});
 const client=new exports.SoloDemoClient({getSupportError:()=>supportError,createTransport:()=>new Transport(),fetch:async(url,opts)=>{calls.push({url,body:JSON.parse(opts.body)});if(request)return request(url,opts);return url.endsWith('/end')?{ok:true,status:200}:reply(String(++count));}});
 t.after(()=>client.dispose().catch(()=>{}));return{client,calls,transports};
}
function ready(t){t.data('agent_ready',{});t.status({state:'connected',error:'',audioSubscribed:true,microphoneLevel:0,agentLevel:0});}
test('one start request; readiness requires connected media plus backend ready',async t=>{
 const x=setup(t);const a=x.client.start(),b=x.client.start();assert.equal(a,b);await a;assert.equal(x.calls.length,1);assert.equal(x.calls[0].body.participant_name,'demo-visitor');assert.equal(x.transports[0].args.url,'wss://voice.test');
 x.transports[0].data('agent_ready',{});assert.equal(x.client.getSnapshot().ready,false);ready(x.transports[0]);assert.equal(x.client.getSnapshot().phase,'listening');x.transports[0].data('first_audio_queued',{});assert.equal(x.client.getSnapshot().phase,'listening');
 x.transports[0].status({state:'connected',error:'',audioSubscribed:true,microphoneLevel:.3,agentLevel:.2});assert.equal(x.client.getSnapshot().phase,'speaking');assert.equal(x.client.getSnapshot().levels.patient,.3);
});
test('record revisions, generated captions and per-turn metrics preserve their meaning',async t=>{
 const x=setup(t);await x.client.start();const s=x.transports[0];ready(s);
 s.data('assistant_text',{context_id:'p1',text:'hello',is_final:true});assert.equal(x.client.getSnapshot().messages[0].source,'generated');s.data('tts_interrupted',{context_id:'p1'});assert.equal(x.client.getSnapshot().messages[0].interrupted,true);
 const record=revision=>({event:'advance_decision',record_source:'solo_async_extraction',capture_revision:revision,collected_fields:{score:{value:7,status:'confirmed',evidence:'七分',source_node:'symptom'}},policy_missing:['consent']});s.data('dual_llm_debug',record(2));s.data('dual_llm_debug',record(1));assert.equal(x.client.getSnapshot().record.revision,2);assert.equal(x.client.getSnapshot().record.fields.score.status,'captured');
 s.data('turn_metrics',{turn_id:'t1',speech_end_to_audio_ms:900,output_queue_wait_ms:1,llm_ttft_ms:300,tts_ttfa_ms:200});assert.equal(x.client.getSnapshot().metrics.speechEndToQueuedMs,900);
 s.data('turn_metrics',{turn_id:'t2',speech_end_to_audio_ms:800,output_queue_wait_ms:-1,llm_ttft_ms:250,tts_ttfa_ms:-1});assert.equal(x.client.getSnapshot().metrics.speechEndToQueuedMs,null);assert.equal(x.client.getSnapshot().metrics.ttsTtfaMs,null);
});
test('stop during HTTP startup closes the late server session without connecting media',async t=>{
 let deliver;const response=new Promise(r=>deliver=r);const x=setup(t,{request:async url=>url.endsWith('/end')?{ok:true,status:200}:response});
 const starting=x.client.start();await tick();const stopped=x.client.stop();deliver(reply('late'));await Promise.all([starting,stopped]);assert.equal(x.transports.length,1);assert.equal(x.transports[0].args,undefined);assert.equal(x.transports[0].disconnected,true);assert.equal(x.calls.filter(c=>c.url.endsWith('/late/end')).length,1);assert.equal(x.client.getSnapshot().phase,'ended');
});
test('new call waits for old teardown; old events cannot contaminate it',async t=>{
 const x=setup(t);await x.client.start();const old=x.transports[0];ready(old);const stopped=x.client.stop();const starting=x.client.start();await Promise.all([stopped,starting]);assert.equal(x.client.getSnapshot().sessionId,'2');assert.equal(x.transports.length,2);old.data('asr_final',{turn_id:'old',text:'stale'});old.data('error',{message:'old failure'});assert.equal(x.client.getSnapshot().messages.length,0);assert.notEqual(x.client.getSnapshot().phase,'failed');
});
test('media failure releases the scoped server room and reports the error',async t=>{
 const x=setup(t,{connectError:true});await assert.rejects(x.client.start(),/media failed/);await x.client.stop();assert.equal(x.client.getSnapshot().phase,'failed');assert.equal(x.calls.filter(c=>c.url.endsWith('/end')).length,1);assert.equal(x.transports[0].disconnected,true);
});
test('capacity rejection never creates a media room',async t=>{
 const x=setup(t,{request:async()=>({ok:false,status:503})});await assert.rejects(x.client.start(),/busy/);assert.equal(x.transports.length,1);assert.equal(x.transports[0].args,undefined);assert.equal(x.client.getSnapshot().phase,'failed');
});
test('failed server cleanup remains retryable rather than losing its capability',async t=>{
 let ends=0;const x=setup(t,{request:async url=>url.endsWith('/end')?{ok:++ends>1,status:ends>1?200:500}:reply()});await x.client.start();await assert.rejects(x.client.stop(),/cleanup/);await x.client.stop();assert.equal(ends,2);
});
test('reset clears captions, extracted fields and timings before a second call',async t=>{
 const x=setup(t);await x.client.start();x.transports[0].data('asr_final',{turn_id:'one',text:'patient'});await x.client.reset();assert.equal(x.client.getSnapshot().phase,'idle');assert.equal(x.client.getSnapshot().sessionId,null);assert.equal(x.client.getSnapshot().messages.length,0);assert.equal(x.client.getSnapshot().record.revision,0);
});

test('complete smart-turn caption survives a delayed per-sentence ASR final',async t=>{
 const x=setup(t);await x.client.start();const s=x.transports[0];
 s.data('asr_final',{turn_id:'turn',text:'My score was three.'});
 s.data('smart_turn_emit',{turn_id:'turn',text:'My score was three. Actually, it is seven.'});
 s.data('asr_final',{turn_id:'turn',text:'Actually, it is seven.'});
 assert.equal(x.client.getSnapshot().messages.length,1);
 assert.equal(x.client.getSnapshot().messages[0].text,'My score was three. Actually, it is seven.');
 assert.equal(x.client.getSnapshot().speechLanguage,'en');
});

test('unsupported browsers never admit a server session or request media',async t=>{
 const x=setup(t,{supportError:'This browser cannot run live audio calls. Open this page in Chrome.'});
 await assert.rejects(x.client.start(),/Open this page in Chrome/);
 assert.equal(x.calls.length,0);assert.equal(x.transports.length,0);
 assert.equal(x.client.getSnapshot().phase,'failed');assert.equal(x.client.getSnapshot().sessionId,null);
 await x.client.reset();assert.equal(x.client.getSnapshot().phase,'idle');
});

test('blocked Chrome receives a WebRTC notice rather than an instruction to switch to Chrome',()=>{
 const serviceSource=ts.transpileModule(fs.readFileSync(__dirname+'/LiveKitService.ts','utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.CommonJS}}).outputText;
 const exports={};
 vm.runInNewContext(serviceSource,{exports,require:()=>({isBrowserSupported:()=>false}),navigator:{userAgent:'Mozilla/5.0 Chrome/145.0.0.0'}});
 const message=exports.LiveKitService.getSupportError();assert.match(message,/Allow WebRTC for this site/);assert.doesNotMatch(message,/Open this page in Chrome/);
});

test('permission is acquired before admission; no network or ready timeout runs during the prompt',async t=>{
 let allow;const permission=new Promise(r=>allow=r),timers=[];const x=setup(t,{prepare:()=>permission,timerDurations:timers});
 const starting=x.client.start();await tick();assert.equal(x.calls.length,0);assert.deepEqual(timers,[]);
 allow();await starting;assert.equal(x.calls.length,1);assert.deepEqual(timers,[15000,35000]);
});
test('ending while microphone permission is pending creates no server session',async t=>{
 const x=setup(t,{prepare:service=>new Promise(resolve=>service.cancelPreparation=resolve)});
 const starting=x.client.start();await tick();await x.client.stop();await starting;
 assert.equal(x.calls.length,0);assert.equal(x.transports[0].disconnected,true);assert.equal(x.client.getSnapshot().phase,'ended');
});
test('audio readiness deadline starts after connect and microphone publication, not while connect is pending',async t=>{
 let finish;const connected=new Promise(r=>finish=r),timers=[];const x=setup(t,{connect:()=>connected,timerDurations:timers});
 const starting=x.client.start();await tick();assert.deepEqual(timers,[15000]);
 finish();await starting;assert.deepEqual(timers,[15000,35000]);
});
test('microphone delivered after cancellation is stopped without joining any room',async()=>{
 const serviceSource=ts.transpileModule(fs.readFileSync(__dirname+'/LiveKitService.ts','utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2020,module:ts.ModuleKind.CommonJS}}).outputText;
 let deliver,stops=0;const captured=new Promise(r=>deliver=r),exports={};
 class Room{on(){}async disconnect(){}removeAllListeners(){}}
 vm.runInNewContext(serviceSource,{exports,require:()=>({Room,RoomEvent:{},createLocalAudioTrack:()=>captured}),clearInterval});
 const service=new exports.LiveKitService(),pending=service.prepareMicrophone();await service.disconnect();await pending;
 deliver({stop:()=>stops++});await tick();assert.equal(stops,1);
});
