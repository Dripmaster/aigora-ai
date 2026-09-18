"""Explicit opt-in live smoke: synthetic data only, at most nine low-token model calls.
Runs this checkout's FastAPI locally; never contacts the production application or DB.
"""
import argparse,json,os,socket,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import httpx

CASES=[
 ('/classify-gpt',{'text':'매달 10만원씩 12개월간 모으고 월급 다음 날 자동이체하겠습니다.','user_id':'financial-api-smoke','context':{'lesson_id':1}}),
 ('/qa',{'nickname':'financial-api-smoke','discussion_topic':'상품 고르기','video_id':'financial_1','questionText':'소득이 불규칙할 때 정기적금과 자유적금 중 어떤 차이를 확인해야 하나요?','chat_history':[]}),
 ('/question',{'nickname':'financial-api-smoke','discussion_topic':'역산하기','video_id':'financial_1','chat_history':[]}),
 ('/evaluate',{'user_id':'financial-api-smoke','user_messages':[{'text':'매달 10만원씩 12개월 모으고 월급 다음 날 자동이체하겠습니다.'}],'discussion_context':{'topic':'목표 정하기','round_number':1}}),
 ('/discussion-overall',{'all_user_messages':[{'nickname':'financial-api-smoke','text':'매달 10만원씩 12개월 모으겠습니다.'}],'discussion_context':{'topic':'목표 정하기','round_number':1}}),
 ('/user-summary',{'user_id':'financial-api-smoke','chat_history':[{'nickname':'financial-api-smoke','text':'매달 10만원씩 12개월 모으겠습니다.'}],'discussion_topics':[{'name':'목표 정하기'}]}),
 ('/encouragement',{'nickname':'financial-api-smoke','chat_history':[]}),
 ('/profile',{'user_id':'financial-api-smoke','messages':[{'text':'매달 10만원씩 12개월 모으겠습니다.'}]}),
 ('/classify',{'text':'보험료 부담과 보장 범위를 확인하겠습니다.','user_id':'financial-api-smoke','context':{'lesson_id':2}}),
]

def run(output):
 root=Path(__file__).resolve().parent.parent
 with socket.socket() as probe:
  probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
 process=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port)],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 report={'timestamp':datetime.now(timezone.utc).isoformat(),'scope':'local checkout, real provider, synthetic messages; no production/DB','cases':[]}
 try:
  with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=45,trust_env=False) as client:
   for _ in range(80):
    try:
     response=client.get('/health')
     if response.status_code==200:break
    except httpx.HTTPError:pass
    time.sleep(.1)
   else:raise RuntimeError('Local API did not start')
   report['health']=response.json()
   if not report['health']['provider_configured']:raise RuntimeError('Provider key not configured')
   for endpoint,payload in CASES:
    start=time.monotonic();r=client.post(endpoint,json=payload)
    body=r.json()
    # Do not copy error details (SDK messages can include fragments of credentials).
    item={'endpoint':endpoint,'http_status':r.status_code,'seconds':round(time.monotonic()-start,2),'response':body if r.status_code==200 else {'error':'request_failed'}}
    report['cases'].append(item)
    print(endpoint,r.status_code,item['seconds'],flush=True)
    if endpoint=='/classify-gpt' and (r.status_code!=200 or body.get('evaluation_status')!='assessed'):
     report['stopped']='Initial live classifier did not produce an assessment; remaining paid calls skipped.'
     break
 finally:
  process.terminate()
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:process.kill();process.wait()
  Path(output).parent.mkdir(parents=True,exist_ok=True)
  Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true',help='Allow real model calls');parser.add_argument('--output',default='docs/api-live-smoke-2026-09-15.json');args=parser.parse_args()
 if not args.live:parser.error('Real model calls require explicit --live')
 run(args.output)
