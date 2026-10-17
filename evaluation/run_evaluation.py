"""Run only frozen questions through the production workflow; never load labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

from cicd_support_rag.bot import ask
from cicd_support_rag.generation import generation_client,GenerationError

ROOT=Path(__file__).resolve().parents[1]

class MeteredModel:
    def __init__(self):
        self.client=generation_client()
        self.model=self.client.model
        self.calls=[]
    def complete(self,schema,system,payload):
        started=time.monotonic()
        call={'schema':schema.__name__}
        try:
            out=self.client.complete(schema,system,payload)
            call['usage']=getattr(self.client,'last_usage',None)
            call['status']='ok'
            return out
        except GenerationError as error:
            call['status']='error';call['error']=str(error)
            raise
        finally:
            call['seconds']=round(time.monotonic()-started,4)
            self.calls.append(call)

def run_case(item,directory):
    case,mode=item
    target=directory/f"{case['id']}-{mode}.json"
    if target.exists():return json.loads(target.read_text()),True
    generator=MeteredModel();started=time.monotonic();stages=[]
    try:
        result=ask(case['question'],model=generator,retrieval_mode=mode,
                   on_stage=lambda stage:stages.append({'stage':stage,'elapsed_seconds':round(time.monotonic()-started,4)}))
        record={'id':case['id'],'mode':mode,'question':case['question'],'run_status':'completed',
                'completed_at_utc':datetime.now(timezone.utc).isoformat(),'result':result}
    except Exception as error:
        # No raw exception messages from network clients or credentials.
        record={'id':case['id'],'mode':mode,'question':case['question'],'run_status':'execution_error',
                'error_type':type(error).__name__}
    record['generation_calls']=generator.calls
    if any(c['status']=='error' for c in generator.calls):record['run_status']='generation_error'
    record['stages']=stages
    record['wall_seconds']=round(time.monotonic()-started,4)
    record['token_usage']={k:sum((c.get('usage') or {}).get(k,0) or 0 for c in generator.calls) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');temp.replace(target)
    return record,False

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'evaluation/runs/2026-10-17-run01')
    args=parser.parse_args();directory=args.run_dir;directory.mkdir(parents=True,exist_ok=True)
    source=ROOT/'evaluation/golden-v1/pipelinehub-questions-v1.jsonl'
    manifest=json.loads((source.parent/'manifest.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==manifest['files'][source.name]
    cases=[json.loads(x) for x in source.read_text().splitlines()]
    assert len(cases)==20 and all(set(c)=={'id','question'} for c in cases)
    model=generation_client()
    config={'question_file_sha256':digest,'gold_manifest_sha256':hashlib.sha256((source.parent/'manifest.json').read_bytes()).hexdigest(),
            'generation_model':model.model,'provider':type(model).__name__,'modes':['keyword','semantic','hybrid'],
            'production_workflow':'retrieve 10; drop deprecated titles; first 6 passages; unchanged draft/review prompts and source-caution rendering',
            'candidate_limit_per_retriever':15,'passages_in_context_max':6,'concurrency':2,'shuffle_seed':20261017,
            'same_questions_all_modes':True,'labels_loaded_by_runner':False,'generation_temperature':0,
            'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/cicd_support_rag').glob('*.py')},
            'measurement_notes':'Single run per case/mode. End-to-end includes retriever/model-client initialization. Two requests in flight; not an isolated single-user latency benchmark. No answer-dependent retries or tuning.'}
    configfile=directory/'run-config.json'
    if configfile.exists():assert json.loads(configfile.read_text())==config,'Configuration changed; use a new run directory'
    else:configfile.write_text(json.dumps(config,indent=2)+'\n')
    tasks=[(c,mode) for c in cases for mode in config['modes']];random.Random(config['shuffle_seed']).shuffle(tasks)
    started=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_case,task,directory):task for task in tasks}
        for future in as_completed(futures):
            record,cached=future.result();results.append(record)
            output=record.get('result',{})
            print(json.dumps({'completed':len(results),'of':60,'id':record['id'],'mode':record['mode'],'status':output.get('status',record['run_status']),
                              'seconds':record['wall_seconds'],'cached':cached}),flush=True)
    summary={'completed_runs':len(results),'wall_seconds':round(time.monotonic()-started,3),
             'errors':sum(r['run_status']!='completed' for r in results),
             'raw_results_saved':True,'answer_quality_scored':False}
    (directory/'execution-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
