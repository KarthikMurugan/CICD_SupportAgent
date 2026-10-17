"""Compute transparent metrics from saved bot runs and post-run rubric judgments."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evaluation/runs/2026-10-17-run01'
GOLD=ROOT/'evaluation/golden-v1/pipelinehub-golden-v1.json'

def percentile(values,p):
    x=sorted(values);index=(len(x)-1)*p;lo=int(index);hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(index-lo)

def main():
    package=json.loads(GOLD.read_text());gold={c['id']:c for c in package['cases']}
    overrides_path=RUN/'review-overrides.json'
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    rows=[];details=[]
    for file in sorted(RUN.glob('PH-G*.json')):
        raw=json.loads(file.read_text());case=gold[raw['id']];result=raw.get('result',{})
        judgefile=RUN/'judgments'/file.name
        review=json.loads(judgefile.read_text()) if judgefile.exists() else {'status':'not_scored'}
        judged=review.get('judgment',{})
        override=overrides.get(file.stem,{})
        points={p['criterion_id']:p for p in judged.get('points',[])}
        for key,value in override.get('points',{}).items():
            points[key]={'criterion_id':key,'met':value['met'],'reason':value['reason'],'reviewer':'Primary assistant post-run rubric review; not a human SME'}
        claims={str(c['claim_number']):c for c in judged.get('claim_support',[])}
        for key,value in override.get('claim_support',{}).items():
            claims[key]={'claim_number':int(key),'supported_by_cited_evidence':value['supported'],'reason':value['reason'],'reviewer':'Primary assistant post-run source review; not a human SME'}
        unacceptable=override.get('unacceptable_claim_present',judged.get('unacceptable_claim_present'))
        scored=review['status']=='scored' and len(points)==len(case['required_answer_points'])
        met=sum(bool(p['met']) for p in points.values());required=len(case['required_answer_points'])
        critical_missed=[p['criterion_id'] for p in case['required_answer_points'] if p['critical'] and not points.get(p['criterion_id'],{}).get('met',False)]
        actual=result.get('status','execution_error');executed=raw['run_status']=='completed'
        route=executed and actual==case['expected_status']
        group_hits=[any(s['document_id'] in group for s in result.get('retrieved_sources',[])) for group in case['required_evidence_groups']]
        doc_recall=sum(group_hits)/len(group_hits) if group_hits else None
        relevant=set(case['gold_document_ids'])
        ranks=[i+1 for i,s in enumerate(result.get('retrieved_sources',[])) if s['document_id'] in relevant]
        mrr=1/min(ranks) if ranks else (0 if relevant else None)
        quotes=result.get('supporting_quotes',[])
        exact_valid=sum(any(q['document_id']==d['document_id'] and q['quote'] in d['text'] for d in result.get('retrieved_sources',[])) for q in quotes)
        cautions=result.get('source_cautions',[])
        cautions_valid=sum(any(q['document_id']==d['document_id'] and q['quote'] in d['text'] for d in result.get('retrieved_sources',[])) for q in cautions)
        delivered=(result.get('draft') or {}).get('claims',[]) if actual=='answered' else []
        all_supported=len(claims)==len(delivered) and all(c['supported_by_cited_evidence'] for c in claims.values())
        citation_ok=actual!='answered' or (bool(quotes) and exact_valid==len(quotes) and all_supported)
        case_pass=bool(scored and route and met==required and unacceptable is False and citation_ok)
        resolved=case_pass and case['expected_status']=='answered' and actual=='answered'
        row={'id':raw['id'],'category':case['category'],'mode':raw['mode'],'expected_status':case['expected_status'],
             'actual_status':actual,'execution_status':raw['run_status'],'routing_correct':route,
             'required_points_met':met if scored else None,'required_points_total':required,'point_coverage':met/required if scored else None,
             'critical_points_missed':'; '.join(critical_missed) if scored else '',
             'unacceptable_claim_present':unacceptable,'rubric_scored':scored,'rubric_case_pass':case_pass if scored else None,
             'simulated_resolved':resolved if scored else None,'gold_document_group_recall_at_6':doc_recall,
             'all_gold_groups_found_at_6':all(group_hits) if group_hits else None,'first_gold_document_mrr_at_6':mrr,
             'latency_seconds':raw['wall_seconds'],'prompt_tokens':raw['token_usage']['prompt_tokens'],
             'completion_tokens':raw['token_usage']['completion_tokens'],'generation_cost_estimate_usd':(raw['token_usage']['prompt_tokens']*.6+raw['token_usage']['completion_tokens']*3.6)/1_000_000,
             'delivered_draft_claims':len(delivered),'claims_supported_by_cited_evidence_ai':sum(c['supported_by_cited_evidence'] for c in claims.values()),
             'source_cautions_valid':cautions_valid,'source_cautions_total':len(cautions),'exact_quotes_valid':exact_valid,'quotes_total':len(quotes),'review_overridden':bool(override),
             'question':case['question'],'answer':result.get('answer',''),'review_notes':override.get('note',judged.get('relevance_note',''))}
        rows.append(row)
        details.append({'id':raw['id'],'mode':raw['mode'],'required_points':list(points.values()),'claim_support':list(claims.values()),
                        'unacceptable_claim_present':unacceptable,'original_judge':str(judgefile.relative_to(ROOT)),
                        'primary_assistant_review':override or None})
    assert len(rows)==60
    summaries=[]
    for mode in ['keyword','semantic','hybrid']:
        subset=[r for r in rows if r['mode']==mode];assert len(subset)==20
        answerable=[r for r in subset if r['expected_status']=='answered'];answered=[r for r in subset if r['actual_status']=='answered']
        supported=sum(r['claims_supported_by_cited_evidence_ai'] for r in answered);claim_count=sum(r['delivered_draft_claims'] for r in answered)
        totalpoints=sum(r['required_points_total'] for r in subset);metpoints=sum(r['required_points_met'] or 0 for r in subset)
        resolved=sum(bool(r['simulated_resolved']) for r in subset)
        summaries.append({'mode':mode,'queries':20,'execution_errors':sum(r['execution_status']!='completed' for r in subset),
            'answered':len(answered),'clarified':sum(r['actual_status']=='needs_clarification' for r in subset),
            'escalated':sum(r['actual_status']=='needs_human_review' for r in subset),
            'routing_correct':sum(r['routing_correct'] for r in subset),'routing_accuracy':sum(r['routing_correct'] for r in subset)/20,
            'rubric_cases_scored':sum(r['rubric_scored'] for r in subset),'rubric_cases_passed':sum(bool(r['rubric_case_pass']) for r in subset),
            'rubric_pass_rate_ai_assisted':sum(bool(r['rubric_case_pass']) for r in subset)/20,
            'required_points_met':metpoints,'required_points_total':totalpoints,'required_point_coverage_ai':metpoints/totalpoints,
            'simulated_resolved':resolved,'simulated_fcr_all_20_ai':resolved/20,'simulated_resolution_answerable_16_ai':resolved/16,
            'false_human_escalations_answerable':sum(r['actual_status']=='needs_human_review' for r in answerable),
            'gold_document_group_recall_at_6':statistics.mean(r['gold_document_group_recall_at_6'] for r in answerable),
            'all_gold_groups_found_cases':sum(r['all_gold_groups_found_at_6'] for r in answerable),
            'gold_document_mrr_at_6':statistics.mean(r['first_gold_document_mrr_at_6'] for r in answerable),
            'latency_mean_seconds':statistics.mean(r['latency_seconds'] for r in subset),
            'latency_median_seconds':statistics.median(r['latency_seconds'] for r in subset),'latency_p95_seconds':percentile([r['latency_seconds'] for r in subset],.95),
            'answered_latency_median_seconds':statistics.median(r['latency_seconds'] for r in answered) if answered else None,
            'responses_under_10_seconds':sum(r['latency_seconds']<=10 for r in subset),
            'supported_draft_claims_ai':supported,'delivered_draft_claims':claim_count,
            'draft_claim_citation_support_ai':supported/claim_count if claim_count else None,
            'source_cautions_valid':sum(r['source_cautions_valid'] for r in subset),'source_cautions_total':sum(r['source_cautions_total'] for r in subset),
            'exact_quotes_valid':sum(r['exact_quotes_valid'] for r in subset),'quotes_total':sum(r['quotes_total'] for r in subset),
            'prompt_tokens':sum(r['prompt_tokens'] for r in subset),'completion_tokens':sum(r['completion_tokens'] for r in subset),
            'generation_cost_estimate_usd':sum(r['generation_cost_estimate_usd'] for r in subset),
            'unsupported_routing_correct':sum(r['routing_correct'] for r in subset if r['category']=='unsupported_escalation'),
            'ambiguous_routing_correct':sum(r['routing_correct'] for r in subset if r['category']=='ambiguous_clarification'),
            'category_results':{category:{'cases':len(ss:=[r for r in subset if r['category']==category]),'case_passes':sum(bool(r['rubric_case_pass']) for r in ss),'routing_correct':sum(r['routing_correct'] for r in ss)} for category in ['exact_match','semantic_troubleshooting','multi_passage','ambiguous_clarification','unsupported_escalation']},
            'route_confusion':{expected:{actual:sum(r['expected_status']==expected and r['actual_status']==actual for r in subset) for actual in ['answered','needs_clarification','needs_human_review']} for expected in ['answered','needs_clarification','needs_human_review']}})
    report={'evaluation_type':'Offline end-to-end evaluation with real Nebius/Pinecone requests and paired retrieval-mode comparison',
            'dataset_version':'1.0','cases':20,'runs':60,'gold_dataset_sha256':hashlib.sha256(GOLD.read_bytes()).hexdigest(),
            'generation_model':'Qwen/Qwen3.5-397B-A17B','judge_model':'Primary Codex assistant in this chat; no external grading call',
            'quality_labels':'Primary-assistant AI review of all 60 saved responses against frozen labels, with per-criterion and per-claim reasons; not independent human validation.',
            'faithfulness_scope':'AI estimate covers delivered structured draft claims only. Verbatim source cautions are checked for source traceability separately and may be irrelevant. Clarification statements and all rendered prose have not been exhaustively claim-segmented.',
            'retrieval_scope':'Macro mean gold-document-group recall in the final maximum six retrieved passages, on 16 answerable cases. Document matches are not passage entailment. Labels permit manually verified alternatives, but no alternative documents were automatically added.',
            'latency_scope':'One trial per question/mode, randomized order, two requests concurrently. Includes retriever initialization; not a statistically reliable production latency benchmark.',
            'fcr_scope':'Simulated offline strict rubric resolution only. Overall denominator 20 caps attainable FCR at 80% because 4 cases require clarification/escalation. Answerable-only denominator 16 reported separately. No actual customer FCR measured.',
            'price_source':'https://tokenfactory.nebius.com/api/public/models_info',
            'grading_privacy':'No gold labels were sent to a separate external grading API. External grading was rejected by automatic approval review; primary assistant reviewed captured responses in this chat instead.',
            'limitations':['Same assistant authored the labels and reviews; no independent human SME review.','Only 20 questions and one run per mode; small-sample, single-trial results.', 'Compound criteria require all details, including Save/Edit in some factual questions; strict resolution may understate usefulness of correct core answers. SME review is needed before treating this as submission-grade gold.','Primary-assistant rubric review is fallible; see all per-criterion and per-claim judgments.','No prompt/routing/retrieval tuning during this frozen run. If used for tuning afterward, this set becomes development data.','Generation cost is a token-based estimate at $0.60/M input and $3.60/M output, excluding judge/Pinecone costs, taxes and billing adjustments.'],
            'summaries':summaries}
    (RUN/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (RUN/'scored-details.json').write_text(json.dumps(details,indent=2,ensure_ascii=False)+'\n')
    with (RUN/'case-results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (RUN/'case-results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
