"""Persist primary-assistant review of the saved answers. No external API calls."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evaluation/runs/2026-10-17-run01'
GOLD=ROOT/'evaluation/golden-v1/pipelinehub-golden-v1.json'
# Observed omissions against the unchanged compound criteria. Each list names
# the full criterion that was not satisfied; partial content is described below.
missed={
 'PH-G05':{'keyword':[2,3],'semantic':[2,3],'hybrid':[2,3]},
 'PH-G06':{'keyword':[1,2],'semantic':[1,2,3],'hybrid':[1,2]},
 'PH-G07':{'keyword':[3,4],'semantic':[3,4],'hybrid':[3,4]},
 'PH-G08':{'keyword':[],'semantic':[],'hybrid':[1,2]},
 'PH-G09':{'keyword':[1,2,3],'semantic':[3],'hybrid':[3]},
 'PH-G11':{'keyword':[3],'semantic':[3],'hybrid':[3]},
 'PH-G12':{'keyword':[1,2,3,4],'semantic':[],'hybrid':[4]},
 'PH-G13':{'keyword':[1,2,3,4,5],'semantic':[5],'hybrid':[2]},
 'PH-G14':{'keyword':[1,2,3,4],'semantic':[1,2,3,4],'hybrid':[1,2,3]},
 'PH-G15':{'keyword':[3,4],'semantic':[1,3],'hybrid':[3,4]},
 'PH-G16':{'keyword':[1,2,3,4,5,6],'semantic':[1,2,3,4,5,6],'hybrid':[1,4,5]},
 'PH-G17':{'keyword':[1,2,3],'semantic':[1,2,3],'hybrid':[1,2,3]},
 'PH-G18':{'keyword':[2],'semantic':[1,2],'hybrid':[1,2]},
}
notes={
 'PH-G05':{2:'Publish Type None is correct, but the required Save step is omitted.',3:'Publish CI step is identified, but explicitly triggering the pipeline is omitted.'},
 'PH-G06':{1:'Config Notifications and Project Config are named, but the Edit step is omitted.',2:'Comma-separated addresses are correct, but Save is omitted.'},
 'PH-G07':{3:'The full enabled CodeScan/DIRTY/component eligibility criterion is incomplete. Keyword omits specific type/step/DIRTY; semantic/hybrid omit the enabled-for-component condition.',4:'Component/step selection before submission is omitted. Keyword also omits reason length; semantic/hybrid show the length only in a caution.'},
 'PH-G09':{3:'The V3 service scope and resource-cleanup validation purpose are not explained.'},
 'PH-G11':{3:'The final answer does not explain the post-provisioning, current-status-dependent Enable/Disable UI choices.'},
 'PH-G12':{4:'The unsupported multiple-masking-runs limitation is omitted.'},
 'PH-G13':{2:'Versions/patches/cloud-native state are covered, but identity/database/service-integration matching is omitted.',5:'The multiple-masking-runs limitation is omitted.'},
 'PH-G14':{1:'Uses CI project/build navigation and optional BATSnapper information rather than the provisioned pod Run Tests flow for the user\'s LOB.',2:'AcceptanceTests Label, Order prop and Git branch auto-population are omitted.',3:'The new PipelineHub ID and queued/running/success pod-test tracking sequence are omitted.'},
 'PH-G15':{1:'The reply mentions releasing a lock/HOLD but not the Family pod Unlock Pod action or hidden actions; the cited HOLD workflow is different.',3:'Resize purpose/stamps appear in prose or cautions, but resize eligibility validation is omitted.',4:'The full no-conflict plus Active/AVAILABLE validation criterion is incomplete: keyword omits the states; hybrid omits an explicit conflict check.'},
 'PH-G16':{1:'The complete LOB/non-EOL copy eligibility rule is not explicitly explained; an active-release scenario alone is not a statement of both rules.',4:'Group name, description and LOB are listed, but the brief non-sensitive description requirement is omitted.',5:'Adding/staging members and saving are described, but membership verification before Save is not explicitly instructed.'},
 'PH-G17':{1:'Does not ask for both current and requested software versions.',2:'Does not establish both successful upgrade completion and an appropriate backup; semantic asks a backup name only.',3:'Does not distinguish same-version backup restore from n-1 restore after a successful upgrade. Keyword/hybrid instead contrast pod restore with a snapshot/artifact interpretation.'},
 'PH-G18':{1:'Architecture and size are asked, but database platform is omitted.',2:'The full timing/Marketing SKU and FMK/CDP decision information is incomplete. Keyword omits both; semantic asks FMK/CDP but not timing; hybrid asks timing/Marketing SKU but not FMK/CDP.'},
}
unsupported={
 'PH-G09-semantic':{4:'The quote establishes redeploying the same workload; restoring the environment state is broader and not established by this excerpt.'},
 'PH-G09-hybrid':{2:'The cited higher-BOM-version line is a CloudNative Redeploy lookup rule. It does not by itself establish the claim\'s added upgrade/patch and service-reintroduction scope.'},
 'PH-G12-semantic':{4:'The cited role restrictions belong to other OnPod operations, not data masking. The OnPod document contains no masking procedure.',5:'The RuntimeDiagnostics-enabled requirement is for startRuntimeDiagnosticsCredentialsJob, not the data masking action. This is a wrong-workflow prerequisite.'},
 'PH-G15-semantic':{4:'Conflates release of a HOLD with the separate Family pod Unlock Pod action. AVAILABLE scheduling evidence does not establish that equivalence.'},
}
relevance={
 'PH-G02':'Extra caution about long-running pods may distract from the role/schedule answer.',
 'PH-G05':'Core identifiers are right; the frozen rubric also requires Save and pipeline trigger details, making this a strict completeness failure.',
 'PH-G06':'Core field/format/triggers are right for keyword/hybrid; the frozen rubric also requires Edit and Save. Semantic suffered an incomplete generation response.',
 'PH-G07':'Some source cautions are standalone headings rather than useful restrictions. The six-claim cap leaves dialog prerequisites incomplete.',
 'PH-G08':'Hybrid withheld a supported answer despite retrieving the correct recovery instruction.',
 'PH-G09':'Additional version/redeployment explanations extend beyond the requested documented behavior; cleanup purpose is omitted.',
 'PH-G11':'The destructive-data warning is correct; the rubric also requires the status-dependent UI explanation. Extra unrelated cautions appear.',
 'PH-G12':'Semantic adds unsafe prerequisites from a different OnPod workflow. Hybrid omits the multiple-runs limitation; keyword asks unnecessary clarification.',
 'PH-G13':'Multi-document answer is partly complete, but each mode omits a different important condition.',
 'PH-G14':'Hybrid gives supported facts about other CI workflows rather than the requested pod-test start/tracking procedure. Keyword/semantic withhold the answer.',
 'PH-G15':'Resize/locking guidance is incomplete; semantic confuses HOLD release with the Family pod lock workflow.',
 'PH-G16':'Hybrid combines both topics but misses some strict procedural criteria. Other modes escalate.',
 'PH-G17':'Correctly asks for clarification, but the questions do not collect the restore eligibility details required by the rubric.',
 'PH-G18':'Correct clarification route, but the question misses some method-selection inputs.',
 'PH-G19':'Generic evidence-gap fallback is accepted in the context of the user\'s named request; it neither invents YAML nor denies product capability.',
 'PH-G20':'Generic evidence-gap fallback is accepted in context; it neither invents connector scopes nor claims an incident was created.',
}

def main():
    (RUN/'judgments').mkdir(exist_ok=True)
    expected=json.loads((RUN/'review-input-fingerprints.json').read_text())
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('PH-G*.json')}==expected, 'These annotations apply only to the frozen run01 responses'
    gold={c['id']:c for c in json.loads(GOLD.read_text())['cases']}
    for file in sorted(RUN.glob('PH-G*.json')):
        raw=json.loads(file.read_text());case=gold[raw['id']];result=raw.get('result',{})
        missing=missed.get(raw['id'],{}).get(raw['mode'],[])
        points=[]
        for number,point in enumerate(case['required_answer_points'],1):
            met=number not in missing
            if result.get('status')=='needs_human_review' and case['expected_status']=='answered':
                met=False;reason='The final reply withholds the answer; an internal draft does not satisfy delivered-answer criteria.'
            elif raw['run_status']!='completed':
                met=False;reason='No valid completed generation was delivered.'
            else:
                reason=notes.get(raw['id'],{}).get(number,'Not conveyed in the final response.') if not met else 'The final response conveys this criterion in context; equivalent wording and valid source cautions count.'
            points.append({'criterion_id':point['criterion_id'],'met':met,'reason':reason})
        claims=(result.get('draft') or {}).get('claims',[]) if result.get('status')=='answered' else []
        claim_review=[]
        for i,claim in enumerate(claims,1):
            issue=unsupported.get(file.stem,{}).get(i)
            claim_review.append({'claim_number':i,'supported_by_cited_evidence':not bool(issue),
                                 'reason':issue or 'Claim meaning is supported by the cited quote(s) and the retrieved source context reviewed with the canonical evidence.'})
        judgment={'points':points,'claim_support':claim_review,'unacceptable_claim_present':False,
                  'unacceptable_claim_reason':'None of the specifically listed prohibited claims occurs in the final reply. Separate unsupported claims are recorded above.',
                  'relevance_note':relevance.get(raw['id'],'Relevant response with traceable source evidence.')}
        review={'id':raw['id'],'mode':raw['mode'],'judge_model':'Primary Codex assistant in this chat; no external grading call',
                'review_type':'AI-assisted local review against frozen labels; not a human SME review',
                'raw_response_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),
                'status':'scored','judgment':judgment,'external_grading_calls':0}
        (RUN/'judgments'/file.name).write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
    summary={'scored':60,'errors':0,'reviewer':'Primary Codex assistant','external_grading_calls':0,'human_reviewed':False,
             'approval_note':'Automatic approval review rejected sending gold labels to an external Nebius judge. No such upload occurred; primary assistant reviewed saved answers locally.',
             'grading_policy':'Frozen criteria unchanged. All components of each compound criterion are required. This can fail useful core answers for omitted Save/Edit steps or contextual details; an SME should review rubric scope before a submission-grade claim.'}
    (RUN/'judge-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved 60 local rubric reviews; no grading API calls.')

if __name__=='__main__':main()
