"""Author the corpus-grounded holdout without calling retrieval or generation."""
import ast
import csv
import hashlib
from html.parser import HTMLParser
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CORPUS=ROOT.parent/'sources/PipelineHub-corpus.json'
raw=CORPUS.read_bytes(); corpus=json.loads(raw)
docs={d['id']:d for d in corpus['documents']}
prior=set(); checked=[]
def walk(v):
    if isinstance(v,dict):
        for k,x in v.items():
            if k in {'question','query'} and isinstance(x,str): prior.add(x)
            else: walk(x)
    elif isinstance(v,list):
        for x in v: walk(x)
for folder in ['evaluation','data/bot-checks','data/ui-checks']:
    for p in sorted((ROOT/folder).glob('*.json')):
        walk(json.loads(p.read_text())); checked.append(str(p.relative_to(ROOT)))
for p in sorted((ROOT/'tests').glob('test*.py')):
    checked.append(str(p.relative_to(ROOT)))
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and '?' in n.value:
            prior.add(n.value)
# Exclude starter/demo questions found in code and visible UI examples too.
for name in ['src/cicd_support_rag/bot.py','src/cicd_support_rag/web/app.js','src/cicd_support_rag/web/index.html']:
    p=ROOT/name;checked.append(name)
    text=p.read_text()
    for m in re.finditer(r'''["']([^"'\n]{8,200}\?)["']''',text):prior.add(m.group(1))

class DemoPrompts(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key == 'data-prompt': prior.add(value)
DemoPrompts().feed((ROOT/'src/cicd_support_rag/web/index.html').read_text())
prior = {q for q in prior if re.match(r'^(How|What|Which|Can|My|It)\b', q) and len(q) < 400}

def ev(doc_id,start=None,end=None):
    d=docs[doc_id]; text=d['content']; a=text.index(start) if start else 0
    b=text.index(end,a+len(start or '')) if end else len(text)
    quote=text[a:b].rstrip();b=a+len(quote)
    return {'document_id':doc_id,'parent_id':d['parent_id'],'title':d['title'],
            'char_start':a,'char_end':b,'offset_basis':'canonical document content; Python Unicode code points; end exclusive',
            'quote':quote,'quote_sha256':hashlib.sha256(quote.encode()).hexdigest()}

records=[]
def add(case_id,category,question,answer,points,avoid,evidence,status='answered',clarification_slots=None,absence_terms=None):
    evidence=[dict(e,evidence_id=f'{case_id}-E{i:02}') for i,e in enumerate(evidence,1)]
    parsed=[]
    for i,(text,refs,critical) in enumerate(points,1):
        parsed.append({'criterion_id':f'{case_id}-C{i:02}','text':text,'evidence_ids':[evidence[j-1]['evidence_id'] for j in refs],'critical':critical})
    ids=list(dict.fromkeys(e['document_id'] for e in evidence))
    records.append({'id':case_id,'category':category,'question':question,'expected_status':status,
        'reference_answer':answer,'required_answer_points':parsed,'unacceptable_claims':avoid,
        'gold_document_ids':ids if status=='answered' else [],
        'required_evidence_groups':[[d] for d in ids] if status=='answered' else [],
        'supporting_evidence':evidence,'clarification_slots':clarification_slots or [],
        'absence_search_terms':absence_terms or [],'answerable_without_clarification':status=='answered',
        'resolution_eligible':status=='answered','citation_policy':'Cite supporting document IDs and titles for substantive documentation claims. Any alternative corpus passage is acceptable only if it actually supports the same claim.',
        'annotation_origin':'Authored from canonical corpus, not from chatbot responses; human SME review pending.'})

add('PH-G01','exact_match',
 'In Add a Free Form Tag, what format is accepted, and how many tags may I send in one submission?',
 'Use key=value and submit only one free form tag per submission. The value may contain nested key/value pairs, for example key=key1=value1,key2=value2. [doc-0080]',
 [('Only one free form tag is allowed per submission.',[1],True),('The required structure is key=value.',[1],True),('Nested key/value pairs are permitted inside the value.',[1],False)],
 ['Multiple separate tags may be submitted in one request.','JSON or a colon-separated pair is the required tag format.'],
 [ev('doc-0080','Users can add free form tags')])

add('PH-G02','exact_match',
 'Which PipelineHub role is required for Pod Add-ons > Toggle Auto Scheduling, and what schedules does this action control?',
 'The role is ROLE_PREFLIGHT_AND_FAMILY_ADMIN. Toggle Auto Scheduling allows or disallows monthly and quarterly patch bundle patching events. [doc-0085]',
 [('Name ROLE_PREFLIGHT_AND_FAMILY_ADMIN exactly.',[1],True),('The action controls monthly and quarterly auto-scheduled patch bundle patching.',[1],True)],
 ['Project Control alone is the documented permission.','The action controls only daily CI builds.'],
 [ev('doc-0085')])

add('PH-G03','exact_match',
 'For the Toggle MFA post-pod action, which pod identity configurations are eligible, and is IDM supported?',
 'Toggle MFA is available for IdentityService and IdentityService/ManagedDatabase pods. It is not available for IDM pods, and the UI should not show the feature for those pods. [doc-0098]',
 [('IdentityService pods are eligible.',[1],True),('IdentityService/ManagedDatabase pods are eligible.',[1],True),('IDM pods are not supported and should not show the action.',[1],True)],
 ['Toggle MFA is supported for IDM pods.','All pod types support the action.'],
 [ev('doc-0098','1. PipelineHub will have','2. if MFA')])

add('PH-G04','exact_match',
 'Does the build compatibility table support an EnterpriseLinux9 base image on OL8u6 UEK6? What does it report for compile, Docker creation, and publish?',
 'Yes. That exact row is marked GreenSupported, with successful compile, Docker creation, and publish checks. This is the documented build compatibility result, not a claim about every OS/kernel combination. [doc-0050]',
 [('Match the EnterpriseLinux9 base image with OL8u6 UEK6.',[1],True),('Compile, Docker Creation and Publish all show success.',[1],True),('The row is GreenSupported.',[1],True)],
 ['EnterpriseLinux9 requires UEK7 in every supported build.','Every OS and kernel combination is supported.'],
 [ev('doc-0050','Supported CI/CD Combinations')])

add('PH-G05','exact_match',
 'For a Python Lifecycle Hook container, which component type and ArtifactService Publish Type should I choose, and which CI step must be enabled?',
 'Use MICROSERVICE_PYTHON_COMMON_COMPONENT, set ArtifactService Publish Type to None and save, then include the Publish CI step and trigger the pipeline to build the Lifecycle Hook as a Docker container. [doc-0108]',
 [('Use MICROSERVICE_PYTHON_COMMON_COMPONENT.',[1],True),('Set ArtifactService Publish Type to None and save.',[1],True),('Include the Publish CI step, then trigger the pipeline to produce the container.',[1],True)],
 ['Setting publish type None means the Publish CI step must be disabled.','Use an invented PYTHON_LIFECYCLE_HOOK component type.'],
 [ev('doc-0108','### Step 1:','### Step 4:')])

add('PH-G06','exact_match',
 'In Project Config, which field holds recipients for project configuration-change emails, how do I enter multiple recipients, and which changes trigger those emails?',
 'Open the project, choose Project Config, click Edit, enter comma-separated email addresses in Config Notifications, and Save. Changes to basic information, source repository, build, deploy, or test configuration notify those recipients. [doc-0022]',
 [('Use Project Config, click Edit, and enter recipients in Config Notifications.',[1],True),('Enter multiple email addresses as comma-separated values and Save.',[1],True),('Notifications cover changes to basic information, source repository, build, deploy and test configuration.',[1],True)],
 ['Use a space-separated address list as the documented required format.','These project configuration-change emails are documented only as alerts for failed builds.'],
 [ev('doc-0022','### Configure email notification for project level config change','### Configure email notification for branch level')])

add('PH-G07','semantic_troubleshooting',
 'Our latest UI component build has finished, but the code-quality check retry control is missing or greyed out. What should I check before trying to rerun just that scan?',
 'For Rerun Scans, check that you have Project Control, are on the latest eligible snapshot, and the pipeline contains a UIBuildService component. A build or rerun submission still in progress, no eligible steps/components, or missing permission can disable the action. In the dialog, the currently supported target is CodeScan on UIBuildService Application/Web components; the step must be enabled and its quality DIRTY. Select a component and step, and provide a reason of 10–60 characters. These are diagnostic checks, not a confirmed diagnosis of your run. [doc-0038]',
 [('Require Project Control and the latest eligible snapshot with a UIBuildService component.',[1],True),('Check for build/rerun in progress, missing permission, or no eligible components/steps.',[1],True),('Currently eligible reruns are enabled CodeScan steps with DIRTY quality on supported UIBuildService Application/Web components.',[2],True),('Select a component and a step; reason length must be 10–60 characters.',[2],True)],
 ['Any scanner on any component may be rerun.','A reason shorter than 10 characters is valid.','Declare a specific root cause without the user\'s run details.'],
 [ev('doc-0038','### Rerun Scans Action Button Restrictions','### Rerun Scans Dialog Restrictions'),ev('doc-0038','### Rerun Scans Dialog Restrictions','After Rerun Submission')])

add('PH-G08','semantic_troubleshooting',
 'Adding another display language to an already provisioned environment timed out in PipelineHub. Is there a documented way to keep tracking that operation?',
 'Additional language packs are a post-provisioning Pod Add-on for EnvironmentService pods. If the job times out, the documented recovery is to restart monitoring on the UPDATE_ADDITIONAL_LANGUAGES stage. The guide does not say to recreate the pod or promise that restarting monitoring reruns the installation. [doc-0084]',
 [('The capability is a post-provisioning Pod Add-on for EnvironmentService pods.',[1],False),('Restart monitoring on UPDATE_ADDITIONAL_LANGUAGES after a timeout.',[1],True)],
 ['Recreate the pod as the documented first recovery step.','Restarting monitoring necessarily resubmits the installation.','The operation is guaranteed complete because monitoring timed out.'],
 [ev('doc-0084','### Configure additional languages')])

add('PH-G09','semantic_troubleshooting',
 'I removed a V3 cloud-native service from a test pod, but it returned after an upgrade. Does the documentation explain that behavior?',
 'Yes. Delete a CloudNative Service is a Pod Add-on intended to validate CloudPlatform resource cleanup and delete a particular V3 service. The guide explicitly says later Patching, Upgrade, or EnableCloudMode actions restore deleted cloud-native services. The deletion is therefore not documented as permanent across those actions. [doc-0081]',
 [('Later Upgrade actions restore services deleted with this Pod Add-on.',[1],True),('Patching and EnableCloudMode also restore deleted services.',[1],True),('The action targets V3 services and supports resource-cleanup validation.',[1],False)],
 ['The service cannot return without manual redeployment.','The guide guarantees permanent deletion across upgrades.'],
 [ev('doc-0081','### PipelineHub supports')])

add('PH-G10','semantic_troubleshooting',
 'I can see a provisioning record from my LOB, but Copy Provisioning Job says its release set is EOL. Can I bypass that by editing the copied form?',
 'No documented bypass is provided. Copying requires both association with your LOBs and a release version that is not end-of-life. An EOL source provisioning job is not eligible for copying and produces an error. The editable prefilled form describes eligible active-release copies, not a way around the EOL check. [doc-0091]',
 [('LOB association alone is insufficient; source release must not be EOL.',[1],True),('An EOL source is ineligible and the copy flow shows an error.',[2],True),('Do not prescribe editing the copied form as an EOL bypass.',[1,2],True)],
 ['Copy first and change the version to bypass the EOL restriction.','LOB membership overrides release eligibility.'],
 [ev('doc-0091','Scope','Use Case Definition'),ev('doc-0091','### Use Case 2:')])

add('PH-G11','semantic_troubleshooting',
 'We want to turn off RuntimeDiagnostics on a pod but keep the replicated analytics data. Will disabling it preserve that data?',
 'No. Disable RuntimeDiagnostics removes its instance and deletes the Replication Infrastructure and all replicated data from the associated AnalyticsDatabase instance. It is a Post Provisioning Pod Add-on; the displayed enable/disable choice depends on the current status. Do not disable it expecting the replicated data to remain. [doc-0083]',
 [('Disabling removes the RuntimeDiagnostics instance.',[1],True),('It deletes Replication Infrastructure and all replicated AnalyticsDatabase data.',[1],True),('It is a Post Provisioning Pod Add-on with choices based on current status.',[1],False)],
 ['Disable only pauses replication while preserving all replicated data.','Suggest an undocumented export/backup mechanism as a guaranteed solution.'],
 [ev('doc-0083','### Enable/Disable RuntimeDiagnostics')])

add('PH-G12','semantic_troubleshooting',
 'Our non-production pod is on 11.13.24.10.0-10999 and we want to mask customer fields. Is it eligible, and what else must be in place?',
 'That version is below the documented minimum 11.13.24.10.0-11000, so it does not meet the stated version requirement. Data Masking applies only to non-production pods and needs a DM template already ingested for the release, such as GA or HWM. The guide lists multiple data-masking runs as unsupported. It does not provide an upgrade procedure for this pod. [doc-0092]',
 [('The supplied -10999 build does not meet the minimum 11.13.24.10.0-11000.',[1],True),('Only non-production pods are eligible.',[1],True),('The release must already have an ingested DM template.',[1],True),('Multiple data masking runs are not supported in this guide.',[1],True)],
 ['Non-production status alone makes this pod eligible.','Repeated masking runs are documented as supported.','Prescribe an unprovided upgrade command.'],
 [ev('doc-0092','Support Data Masking as post pod action','More details on Data Making:')])

add('PH-G13','multi_passage',
 'We plan to refresh a TEST pod from Production and then mask its customer data. What prerequisites apply to the refresh and the subsequent masking, and will the refresh copy feature flags and extra language packs?',
 'Before P2T refresh, back up the target. The source and target must match BusinessApp version and patch/cloud-native state; the guide also calls for matching IdentityService/ManagedDatabase/IDM and compatible service integration. Additional language packs are not carried over; feature flags are not carried over and each pod\'s settings are honored. For subsequent masking, verify that the target is non-production, is at least 11.13.24.10.0-11000, and its release has an ingested DM template. Multiple masking runs are unsupported. Do not assume TEST labeling alone establishes every masking prerequisite. [doc-0095] [doc-0092]',
 [('Refresh a TEST target from Production; take a target backup before scheduling.',[1],True),('Match source/target BusinessApp versions, patches, cloud-native state and relevant identity/database/service integration.',[1],True),('Additional language packs do not carry over; feature flags retain individual pod settings.',[1],True),('Masking requires a non-production pod at or above 11.13.24.10.0-11000 and an ingested release DM template.',[2],True),('Multiple masking runs are unsupported.',[2],True)],
 ['P2T automatically copies the source feature flags and extra language packs.','A refresh removes the masking template/version checks.','A production target is eligible for Data Masking.'],
 [ev('doc-0095','### Uptake P2T'),ev('doc-0092','Support Data Masking as post pod action','More details on Data Making:')])

add('PH-G14','multi_passage',
 'After provisioning an IdentityService pod, how do I start BuildAcceptanceTests and track them, and should I expect my manually enabled MFA setting to remain enabled during that run?',
 'On the provisioned pod\'s Preflightpod page, use Run Tests at the bottom right and submit the BuildAcceptanceTests for your LOB. AcceptanceTests Label, Order prop and Git branch should auto-populate. Submission creates a new PipelineHub ID; track it on the pod\'s Preflightpod page through queued, running, and the documented SUCCSS success state. BuildAcceptanceTests automatically disable MFA because they cannot run with MFA on, regardless of provisioning or post-provisioning choice. The excerpt does not promise MFA is re-enabled after the run. [doc-0096] [doc-0098]',
 [('Run Tests is available after provisioning from the Preflightpod page and submits tests for the user\'s LOB.',[1],True),('AcceptanceTests Label, Order prop and Git branch auto-populate.',[1],False),('A new PipelineHub ID tracks the job; queued progresses to running and success (spelled SUCCSS in the source).',[1],True),('BuildAcceptanceTests disable MFA irrespective of earlier MFA choices because they cannot run with it enabled.',[2],True)],
 ['The manual MFA setting remains enabled during BuildAcceptanceTests.','MFA is guaranteed to be automatically re-enabled afterward.'],
 [ev('doc-0096'),ev('doc-0098','4. if BuildAcceptanceTests','Eligible PODS')])

add('PH-G15','multi_passage',
 'We locked a Family pod during maintenance and now want to scale it from XS1 to S2. What must change before the request, and is this a valid scale-out?',
 'Unlock the Family pod through its actions menu before proceeding: a locked pod hides Pod Actions/Pod Add-ons and blocks impacting operations. XS1 to S2 is not a valid scale-out because scale-out keeps the logical size family and changes only the server count, such as XS1 to XS2. Moving logical stamps is a Resize use case, subject to eligibility, correct current/required stamps, and no conflicting pod action. For scale-out/resize, check Active at pod health validation and AVAILABLE at the existing-schedule check. A blocked check should be investigated rather than bypassed. [doc-0088] [doc-0087]',
 [('Unlock via Unlock Pod; a locked Family pod hides the relevant actions.',[1],True),('XS1 to S2 is explicitly invalid for scale-out; same size family and changed count are required.',[2],True),('Changing logical stamp calls for Resize eligibility and correct current/target stamps.',[2],True),('Avoid conflicting actions and verify Active health and AVAILABLE schedule-check states.',[2],True)],
 ['XS1 to S2 is a valid scale-out.','Submit a resize while the pod remains locked or a conflicting action is in progress.'],
 [ev('doc-0088','Available only'),ev('doc-0087','### Scale-out','### Additional Details')])

add('PH-G16','multi_passage',
 'I am copying an active-release pod provisioning record in my LOB, then need to grant SSH access to a team whose group does not exist yet. Which overrides need review, and how should I prepare and select the SSH group?',
 'For an eligible non-EOL record in your LOB, Copy Provisioning Job pre-fills editable fields but excludes CloudNative service, application, and environment-profile overrides; review and set those as needed before submission. For SSH, an admin with ROLE_GRANULAR_SSH_USER_MANAGEMENT must create the LOB-specific group in Granular SSH Group Management with a letters-and-spaces group name, a brief non-sensitive description and the intended LOB, add the intended registered users or valid non-PipelineHub GUIDs, stage them with Add Selected Users, verify membership, and Save Changes. Add Selected Users alone does not save. In the target pod\'s Granular SSH action, select the prepared group from its LOB; at most three groups are allowed. If missing, verify LOB/configuration instead of choosing an unrelated group. [doc-0091] [doc-0094]',
 [('Copy requires the user\'s LOB association and a non-EOL release.',[1],True),('CloudNative service, application and environment-profile overrides are excluded from prefilling and must be reviewed.',[1],True),('A group-management admin needs ROLE_GRANULAR_SSH_USER_MANAGEMENT; create/update the correct LOB-specific group.',[2],True),('Use a group name containing letters and spaces only, a brief non-sensitive description and the intended LOB.',[2],True),
 ('Add intended registered users or valid comma-separated non-PipelineHub GUIDs; stage, review, then Save Changes.',[2],True),('Select the group in Granular SSH for the target pod\'s LOB; maximum three groups and no unrelated substitute.',[2],True)],
 ['All CloudNative overrides copy automatically.','Add Selected Users alone saves the group.','Grant the group-management role as an undocumented requirement for every pod-action submitter.','More than three SSH groups can be selected in one action.'],
 [ev('doc-0091','Scope','### Use Case 2:'),ev('doc-0094','Purpose')])

add('PH-G17','ambiguous_clarification',
 'Can I restore this pod to the version we used before?',
 'Which version is the pod on now, which version do you want to restore to, has it successfully completed an upgrade, and which backups are available? The guide distinguishes same-version restore from n-1 restore after a successful upgrade; these details are needed before choosing a procedure. [doc-0089]',
 [('Ask for current and requested versions.',[1],True),('Ask whether the upgrade completed successfully and whether an appropriate backup exists.',[1],True),('Explain conditionally that same-version and post-successful-upgrade n-1 restore are distinct.',[1],False)],
 ['Assume any arbitrary older version can be restored.','Give an unconditional n-1 restore procedure before checking upgrade/backup details.','Escalate immediately instead of asking for the missing details.'],
 [ev('doc-0089','### PipelineHub UI support for Restore','### PipelineHub refresh button')],
 status='needs_clarification',clarification_slots=['current pod version','requested restore version','successful upgrade status','available backup version(s)'])

add('PH-G18','ambiguous_clarification',
 'Which TestRunner provisioning method should I choose for my pod?',
 'What architecture, database platform and pod size are you using, and do you need TestRunner during provisioning to validate the Marketing SKU or can it be added afterward? Also confirm any FMK/CDP requirement. The documented post-provisioning whitelist method supports ARM/x86, EXA/ManagedDatabase and S/M/L; the Marketing SKU method is documented for x86+EXA, S only, with FMK+CDP selected and CDP quota. Do not choose until the pod requirements are known. [doc-0137]',
 [('Ask for architecture, database platform and size.',[1],True),('Ask whether during-provisioning Marketing SKU validation is required or post-provisioning deployment is acceptable; establish FMK/CDP needs.',[1],True),('Do not select a method unconditionally without these details.',[1],True)],
 ['Treat roadmap ETA statements as delivered support.','Recommend the Marketing SKU method for ARM/ManagedDatabase or M/L/XS sizes as already supported.'],
 [ev('doc-0137','Method 1','FYI Side Note')],
 status='needs_clarification',clarification_slots=['ARM or x86','EXA or ManagedDatabase','pod size','Marketing SKU validation / deployment timing','FMK/CDP requirement and quota'])

add('PH-G19','unsupported_escalation',
 'Give me the PipelineHub settings and YAML to generate SLSA Level 3 provenance and enforce Sigstore Cosign signature verification before deployment.',
 'I cannot verify a PipelineHub SLSA Level 3/Cosign configuration from the provided documentation. Please have a PipelineHub support or security engineer confirm whether this integration is supported and provide the approved configuration. No ticket has been submitted. This is a documentation gap, not evidence that the product lacks the capability.',
 [('State that the requested SLSA Level 3/Cosign configuration cannot be verified from the available corpus.',[],True),('Route to human support/security review without inventing YAML or settings.',[],True),('Do not claim a ticket was submitted or conclude the capability does not exist.',[],True)],
 ['Invent a PipelineHub SLSA/Cosign setting, YAML key, endpoint, or supported compliance level.','Treat generic signing or service provenance mentions as proof of this named integration.','Claim the integration is definitively unsupported by the product.'],[],status='needs_human_review',absence_terms=['SLSA','Sigstore','Cosign','in-toto'])

add('PH-G20','unsupported_escalation',
 'How do I configure a native Jira Service Management connector so a failed PipelineHub deployment automatically opens an incident? List its required API scopes and setup steps.',
 'The provided corpus does not establish a native Jira Service Management incident connector, its setup procedure, or required API scopes. Ask a PipelineHub support/integration engineer to confirm the supported approach. Do not invent connector steps or claim an incident has been created. Issue-tracker links in documentation are not evidence of such an integration.',
 [('Say the named connector, setup steps and API scopes are not established by this corpus.',[],True),('Request human support/integration review rather than inventing configuration.',[],True),('Do not claim the connector is absent from the product or that an incident/ticket was created.',[],True)],
 ['Invent a native connector screen, URL, API scope, webhook contract, or incident submission.','Use ordinary issue-tracker references as proof of an automatic Jira Service Management connector.'],[],status='needs_human_review',absence_terms=['Jira Service Management','ServiceNow','incident ticket','automatic ticket','incident connector'])

counts=Counter(r['category'] for r in records)
assert counts=={'exact_match':6,'semantic_troubleshooting':6,'multi_passage':4,'ambiguous_clarification':2,'unsupported_escalation':2}
assert len(set(r['id'] for r in records))==20
norm=lambda s:re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
prior_n={norm(q) for q in prior}
assert not any(norm(r['question']) in prior_n for r in records)
for r in records:
    for e in r['supporting_evidence']:
        assert docs[e['document_id']]['content'][e['char_start']:e['char_end']]==e['quote']
        assert 'deprecated' not in e['title'].lower()
        assert docs[e['document_id']]['content'].strip()
    if r['category']=='multi_passage':assert len(r['required_evidence_groups'])>=2
    for p in r['required_answer_points']:
        assert all(x in {e['evidence_id'] for e in r['supporting_evidence']} for x in p['evidence_ids'])

absence_audit={}
for r in records:
    if not r['absence_search_terms']:continue
    hits={term:[d['id'] for d in docs.values() if term.lower() in d['content'].lower()] for term in r['absence_search_terms']}
    assert all(not x for x in hits.values())
    absence_audit[r['id']]={'searched_documents':len(docs),'usable_documents':sum(bool(d['content'].strip()) for d in docs.values()),'case_insensitive_literal_search':hits,'interpretation':'No occurrences of these named capabilities. This supports a corpus-coverage gap only; absence of keywords alone is not proof about product capabilities. Related sections were reviewed when choosing the cases.'}

metadata={
 'dataset_name':'PipelineHub Golden Evaluation Dataset','version':'1.0','authored_date':'2026-10-17',
 'corpus_sha256':hashlib.sha256(raw).hexdigest(),'corpus_dataset_version':corpus['dataset_version'],
 'corpus_source':'PipelineHub-corpus.json (canonical adapted corpus)',
 'case_count':20,'category_counts':dict(counts),'expected_status_counts':dict(Counter(r['expected_status'] for r in records)),
 'split':'frozen evaluation holdout; use separate cases for tuning',
 'bot_executed_on_these_questions':False,'scores':None,
 'review_status':'Source spans mechanically verified; authored annotations require user/SME review before claiming independently validated gold.',
 'independence':'Questions and labels were authored directly from the canonical corpus without querying the bot, retriever, or generation provider. Existing persisted development questions, unit-test strings and visible demo prompts were screened for duplicates. Prior topics (overview, OAuth registration, generic failure, project pausing and uptime/service credits) were excluded. This is a new case set, not an external blind benchmark; same project assistant authored it and unrecorded historical prompts cannot be audited.',
 'source_overlap_policy':'Documents may also have appeared incidentally in earlier retrievals. Independence means different question/intent/answer cases, not a document-disjoint corpus split.',
 'source_limitations':['All answers describe adapted illustrative documentation, not verified executable platform behavior.','example.com URLs are placeholders; external linked content/media/attachments were not used.','No deprecated-titled document is used as gold support; pages with moved-content notices are labeled only as claims of this frozen corpus.','Accept semantically equivalent answers and valid alternative supporting corpus passages.'],
 'classification_rules':{'exact_match':'Question explicitly names a setting, role, identifier, component or compatibility row; scoring still uses semantic correctness, not whole-string match.','semantic_troubleshooting':'Symptom or intent is expressed in user language and must be mapped to documented behavior; no fabricated diagnosis.','multi_passage':'Complete answer requires both listed evidence groups, not merely two citations to one fact.','ambiguous_clarification':'Missing user details prevent selecting the supported procedure; ask targeted questions.','unsupported_escalation':'Requested named capability or configuration is not established by the corpus; explain the evidence gap and route to human review.'},
 'scoring':{'answerable_cases':16,'all_cases':20,'point_coverage':'Satisfied required_answer_points / total required points. Paraphrases accepted; evaluate evidence support, not keyword occurrence.','case_pass':'Expected route AND all required points AND no unacceptable claim AND valid supporting citations for factual documentation claims. Critical fields flag particularly consequential omissions; all listed required points still count for completeness.','faithfulness':'Supported factual claims / all factual claims in generated answers; inspect cited evidence manually. A valid document ID alone is insufficient.','retrieval':'For the 16 answerable cases, report document-group recall@k; score all required evidence groups for multi-passage questions. Accept alternative passages only after verifying they support the criterion. Do not score unsupported cases against invented relevant documents.','simulated_first_contact_resolution_all':'Correct, complete, cited answers without clarification/human intervention divided by all 20 queries; maximum possible on this designed mix is 16/20 (80%).','simulated_first_contact_resolution_answerable':'Correct, complete, cited answers divided by the 16 answerable queries; report separately with explicit denominator.','routing_accuracy':'Correct answered/needs_clarification/needs_human_review status divided by 20. Clarification or escalation can pass routing but is not a resolution.','latency':'Record end-to-end time for each actual run; do not populate measurements before execution.','holdout_policy':'Freeze labels before running keyword/semantic/hybrid modes. Do not feed reference answers into generation or ingest evaluation files. Any changes after seeing results create a new version and require a fresh holdout for an unbiased final score.'},
}
package={'metadata':metadata,'cases':records}
json_path=OUT/'pipelinehub-golden-v1.json';json_path.write_text(json.dumps(package,indent=2,ensure_ascii=False)+'\n')
(OUT/'pipelinehub-questions-v1.jsonl').write_text(''.join(json.dumps({'id':r['id'],'question':r['question']},ensure_ascii=False)+'\n' for r in records))
columns=['id','category','question','expected_status','reference_answer','required_answer_points','unacceptable_claims','gold_document_ids','required_evidence_groups','clarification_slots']
with (OUT/'pipelinehub-golden-v1.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
    for r in records:
        row={k:r[k] for k in columns}
        for k in ['required_answer_points','unacceptable_claims','gold_document_ids','required_evidence_groups','clarification_slots']:row[k]=json.dumps(row[k],ensure_ascii=False)
        w.writerow(row)
comparisons=[]
for r in records:
    near=max(prior,key=lambda q:SequenceMatcher(None,norm(r['question']),norm(q)).ratio())
    comparisons.append({'id':r['id'],'nearest_existing_question':near,'character_similarity':round(SequenceMatcher(None,norm(r['question']),norm(near)).ratio(),3),'exact_duplicate':False})
audit={'corpus_sha256':metadata['corpus_sha256'],'checked_files':checked,'existing_questions_and_test_prompts':sorted(prior),'near_duplicate_screen':comparisons,'intent_review':'Authored cases avoid the known prior intents; similarity is a screening aid, not proof of semantic independence.','absence_checks':absence_audit,'evidence_spans_verified':sum(len(r['supporting_evidence']) for r in records),'classification_counts_verified':dict(counts),'bot_or_retrieval_calls':0,'human_review_pending':True}
(OUT/'dataset-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
manifest={'dataset_version':'1.0','frozen_date':'2026-10-17','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [json_path,OUT/'pipelinehub-questions-v1.jsonl',OUT/'pipelinehub-golden-v1.csv']},'corpus_sha256':metadata['corpus_sha256'],'bot_executed':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'cases':len(records),'categories':counts,'evidence_spans_verified':audit['evidence_spans_verified'],'prior_prompts_screened':len(prior),'corpus_hash':metadata['corpus_sha256'],'bot_calls':0},indent=2))
