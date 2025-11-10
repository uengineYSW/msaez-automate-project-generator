"""
RequirementsMappingGenerator - LangGraph 워크플로우
Bounded Context별 관련 요구사항 매핑 및 추출
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json
from datetime import datetime
import sys
from pathlib import Path
import re

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_generator.config import Config
from src.project_generator.utils.logging_util import LoggingUtil


class RequirementsMappingState(TypedDict):
    """Requirements Mapping 상태"""
    # Inputs
    bounded_context: Dict  # BC 정보 (name, alias, aggregates, events, importance, implementationStrategy)
    requirement_chunk: Dict  # 요구사항 청크 (userStory, events, ddl 등)
    
    # Outputs
    relevant_requirements: List[Dict]  # [{type: "userStory"|"DDL"|"Event", refs: [...]}]
    
    # Metadata
    progress: int
    logs: Annotated[List[Dict], "append"]
    is_completed: bool
    error: str


class RequirementsMappingWorkflow:
    """
    Requirements Mapping 워크플로우
    특정 Bounded Context에 관련된 요구사항을 찾아 매핑
    """
    
    def __init__(self):
        # Structured Output Schema (Frontend의 Zod schema와 동일)
        self.response_schema = {
            "type": "object",
            "title": "RequirementsMappingResponse",
            "description": "Relevant requirements for a bounded context with traceability",
            "properties": {
                "relevantRequirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["userStory", "DDL", "Event"]
                            },
                            "refs": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": ["number", "string"]}
                                }
                            }
                        },
                        "required": ["type", "refs"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["relevantRequirements"],
            "additionalProperties": False
        }
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        self.llm_structured = self.llm.with_structured_output(
            self.response_schema,
            strict=True
        )
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        workflow = StateGraph(RequirementsMappingState)
        
        workflow.add_node("map_requirements", self.map_requirements)
        workflow.add_node("finalize", self.finalize)
        
        workflow.set_entry_point("map_requirements")
        workflow.add_edge("map_requirements", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def map_requirements(self, state: RequirementsMappingState) -> Dict:
        """
        BC에 관련된 요구사항 매핑
        """
        bounded_context = state["bounded_context"]
        requirement_chunk = state["requirement_chunk"]
        
        bc_name = bounded_context.get('name', 'Unknown')
        is_ui_bc = (bc_name == 'ui')
        
        LoggingUtil.info("RequirementsMapper", f"📝 Mapping requirements for BC: {bc_name}{' (UI BC)' if is_ui_bc else ''}")
        
        # 요구사항 텍스트 추출 및 라인 번호 추가
        requirements_text = self._get_line_numbered_requirements(requirement_chunk)
        
        if not requirements_text or requirements_text.strip() == "":
            LoggingUtil.info("RequirementsMapper", "⚠️ Empty requirements text, returning empty result")
            return {
                "relevant_requirements": [],
                "progress": 80,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": f"No requirements text to map for {bc_name}"
                }]
            }
        
        # 언어 감지
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in requirements_text[:500])
        language = "Korean" if has_korean else "English"
        
        # Frontend와 동일한 프롬프트 구성
        prompt = self._build_prompt(bounded_context, requirements_text, language, is_ui_bc)
        
        try:
            # Structured Output 사용
            result_data = self.llm_structured.invoke(prompt)
            
            relevant_reqs = result_data.get('relevantRequirements', [])
            
            # Frontend와 동일한 순서:
            # 1. _wrapRefArrayToModel: refs를 한 겹 더 감싸기 (DDL 특별 처리 없음!)
            for req in relevant_reqs:
                if req.get('refs') and len(req['refs']) > 0:
                    req['refs'] = [req['refs']]
            
            # 2. sanitizeAndConvertRefs: refs 변환
            # LLM이 반환한 refs 형식: [[startLine, "phrase"], [endLine, "phrase"]]
            # 이를 [[[startLine, startCol], [endLine, endCol]]] 형식으로 변환해야 함
            sanitized_reqs = self._sanitize_and_convert_refs(relevant_reqs, requirements_text)
            
            # 3. getReferencedUserRequirements: text 필드 추가
            enriched_reqs = self._add_text_to_requirements(sanitized_reqs, requirement_chunk)
            
            LoggingUtil.info("RequirementsMapper", f"📝 After text extraction: {len(enriched_reqs)} requirements")
            
            LoggingUtil.info("RequirementsMapper", f"✅ Found {len(enriched_reqs)} relevant requirements for {bc_name}")
            
            return {
                "relevant_requirements": enriched_reqs,
                "progress": 80,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Mapped {len(enriched_reqs)} requirements to {bc_name}"
                }]
            }
            
        except Exception as e:
            error_msg = f"Failed to map requirements: {str(e)}"
            LoggingUtil.exception("RequirementsMapper", "Mapping failed", e)
            
            return {
                "relevant_requirements": [],
                "error": error_msg,
                "progress": 80,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": error_msg
                }]
            }
    
    def finalize(self, state: RequirementsMappingState) -> Dict:
        """최종 결과 정리"""
        return {
            "is_completed": True,
            "progress": 100,
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "message": "Requirements mapping completed"
            }]
        }
    
    def _build_prompt(self, bounded_context, requirements_text, language, is_ui_bc=False) -> str:
        """프롬프트 구성 (Frontend와 동일)"""
        
        bc_name = bounded_context.get('name', '')
        bc_alias = bounded_context.get('alias', '')
        bc_importance = bounded_context.get('importance', '')
        bc_strategy = bounded_context.get('implementationStrategy', '')
        bc_aggregates = json.dumps(bounded_context.get('aggregates', []), ensure_ascii=False, indent=2)
        bc_events = json.dumps(bounded_context.get('events', []), ensure_ascii=False, indent=2)
        
        # UI BC에 대한 특별 프롬프트
        ui_specific_prompt = ""
        if is_ui_bc:
            ui_specific_prompt = """
            <section id="ui_bounded_context_rules">
                <title>SPECIAL INSTRUCTIONS FOR UI BOUNDED CONTEXT</title>
                <description>For UI bounded context, apply strict filtering rules to focus only on user interface concerns.</description>
                
                <subsection id="ui_mapping_scope">
                    <title>ONLY Map Requirements Related To:</title>
                    <item>User interface elements (buttons, forms, tables, charts, etc.)</item>
                    <item>Non-functional requirements</item>
                    <item>Screen layouts and navigation</item>
                    <item>User interactions (clicks, inputs, selections, etc.)</item>
                    <item>Visual design and styling requirements</item>
                    <item>User experience (UX) flows and user journeys</item>
                    <item>Display of data and information presentation</item>
                    <item>Responsive design and accessibility requirements</item>
                    <item>Frontend validation rules and error messages</item>
                    <item>User feedback and notifications</item>
                    <item>Screen transitions and animations</item>
                </subsection>
                
                <subsection id="ui_exclusions">
                    <title>MUST NOT Map:</title>
                    <exclusion>Events or Actors</exclusion>
                    <exclusion>Functional requirements</exclusion>
                    <exclusion>Business logic or backend processes</exclusion>
                    <exclusion>Data processing or calculations</exclusion>
                    <exclusion>API calls or data fetching logic</exclusion>
                    <exclusion>Database operations</exclusion>
                    <exclusion>Server-side validation</exclusion>
                    <exclusion>Authentication/authorization logic</exclusion>
                    <exclusion>Data persistence or storage</exclusion>
                </subsection>
                
                <critical_rule>Focus ONLY on what users see and interact with on the screen. If the requirement is not related to the UI, return empty array.</critical_rule>
            </section>
"""
        
        # Frontend의 __buildTaskGuidelinesPrompt()와 동일한 프롬프트
        prompt = f"""<instruction>
    <core_instructions>
        <title>Requirements Mapping Task</title>
        <task_description>Your task is to analyze the provided requirements chunk and determine if it contains any content relevant to the specified Bounded Context. You must identify relevant requirements and provide precise traceability references.</task_description>
        
        <input_description>
            <title>You will be given:</title>
            <item id="1">**Bounded Context Information:** Name, alias, implementation strategy, importance, aggregates, and events</item>
            <item id="2">**Requirements Chunk:** Either text requirements with line numbers, analysis results with actors/events, or DDL schemas</item>
        </input_description>

        <guidelines>
            <title>Requirements Mapping Guidelines</title>
            
            <section id="relevance_assessment">
                <title>Relevance Assessment Criteria</title>
                <rule id="1">**Direct References:** Look for explicit mentions of the Bounded Context's name, alias, or aggregates</rule>
                <rule id="2">**Business Processes:** Identify workflows that this Bounded Context is responsible for</rule>
                <rule id="3">**Data Structures:** Match entities that align with the Bounded Context's aggregates</rule>
                <rule id="4">**Event Relationships:** Find events that are published or consumed by this Bounded Context</rule>
                <rule id="5">**User Stories:** Identify functionality within this Bounded Context's domain</rule>
                <rule id="6">**DDL Analysis:** Consider DDL tables whose field names (like order_id, product_id) relate to the context's aggregates, even if table names don't directly match</rule>
            </section>

            <section id="reference_precision">
                <title>Reference Traceability Requirements</title>
                <rule id="1">**Refs Format:** Each ref must contain [[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]</rule>
                <rule id="2">**Minimal Phrases:** Use 1-2 word phrases that uniquely identify the position in the line</rule>
                <rule id="3">**Shortest Possible:** Use the shortest possible phrase that can locate the specific part of requirements</rule>
                <rule id="4">**Valid Line Numbers:** Only reference lines that exist in the provided content</rule>
                <rule id="5">**Precision:** Point to exact line numbers and phrases for accurate traceability</rule>
            </section>

            <section id="decision_strategy">
                <title>Decision Strategy</title>
                <rule id="1">**Context Awareness:** Consider the Bounded Context's implementation strategy and importance</rule>
                <rule id="2">**Indirect Relationships:** Look for indirect relationships through aggregates and events</rule>
                <rule id="3">**Domain Alignment:** Include content if it's part of the same business domain</rule>
                <rule id="4">**Inclusion Bias:** When in doubt, err on the side of inclusion if the relationship is plausible</rule>
            </section>

{ui_specific_prompt}        </guidelines>

        <refs_format_example>
            <title>Example of refs Format</title>
            <description>If requirements contain:</description>
            <example_requirements>
<1>Users can browse and purchase products</1>
<2>Payment processing with multiple providers</2>
<3>Order tracking and status updates</3>
<4>Inventory management for products</4>
            </example_requirements>
            <example_refs>
- For "Order" bounded context referencing line 1 → refs: [[1, "Users"], [1, "purchase"]]
- For "Order" bounded context referencing line 3 → refs: [[3, "Order"], [3, "tracking"]]
            </example_refs>
        </refs_format_example>
    </core_instructions>
    
    <output_format>
        <title>JSON Output Format</title>
        <description>The output must be a JSON object structured as follows:</description>
        <schema>
{{
    "relevantRequirements": [
        {{
            "type": "userStory" || "DDL" || "Event",
            "refs": [[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]
        }}
    ]
}}
        </schema>
        <field_requirements>
            <requirement id="1">Return empty array if no relevant content is found</requirement>
            <requirement id="2">Each relevant item must specify type ("userStory", "DDL", or "Event")</requirement>
            <requirement id="3">Provide accurate line number references with contextual phrases</requirement>
            <requirement id="4">Refs must use minimal phrases to identify exact locations</requirement>
        </field_requirements>
    </output_format>
</instruction>

<inputs>
<bounded_context>
  <name>{bc_name}</name>
  <alias>{bc_alias}</alias>
  <importance>{bc_importance}</importance>
  <implementation_strategy>{bc_strategy}</implementation_strategy>
  <aggregates>{bc_aggregates}</aggregates>
  <events>{bc_events}</events>
</bounded_context>

<requirements_chunk>
{requirements_text}
</requirements_chunk>
</inputs>

<language_guide>Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.</language_guide>"""

        return prompt
    
    def _sanitize_and_convert_refs(self, requirements, requirements_text) -> list:
        """
        Frontend의 RefsTraceUtil.sanitizeAndConvertRefs와 동일한 역할
        refs 형식 변환: [[startLine, "phrase"], [endLine, "phrase"]] → [[[startLine, startCol], [endLine, endCol]]]
        """
        lines = requirements_text.split('\n')
        
        # 라인 번호 맵 생성: 절대 라인 번호 → 배열 인덱스
        line_number_map = {}
        for idx, line in enumerate(lines):
            match = re.match(r'^<(\d+)>(.*)</\d+>$', line)
            if match:
                line_num = int(match.group(1))
                line_number_map[line_num] = idx
        
        sanitized = []
        
        for req in requirements:
            req_copy = dict(req)
            refs = req.get('refs', [])
            
            if not refs:
                sanitized.append(req_copy)
                continue
            
            converted_refs = []
            for ref in refs:
                if not isinstance(ref, list):
                    continue
                
                # ref 형식 확인: 이미 _wrapRefArrayToModel로 감싸진 상태
                # refs = [[[startLine, "phrase"], [endLine, "phrase"]]]
                # ref = [[startLine, "phrase"], [endLine, "phrase"]]
                
                if len(ref) == 0:
                    continue
                
                # 실제 ref 추출 (중첩 제거)
                actual_ref = ref
                if len(ref) == 1 and isinstance(ref[0], list):
                    # 중첩된 경우: [[[203, "CREATE"], [220, ";"]]] → [[203, "CREATE"], [220, ";"]]
                    actual_ref = ref[0]
                
                if not isinstance(actual_ref, list) or len(actual_ref) < 2:
                    continue
                
                # [[203, "CREATE"], [220, ";"]] 형식 확인
                if isinstance(actual_ref[0], list) and isinstance(actual_ref[1], list):
                    start_ref = actual_ref[0]
                    end_ref = actual_ref[1]
                elif isinstance(actual_ref[0], (int, float)) and isinstance(actual_ref[1], (int, float, str)):
                    # [203, "CREATE"] 형식 (단일 ref)
                    start_ref = actual_ref
                    end_ref = actual_ref
                else:
                    # 형식이 맞지 않으면 건너뛰기
                    LoggingUtil.warning("RequirementsMapper", f"Skipping invalid ref format: {ref}")
                    continue
                
                # 라인 번호와 phrase 추출
                # ref 형식: [[startLine, "phrase"], [endLine, "phrase"]]
                # 또는 [startLine, "phrase"] (단일 요소)
                
                # start_ref 처리
                if isinstance(start_ref, list) and len(start_ref) >= 2:
                    # [203, "CREATE"] 형식
                    start_line = start_ref[0]
                    start_phrase = start_ref[1] if len(start_ref) > 1 else ""
                elif isinstance(start_ref, list) and len(start_ref) == 1:
                    # [203] 형식
                    start_line = start_ref[0]
                    start_phrase = ""
                elif isinstance(start_ref, (int, float)):
                    # 203 형식
                    start_line = start_ref
                    start_phrase = ""
                else:
                    # 문자열이면 건너뛰기
                    continue
                
                # end_ref 처리
                if isinstance(end_ref, list) and len(end_ref) >= 2:
                    # [220, ";"] 형식
                    end_line = end_ref[0]
                    end_phrase = end_ref[1] if len(end_ref) > 1 else ""
                elif isinstance(end_ref, list) and len(end_ref) == 1:
                    # [220] 형식
                    end_line = end_ref[0]
                    end_phrase = ""
                elif isinstance(end_ref, (int, float)):
                    # 220 형식
                    end_line = end_ref
                    end_phrase = ""
                else:
                    # 문자열이면 건너뛰기
                    continue
                
                # 라인 번호가 숫자인지 확인
                try:
                    start_line_num = int(start_line)
                    end_line_num = int(end_line)
                except (ValueError, TypeError):
                    # 숫자가 아니면 건너뛰기
                    LoggingUtil.warning("RequirementsMapper", f"Skipping invalid ref: start_line={start_line}, end_line={end_line}")
                    continue
                
                # 라인 번호 맵에서 인덱스 찾기
                if start_line_num not in line_number_map or end_line_num not in line_number_map:
                    LoggingUtil.warning("RequirementsMapper", f"Line number not found in map: start_line={start_line_num}, end_line={end_line_num}, available_lines={sorted(line_number_map.keys())[:10]}...")
                    continue
                
                start_idx = line_number_map[start_line_num]
                end_idx = line_number_map[end_line_num]
                
                if 0 <= start_idx < len(lines) and 0 <= end_idx < len(lines):
                    start_line_text = lines[start_idx]
                    end_line_text = lines[end_idx]
                    
                    # XML 태그 제거: <130>text</130> → text (프론트엔드와 동일한 방식)
                    start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_text)
                    if start_match:
                        start_line_text = start_match.group(1)
                    end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_text)
                    if end_match:
                        end_line_text = end_match.group(1)
                    
                    # phrase의 위치 찾기 (컬럼 위치)
                    # 프론트엔드의 tryRelocate 로직과 유사: phrase를 찾을 수 없으면 ±5 라인 탐색
                    def find_phrase_position(line_num, phrase, current_line_text, is_end=False):
                        """phrase를 찾고, 없으면 ±5 라인 범위에서 탐색"""
                        if not phrase or not isinstance(phrase, str) or not phrase.strip():
                            return None
                        
                        # 현재 라인에서 찾기 (이미 추출된 텍스트 사용)
                        if current_line_text and phrase in current_line_text:
                            pos = current_line_text.find(phrase)
                            return pos + (len(phrase) if is_end else 0) + 1  # 1-based
                        
                        # 현재 라인에서 못 찾으면 ±5 라인 범위에서 탐색
                        # 라인 번호 목록 정렬
                        sorted_line_nums = sorted(line_number_map.keys())
                        current_line_idx = None
                        for idx, ln in enumerate(sorted_line_nums):
                            if ln == line_num:
                                current_line_idx = idx
                                break
                        
                        if current_line_idx is None:
                            LoggingUtil.warning("RequirementsMapper", f"Line {line_num} not found in line_number_map")
                            return None
                        
                        for offset in range(1, 6):
                            # 위쪽 라인 확인
                            if current_line_idx - offset >= 0:
                                check_line = sorted_line_nums[current_line_idx - offset]
                                check_idx = line_number_map[check_line]
                                if 0 <= check_idx < len(lines):
                                    check_content = lines[check_idx]
                                    match = re.match(rf'^<{check_line}>(.*)</{check_line}>$', check_content)
                                    if match:
                                        check_content = match.group(1)
                                    if phrase in check_content:
                                        pos = check_content.find(phrase)
                                        return pos + (len(phrase) if is_end else 0) + 1
                            
                            # 아래쪽 라인 확인
                            if current_line_idx + offset < len(sorted_line_nums):
                                check_line = sorted_line_nums[current_line_idx + offset]
                                check_idx = line_number_map[check_line]
                                if 0 <= check_idx < len(lines):
                                    check_content = lines[check_idx]
                                    match = re.match(rf'^<{check_line}>(.*)</{check_line}>$', check_content)
                                    if match:
                                        check_content = match.group(1)
                                    if phrase in check_content:
                                        pos = check_content.find(phrase)
                                        return pos + (len(phrase) if is_end else 0) + 1
                        
                        # phrase를 찾지 못하면 기본 위치 사용
                        return None
                    
                    # start_phrase 처리
                    start_col_result = find_phrase_position(start_line_num, start_phrase, start_line_text, is_end=False)
                    if start_col_result is not None:
                        start_col = start_col_result
                    else:
                        # phrase를 찾을 수 없으면 라인 시작부터
                        start_col = 1
                    
                    # end_phrase 처리
                    end_col_result = find_phrase_position(end_line_num, end_phrase, end_line_text, is_end=True)
                    if end_col_result is not None:
                        end_col = end_col_result
                    else:
                        # phrase를 찾을 수 없으면 라인 끝까지
                        end_col = len(end_line_text) if end_line_text else 1
                    
                    # 최소값은 1 (1-based column)
                    start_col = max(1, start_col)
                    end_col = max(1, end_col)
                    
                    converted_refs.append([[start_line_num, start_col], [end_line_num, end_col]])
            
            if converted_refs:
                req_copy['refs'] = converted_refs
            else:
                req_copy['refs'] = []
            
            sanitized.append(req_copy)
        
        return sanitized
    
    def _get_line_numbered_requirements(self, requirement_chunk) -> str:
        """라인 번호가 추가된 요구사항 반환 (Frontend의 _parseRequirements와 동일)"""
        
        # type이 없으면 text 필드 사용 (일반 요구사항 텍스트)
        if not requirement_chunk.get('type'):
            text = requirement_chunk.get('text', '')
            start_line = requirement_chunk.get('startLine', 1)
            
            if text:
                lines = text.split('\n')
                numbered_lines = [f"<{i+start_line}>{line}</{i+start_line}>" for i, line in enumerate(lines)]
                return '\n'.join(numbered_lines)
            
            return ""
        
        # type이 "analysisResult"이면 events를 Markdown으로 변환
        if requirement_chunk.get('type') == 'analysisResult':
            events = requirement_chunk.get('events', [])
            if not events:
                return ""
            
            markdown = '### Events\n\n'
            for event in events:
                markdown += self._make_event_markdown(event) + '\n'
            
            # 라인 번호 추가
            lines = markdown.strip().split('\n')
            numbered_lines = [f"<{i+1}>{line}</{i+1}>" for i, line in enumerate(lines)]
            return '\n'.join(numbered_lines)
        
        return ""
    
    def _make_event_markdown(self, event) -> str:
        """Event를 Markdown으로 변환 (Frontend와 동일)"""
        markdown = f"**Event: {event.get('name', '')} ({event.get('displayName', '')})**\n"
        markdown += f"- **Actor:** {event.get('actor', '')}\n"
        markdown += f"- **Description:** {event.get('description', '')}\n"
        
        if event.get('inputs') and len(event.get('inputs', [])) > 0:
            markdown += f"- **Inputs:** {', '.join(event['inputs'])}\n"
        
        if event.get('outputs') and len(event.get('outputs', [])) > 0:
            markdown += f"- **Outputs:** {', '.join(event['outputs'])}\n"
        
        return markdown
    
    def _wrap_refs(self, requirements: list) -> list:
        """
        기존 유틸을 그대로 사용하기 위해서 refs 속성을 []로 한번 더 감싸기
        Frontend의 _wrapRefArrayToModel과 동일
        
        DDL 타입은 number-only refs로 변환: [[lineNum, "phrase"]] → [[lineNum, lineNum]]
        """
        wrapped_reqs = []
        for req in requirements:
            wrapped_req = dict(req)
            if wrapped_req.get('refs') and len(wrapped_req['refs']) > 0:
                # DDL 타입은 refs를 number-only로 변환
                if wrapped_req.get('type') == 'DDL':
                    number_only_refs = []
                    for ref in wrapped_req['refs']:
                        if isinstance(ref, list) and len(ref) >= 2:
                            # [lineNum, "phrase"] → [lineNum, lineNum]
                            line_num = ref[0] if isinstance(ref[0], (int, float)) else ref[1]
                            number_only_refs.append([int(line_num), int(line_num)])
                    wrapped_req['refs'] = [number_only_refs]
                else:
                    # 일반 타입: refs를 한 겹 더 감싸기
                    wrapped_req['refs'] = [wrapped_req['refs']]
            wrapped_reqs.append(wrapped_req)
        return wrapped_reqs
    
    def _add_text_to_requirements(self, requirements, requirement_chunk) -> list:
        """
        Requirements에 text 필드 추가 (Frontend와 동일한 형식)
        Frontend의 _processNonUIRequirements/_processUIRequirements 로직 참고
        """
        enriched_reqs = []
        
        # analysisResult 타입 (events 포함)
        if requirement_chunk.get('type') == 'analysisResult':
            events = requirement_chunk.get('events', [])
            
            # LLM이 반환한 refs에서 참조된 라인 번호 수집
            # refs 형식: [[startLine, "phrase"], [endLine, "phrase"]] 또는 [lineNum, "phrase"]
            referenced_lines = set()
            for req in requirements:
                refs = req.get('refs', [])
                for ref in refs:
                    if isinstance(ref, list) and len(ref) > 0:
                        # 첫 번째 요소가 숫자면 단일 라인 참조
                        if isinstance(ref[0], (int, float)):
                            referenced_lines.add(int(ref[0]))
                        # 첫 번째 요소가 리스트면 범위 참조 [[start, ""], [end, ""]]
                        elif isinstance(ref[0], list) and len(ref[0]) > 0:
                            start_line = int(ref[0][0])
                            if len(ref) > 1 and isinstance(ref[1], list) and len(ref[1]) > 0:
                                end_line = int(ref[1][0])
                                for line in range(start_line, end_line + 1):
                                    referenced_lines.add(line)
                            else:
                                referenced_lines.add(start_line)
            
            # 라인 번호로 이벤트 매핑 생성 (Frontend의 eventLineMap과 동일)
            event_line_map = {}
            line_counter = 1
            line_counter += 2  # "### Events" + 빈 줄
            
            for idx, event in enumerate(events):
                event_markdown = self._make_event_markdown(event)
                event_lines = event_markdown.split('\n')
                
                # 이벤트가 차지하는 라인 범위 저장
                start_line = line_counter
                end_line = line_counter + len(event_lines) - 1
                
                for line in range(start_line, end_line + 1):
                    event_line_map[line] = {
                        'index': idx,
                        'event': event
                    }
                
                line_counter = end_line + 2  # 이벤트 사이 빈 줄
            
            # 참조된 라인에 해당하는 이벤트 추출
            # Frontend와 동일: 원본 이벤트의 refs를 사용!
            relevant_events = {}
            for line_num in referenced_lines:
                if line_num in event_line_map:
                    idx = event_line_map[line_num]['index']
                    if idx not in relevant_events:
                        event = event_line_map[line_num]['event']
                        relevant_events[idx] = {
                            "type": "Event",
                            "text": json.dumps(event, ensure_ascii=False, indent=2),
                            "refs": event.get('refs', [])  # 원본 이벤트의 refs 사용!
                        }
            
            enriched_reqs = list(relevant_events.values())
        
        # 일반 텍스트 타입 (DDL 또는 userStory 텍스트)
        else:
            text = requirement_chunk.get('text', '')
            start_line = requirement_chunk.get('startLine', 1)
            lines = text.split('\n')
            
            LoggingUtil.info("RequirementsMapper", f"📄 Text extraction: start_line={start_line}, text_lines={len(lines)}, requirement_type={requirement_chunk.get('type', 'text')}")
            
            # 텍스트에서 refs가 가리키는 내용 추출 (Frontend의 TextTraceUtil.getReferencedUserRequirements와 동일)
            # 프론트엔드: startLineOffset = requirementChunk.startLine - 1
            start_line_offset = start_line - 1
            
            for req_idx, req in enumerate(requirements):
                refs = req.get('refs', [])
                if not refs or len(refs) == 0:
                    LoggingUtil.warning("RequirementsMapper", f"  Req[{req_idx}] ({req.get('type', 'unknown')}): No refs")
                    continue
                
                # refs 형식: [[[startLine, startCol], [endLine, endCol]]] (sanitizeAndConvertRefs 후)
                # 프론트엔드의 getReferencedUserRequirements와 동일하게 처리
                referenced_texts = []
                
                for ref_idx, ref in enumerate(refs):
                    if not isinstance(ref, list) or len(ref) < 2:
                        LoggingUtil.warning("RequirementsMapper", f"  Req[{req_idx}] Ref[{ref_idx}]: Invalid format (not list or < 2)")
                        continue
                    
                    # ref 형식: [[startLine, startCol], [endLine, endCol]]
                    start_pos = ref[0]
                    end_pos = ref[1] if len(ref) > 1 else ref[0]
                    
                    if not isinstance(start_pos, list) or not isinstance(end_pos, list):
                        LoggingUtil.warning("RequirementsMapper", f"  Req[{req_idx}] Ref[{ref_idx}]: Invalid pos format - start_pos={type(start_pos)}, end_pos={type(end_pos)}")
                        continue
                    
                    if len(start_pos) < 2 or len(end_pos) < 2:
                        LoggingUtil.warning("RequirementsMapper", f"  Req[{req_idx}] Ref[{ref_idx}]: Invalid pos length - start_pos={start_pos}, end_pos={end_pos}")
                        continue
                    
                    # Frontend와 동일: destructuring [[startLine, startCol], [endLine, endCol]]
                    start_line_num = int(start_pos[0])
                    start_col = int(start_pos[1])
                    end_line_num = int(end_pos[0])
                    end_col = int(end_pos[1])
                    
                    # 프론트엔드와 동일: sLine = startLine - startLineOffset - 1
                    # startLineOffset = requirementChunk.startLine - 1
                    s_line = start_line_num - start_line_offset - 1
                    s_col = start_col - 1
                    e_line = end_line_num - start_line_offset - 1
                    e_col = end_col - 1
                    
                    LoggingUtil.info("RequirementsMapper", f"  Req[{req_idx}] Ref[{ref_idx}]: line_num={start_line_num}~{end_line_num}, col={start_col}~{end_col}, start_line_offset={start_line_offset}, s_line={s_line}, e_line={e_line}, lines_length={len(lines)}")
                    
                    # 범위 검증
                    if s_line < 0 or e_line >= len(lines) or s_line > e_line:
                        LoggingUtil.warning("RequirementsMapper", f"  Req[{req_idx}] Ref[{ref_idx}]: Invalid line range - s_line={s_line}, e_line={e_line}, lines_length={len(lines)}, start_line={start_line}, start_line_offset={start_line_offset}")
                        continue
                    
                    extracted_text = ''
                    
                    if s_line == e_line:
                        # 같은 줄에서 추출 (inclusive이므로 endCol + 1)
                        if 0 <= s_line < len(lines):
                            line_text = lines[s_line]
                            if 0 <= s_col < len(line_text) and 0 <= e_col < len(line_text):
                                extracted_text = line_text[s_col:e_col + 1]
                    else:
                        # 여러 줄에 걸쳐 추출 (Frontend와 동일)
                        # 시작 줄의 일부
                        if 0 <= s_line < len(lines):
                            start_line_text = lines[s_line]
                            if 0 <= s_col < len(start_line_text):
                                extracted_text = start_line_text[s_col:]
                        
                        # 중간 줄들 전체
                        for i in range(s_line + 1, e_line):
                            if 0 <= i < len(lines):
                                extracted_text += '\n' + lines[i]
                        
                        # 끝 줄의 일부 (inclusive이므로 endCol + 1)
                        if 0 <= e_line < len(lines):
                            end_line_text = lines[e_line]
                            if 0 <= e_col < len(end_line_text):
                                extracted_text += '\n' + end_line_text[:e_col + 1]
                    
                    if extracted_text:
                        referenced_texts.append(extracted_text)
                
                if referenced_texts:
                    # 첫 번째 참조 텍스트 사용 (Frontend와 동일)
                    # 빈 텍스트 체크
                    if referenced_texts[0].strip():
                        enriched_reqs.append({
                            "type": req.get('type', 'userStory'),
                            "text": referenced_texts[0],
                            "refs": refs
                        })
        
        return enriched_reqs
    
    def run(self, inputs: Dict) -> Dict:
        """워크플로우 실행"""
        initial_state: RequirementsMappingState = {
            "bounded_context": inputs.get("bounded_context", {}),
            "requirement_chunk": inputs.get("requirement_chunk", {}),
            "relevant_requirements": [],
            "progress": 0,
            "logs": [],
            "is_completed": False,
            "error": ""
        }
        
        result = self.workflow.invoke(initial_state)
        return result

