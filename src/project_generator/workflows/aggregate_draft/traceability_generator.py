from typing import Any, Dict, List, TypedDict, Union
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re

from project_generator.utils.logging_util import LoggingUtil


class TraceabilityGeneratorState(TypedDict):
    inputs: Dict[str, Any]
    inference: str
    progress: int
    result: Dict[str, Any]
    error: str
    timestamp: str


class TraceabilityGenerator:
    """
    LangGraph workflow for adding traceability (refs) to domain objects.
    Maps domain objects (aggregates, enumerations, valueObjects) to functional requirements.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            streaming=False,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            LoggingUtil.info("TraceabilityGenerator", "Traceability generation started")

            # 1. 입력 추출
            generated_draft_options = input_data.get('generatedDraftOptions', [])
            bounded_context_name = input_data.get('boundedContextName', '')
            functional_requirements = input_data.get('functionalRequirements', '')
            trace_map = input_data.get('traceMap', {})

            # 2. 모든 도메인 객체 추출 (중복 제거)
            all_domain_objects = self._extract_all_domain_objects(generated_draft_options)

            # 3. 필터링된 옵션 생성 (name, alias만)
            filtered_options = self._filter_generated_draft_options(generated_draft_options)

            # 4. 라인 번호 추가
            line_numbered_requirements = self._add_line_numbers(functional_requirements)

            # 5. 프롬프트 생성 및 LLM 호출
            prompt = self._build_prompt(
                filtered_options,
                bounded_context_name,
                line_numbered_requirements,
                all_domain_objects
            )

            response = self.llm.invoke(prompt)
            
            # JSON 파싱
            response_text = response.content if hasattr(response, 'content') else str(response)
            result_data = json.loads(response_text)

            # 6. refs 변환 (phrase → indexes)
            draft_trace_map = self._convert_refs_to_indexes(
                result_data,
                functional_requirements,
                line_numbered_requirements,
                trace_map
            )

            LoggingUtil.info(
                "TraceabilityGenerator",
                f"Traceability completed: {len(draft_trace_map.get('aggregates', []))} aggregates, "
                f"{len(draft_trace_map.get('enumerations', []))} enumerations, "
                f"{len(draft_trace_map.get('valueObjects', []))} valueObjects"
            )

            return {
                'draftTraceMap': draft_trace_map,
                'inference': f'Traceability added for {bounded_context_name}',
                'progress': 100
            }

        except Exception as e:
            LoggingUtil.error("TraceabilityGenerator", f"Failed: {str(e)}")
            return {
                'draftTraceMap': {'aggregates': [], 'enumerations': [], 'valueObjects': []},
                'inference': f'Error: {str(e)}',
                'progress': 100,
                'error': str(e)
            }

    def _extract_all_domain_objects(self, generated_draft_options: List[Dict]) -> Dict[str, List[Dict]]:
        """모든 옵션에서 도메인 객체 추출 (중복 제거)"""
        all_objects = {
            'aggregates': [],
            'enumerations': [],
            'valueObjects': []
        }

        seen_aggregates = set()
        seen_enumerations = set()
        seen_value_objects = set()

        for option in generated_draft_options:
            structure_array = option.get('structure', option)
            if not isinstance(structure_array, list):
                structure_array = [structure_array]

            for aggregate_info in structure_array:
                # Aggregate
                if aggregate_info.get('aggregate') and aggregate_info['aggregate'].get('name'):
                    name = aggregate_info['aggregate']['name']
                    if name not in seen_aggregates:
                        all_objects['aggregates'].append({
                            'name': name,
                            'alias': aggregate_info['aggregate'].get('alias', '')
                        })
                        seen_aggregates.add(name)

                # Enumerations
                if aggregate_info.get('enumerations'):
                    for enumeration in aggregate_info['enumerations']:
                        name = enumeration.get('name')
                        if name and name not in seen_enumerations:
                            all_objects['enumerations'].append({
                                'name': name,
                                'alias': enumeration.get('alias', '')
                            })
                            seen_enumerations.add(name)

                # Value Objects
                if aggregate_info.get('valueObjects'):
                    for value_object in aggregate_info['valueObjects']:
                        name = value_object.get('name')
                        if name and name not in seen_value_objects:
                            all_objects['valueObjects'].append({
                                'name': name,
                                'alias': value_object.get('alias', '')
                            })
                            seen_value_objects.add(name)

        return all_objects

    def _filter_generated_draft_options(self, generated_draft_options: List[Dict]) -> List[List[Dict]]:
        """name, alias만 남기고 필터링"""
        filtered = []
        for option in generated_draft_options:
            filtered_structure = []
            structure = option.get('structure', [])
            for struct in structure:
                filtered_agg_info = {
                    'aggregate': {
                        'name': struct.get('aggregate', {}).get('name', ''),
                        'alias': struct.get('aggregate', {}).get('alias', '')
                    },
                    'enumerations': [
                        {'name': e.get('name', ''), 'alias': e.get('alias', '')}
                        for e in struct.get('enumerations', [])
                    ],
                    'valueObjects': [
                        {'name': v.get('name', ''), 'alias': v.get('alias', '')}
                        for v in struct.get('valueObjects', [])
                    ]
                }
                filtered_structure.append(filtered_agg_info)
            if filtered_structure:
                filtered.append(filtered_structure)
        return filtered

    def _add_line_numbers(self, text: str, start_line: int = 1, use_xml: bool = True) -> str:
        """텍스트에 라인 번호 추가 (프론트엔드 TextTraceUtil.addLineNumbers와 동일)"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            if use_xml:
                numbered_lines.append(f'<{line_num}>{line}</{line_num}>')
            else:
                numbered_lines.append(f'{line_num}|{line}')
        return '\n'.join(numbered_lines)

    def _build_prompt(
        self,
        filtered_options: List[List[Dict]],
        bounded_context_name: str,
        line_numbered_requirements: str,
        all_domain_objects: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """프론트엔드와 동일한 프롬프트 생성"""
        
        system_prompt = """You are a Domain-Driven Design (DDD) Traceability Expert.

Your goal is to establish precise traceability mappings between pre-generated domain objects and their source functional requirements, ensuring every domain element has clear justification in the business requirements.

With extensive expertise in Domain-Driven Design and requirements traceability, you specialize in analyzing functional requirements to identify specific text segments that justify domain object creation. You excel at creating precise mappings between domain model elements (aggregates, enumerations, value objects) and their source requirements. Your role ensures that every domain object has clear, traceable justification in the business requirements, maintaining a strong link between business needs and technical implementation. You understand that effective traceability is crucial for domain model validation, change impact analysis, and maintaining alignment between business requirements and system design.

**Task:** Add traceability information (refs) to pre-generated domain objects by mapping them to specific parts of the functional requirements. You must establish clear traceability links that justify the existence of each domain object.

**Guidelines:**

1. **Requirements Analysis:**
   - Carefully analyze the functional requirements to understand the business domain and context
   - Identify text segments that describe business concepts, entities, processes, and data elements
   - Look for explicit mentions of entities, statuses, enumerations, processes, and business rules

2. **Domain Object Mapping:**
   - For each domain object, find the specific requirement text that justifies its creation
   - Match object names and aliases to corresponding business concepts in the requirements
   - Aggregates: Map to entity descriptions, business object definitions, or core domain concepts
   - Enumerations: Map to status values, type classifications, or categorical data mentions
   - Value Objects: Map to attribute groups, specification descriptions, or composite data elements

3. **Traceability Reference (refs) Format:**
   - Each domain object MUST include a 'refs' array containing precise references to requirement text
   - Format: [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
   - Use MINIMAL phrases (1-2 words) that uniquely identify the position in the requirement line
   - Use the shortest possible phrase that can accurately locate the specific part of requirements
   - Only reference line numbers that exist in the provided functional requirements
   - Include multiple ranges if a domain object is derived from multiple requirement sections

**CRITICAL OUTPUT REQUIREMENTS:**
- You MUST output ONLY valid JSON, no explanations, no markdown code fences
- Do NOT include ```json or ``` markers
- Do NOT include any text before or after the JSON object

**Output Format:**
{
    "aggregates": [{"name": "...", "refs": [[[line, "phrase"], [line, "phrase"]]]}],
    "enumerations": [{"name": "...", "refs": [[[line, "phrase"], [line, "phrase"]]]}],
    "valueObjects": [{"name": "...", "refs": [[[line, "phrase"], [line, "phrase"]]]}]
}

**Requirements:**
- All domain object names must exactly match the names provided in the input
- Every domain object from the input must be included in the output with refs
- Line numbers in refs must be valid (exist in the requirements document)
- Phrases in refs must be minimal (1-2 words) and accurately identify the location
- Output ONLY the JSON object, nothing else"""

        user_prompt = f"""**Generated Draft Options:**
{json.dumps(filtered_options, ensure_ascii=False, indent=2)}

**Bounded Context Name:** {bounded_context_name}

**Functional Requirements:**
{line_numbered_requirements}

**Domain Objects to Trace:**
{json.dumps(all_domain_objects, ensure_ascii=False, indent=2)}

Please provide traceability mappings for all domain objects listed above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _convert_refs_to_indexes(
        self,
        output: Dict,
        raw_requirements: str,
        line_numbered_requirements: str,
        trace_map: Dict
    ) -> Dict:
        """refs를 phrase → indexes로 변환 (프론트엔드 RefsTraceUtil과 동일)"""
        
        for object_type in ['aggregates', 'enumerations', 'valueObjects']:
            if object_type not in output:
                continue

            for domain_object in output[object_type]:
                if 'refs' not in domain_object or not domain_object['refs']:
                    continue

                # 1. sanitizeAndConvertRefs: phrase → [[[line, col], [line, col]]]
                sanitized_refs = self._sanitize_and_convert_refs(
                    domain_object['refs'],
                    line_numbered_requirements
                )
                domain_object['refs'] = sanitized_refs

                # 2. validateRefs: 범위 검증
                self._validate_refs(domain_object['refs'], raw_requirements)

                # 3. convertToOriginalRefsUsingTraceMap: traceMap 사용해 원본 라인으로 역변환
                domain_object['refs'] = self._convert_to_original_refs_using_trace_map(
                    domain_object['refs'],
                    trace_map
                )

        return output

    def _sanitize_and_convert_refs(self, refs: List, line_numbered_requirements: str) -> List:
        """Frontend RefsTraceUtil.sanitizeAndConvertRefs와 동일"""
        lines = line_numbered_requirements.split('\n')
        
        # 라인 번호 맵 생성
        line_number_map = {}
        for idx, line in enumerate(lines):
            match = re.match(r'^<(\d+)>.*</\1>$', line)
            if match:
                line_num = int(match.group(1))
                line_number_map[line_num] = idx

        sanitized = []
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue

            start_item = ref[0]
            end_item = ref[1]

            if not isinstance(start_item, list) or len(start_item) < 2:
                continue
            if not isinstance(end_item, list) or len(end_item) < 2:
                continue

            start_line = start_item[0]
            start_phrase = start_item[1] if isinstance(start_item[1], str) else ''
            end_line = end_item[0]
            end_phrase = end_item[1] if isinstance(end_item[1], str) else ''

            try:
                start_line_num = int(start_line)
                end_line_num = int(end_line)
            except (ValueError, TypeError):
                continue

            # 라인 번호 맵으로 인덱스 변환
            if start_line_num not in line_number_map or end_line_num not in line_number_map:
                continue

            start_idx = line_number_map[start_line_num]
            end_idx = line_number_map[end_line_num]

            if 0 <= start_idx < len(lines) and 0 <= end_idx < len(lines):
                start_line_text = lines[start_idx]
                end_line_text = lines[end_idx]

                # XML 태그 제거
                start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_text)
                if start_match:
                    start_line_text = start_match.group(1)

                end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_text)
                if end_match:
                    end_line_text = end_match.group(1)

                # phrase 위치 찾기
                start_col = 0
                if start_phrase and start_phrase in start_line_text:
                    start_col = start_line_text.find(start_phrase)

                end_col = len(end_line_text) - 1
                if end_phrase and end_phrase in end_line_text:
                    end_col = end_line_text.find(end_phrase) + len(end_phrase) - 1

                sanitized.append([[start_line_num, start_col], [end_line_num, end_col]])

        return sanitized

    def _validate_refs(self, refs: List, raw_requirements: str):
        """refs 범위 검증"""
        lines = raw_requirements.split('\n')
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue
            start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            
            # 1-based → 0-based
            if start_line < 1 or start_line > len(lines):
                raise ValueError(f"Invalid start line: {start_line}")
            if end_line < 1 or end_line > len(lines):
                raise ValueError(f"Invalid end line: {end_line}")

    def _convert_to_original_refs_using_trace_map(self, refs: List, trace_map: Dict) -> List:
        """traceMap을 사용해 원본 라인으로 역변환 (Frontend RefsTraceUtil.convertToOriginalRefsUsingTraceMap와 동일)"""
        if not trace_map:
            return refs

        # traceMap: { "elementId": { "refs": [[[line, col], [line, col]]], "isDirectMatching": bool } }
        # 현재 refs의 라인 번호를 traceMap 내부의 refs 라인 번호로 변환
        
        converted_refs = []
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue

            start_line_num = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            start_col = ref[0][1] if isinstance(ref[0], list) and len(ref[0]) > 1 else 0
            end_line_num = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            end_col = ref[1][1] if isinstance(ref[1], list) and len(ref[1]) > 1 else 0

            # traceMap에서 해당 라인 범위를 포함하는 원본 refs 찾기
            original_ref = self._find_original_ref_from_trace_map(
                start_line_num, end_line_num, trace_map
            )

            if original_ref:
                converted_refs.append(original_ref)
            else:
                # traceMap에서 찾지 못하면 원본 그대로 유지
                converted_refs.append([[start_line_num, start_col], [end_line_num, end_col]])

        return converted_refs if converted_refs else refs

    def _find_original_ref_from_trace_map(self, start_line: int, end_line: int, trace_map: Dict):
        """traceMap에서 원본 라인 범위 찾기"""
        # traceMap의 각 element를 순회하며 start_line, end_line이 포함되는지 확인
        for element_id, element_data in trace_map.items():
            if not isinstance(element_data, dict):
                continue
            
            element_refs = element_data.get('refs', [])
            if not element_refs:
                continue

            for element_ref in element_refs:
                if not isinstance(element_ref, list) or len(element_ref) < 2:
                    continue

                ref_start_line = element_ref[0][0] if isinstance(element_ref[0], list) and len(element_ref[0]) > 0 else 0
                ref_end_line = element_ref[1][0] if isinstance(element_ref[1], list) and len(element_ref[1]) > 0 else 0

                # 라인 범위가 포함되는지 확인
                if ref_start_line <= start_line and end_line <= ref_end_line:
                    return element_ref

        return None
