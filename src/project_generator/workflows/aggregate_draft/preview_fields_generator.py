"""
Preview Fields Generator - LangGraph Backend
애그리거트 초안에 대한 필수 속성(Preview Fields)을 생성하는 LangGraph 워크플로우
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
from datetime import datetime
from project_generator.utils.logging_util import LoggingUtil


# ==================== Pydantic Models ====================

class PreviewField(BaseModel):
    """개별 필드"""
    fieldName: str = Field(..., description="The field name (e.g., user_id, title)")
    refs: List[List[List[Any]]] = Field(..., description="Traceability references in format [[[startLineNumber, 'start_phrase'], [endLineNumber, 'end_phrase']]]")


class AggregateFieldAssignment(BaseModel):
    """애그리거트별 필드 할당"""
    aggregateName: str = Field(..., description="The name of the aggregate")
    previewFields: List[PreviewField] = Field(..., description="List of fields assigned to this aggregate")


class PreviewFieldsOutput(BaseModel):
    """최종 출력 형식"""
    inference: str = Field(..., description="Detailed reasoning for field generation")
    result: Dict[str, List[AggregateFieldAssignment]] = Field(
        ..., 
        description="Result containing aggregateFieldAssignments"
    )


# ==================== State ====================

class PreviewFieldsState(TypedDict):
    """Preview Fields Generator의 상태"""
    # Input
    description: str
    aggregateDrafts: List[Dict[str, Any]]
    generatorKey: str
    traceMap: Dict[str, Any]
    
    # Processing
    lineNumberedRequirements: str
    prompt: str
    
    # Output
    inference: str
    aggregateFieldAssignments: List[Dict[str, Any]]
    
    # Metadata
    logs: List[Dict[str, str]]
    progress: int
    isCompleted: bool
    isFailed: bool


# ==================== Preview Fields Generator ====================

class PreviewFieldsGenerator:
    """Preview Fields 생성 워크플로우"""
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        
    def _add_line_numbers(self, text: str) -> str:
        """텍스트에 줄 번호 추가"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"<{i}>{line}</{i}>")
        return '\n'.join(numbered_lines)
    
    def _build_persona_info(self) -> Dict[str, str]:
        """Persona 정보 생성"""
        return {
            "persona": "Domain-Driven Design (DDD) Field Generation Specialist",
            "goal": "To analyze functional requirements and aggregate draft structures within a bounded context, then intelligently generate appropriate field names for each aggregate based on domain semantics and business logic.",
            "backstory": "With extensive experience in domain modeling and database design, I specialize in translating business requirements into concrete data structures."
        }
    
    def _build_guidelines(self) -> str:
        """가이드라인 생성"""
        return """<instruction>
    <core_instructions>
        <title>Aggregate Field Generation Task</title>
        <task_description>Analyze functional requirements and generate appropriate field names with traceability information for each aggregate draft in a bounded context.</task_description>
        
        <guidelines>
            <section id="field_generation_rules">
                <title>Field Generation Rules</title>
                <rule id="1">**Business-Driven Generation:** Generate fields that directly support the business operations described in the functional requirements</rule>
                <rule id="2">**Complete Coverage:** Each aggregate should have a comprehensive set of fields</rule>
                <rule id="3">**Avoid Duplication:** Each field concept should appear in only one aggregate unless there's a clear business need</rule>
                <rule id="4">**Domain Semantics:** Field names should reflect the business language</rule>
                <rule id="5">**Consistent Naming:** Use consistent naming patterns (e.g., snake_case)</rule>
            </section>

            <section id="field_categories">
                <title>Standard Field Categories</title>
                <rule id="1">**Identity Fields:** Each aggregate should have a primary identifier (e.g., "user_id", "order_id")</rule>
                <rule id="2">**Core Business Fields:** Fields that represent the main business data</rule>
                <rule id="3">**Lifecycle Fields:** Standard fields like "created_at", "updated_at", "status" where relevant</rule>
                <rule id="4">**Relationship Fields:** Foreign key references to other aggregates when relationships exist</rule>
            </section>

            <section id="traceability">
                <title>Source Traceability Requirements</title>
                <rule id="1">**Mandatory Refs:** Each field MUST include a 'refs' array that traces back to specific parts of the functional requirements</rule>
                <rule id="2">**Refs Format:** Use format [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]</rule>
                <rule id="3">**Minimal Phrases:** Use 1-2 word phrases that uniquely identify the position</rule>
                <rule id="4">**Valid Line Numbers:** Refs must reference valid line numbers from the requirements</rule>
                <rule id="5">**Complete Traceability:** Every generated field must have at least one traceability reference</rule>
            </section>
        </guidelines>

        <refs_format_example>
            <title>Example of refs Format for Field Traceability</title>
            <description>If functional requirements contain:</description>
            <example_requirements>
<1># Course Management System</1>
<3>As an instructor, I want to create and manage my courses. When creating a course, I need to provide a title, description, and price.</3>
<4>Students can enroll in published courses.</4>
<7>- CourseCreated</7>
<8>- CoursePublished</8>
            </example_requirements>
            <example_refs>
- "course_id" field for Course aggregate → refs: [[[3, "create"], [3, "courses"]]]
- "title" field based on course creation → refs: [[[3, "title"], [3, "price"]]]
- "status" field for course lifecycle → refs: [[[7, "CourseCreated"], [8, "CoursePublished"]]]
- "student_id" field for enrollment → refs: [[[4, "Students"], [9, "StudentEnrolled"]]]
            </example_refs>
        </refs_format_example>
    </core_instructions>
</instruction>"""
    
    def _build_prompt(self, state: PreviewFieldsState) -> str:
        """프롬프트 생성"""
        persona = self._build_persona_info()
        guidelines = self._build_guidelines()
        
        aggregates_xml = self._format_aggregates(state['aggregateDrafts'])
        
        prompt = f"""You are a {persona['persona']}.

{persona['goal']}

{guidelines}

## Your Task:
Analyze the following functional requirements and generate appropriate field names with traceability for each aggregate draft.

### Functional Requirements:
{state['lineNumberedRequirements']}

### Aggregate Drafts:
{aggregates_xml}

### Generator Key: {state.get('generatorKey', 'N/A')}

Please generate comprehensive field sets for each aggregate with proper traceability references."""
        
        return prompt
    
    def _format_aggregates(self, aggregates: List[Dict[str, Any]]) -> str:
        """Aggregate 리스트를 XML 형식으로 변환"""
        xml_parts = []
        for agg in aggregates:
            xml_parts.append(f"  <aggregate>")
            xml_parts.append(f"    <name>{agg.get('name', 'Unknown')}</name>")
            if 'alias' in agg:
                xml_parts.append(f"    <alias>{agg['alias']}</alias>")
            xml_parts.append(f"  </aggregate>")
        return "<aggregates>\n" + "\n".join(xml_parts) + "\n</aggregates>"
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Structured Output 스키마 정의"""
        return {
            "type": "object",
            "title": "PreviewFieldsOutput",
            "description": "Output format for preview fields generation",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Detailed reasoning for the field generation"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "aggregateFieldAssignments": {
                            "type": "array",
                            "description": "Field assignments for each aggregate",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aggregateName": {
                                        "type": "string",
                                        "description": "The name of the aggregate"
                                    },
                                    "previewFields": {
                                        "type": "array",
                                        "description": "List of fields for this aggregate",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "fieldName": {
                                                    "type": "string",
                                                    "description": "The field name"
                                                },
                                                "refs": {
                                                    "type": "array",
                                                    "description": "Traceability references",
                                                    "items": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "array",
                                                            "items": {
                                                                "anyOf": [
                                                                    {"type": "number"},
                                                                    {"type": "string"}
                                                                ]
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            "required": ["fieldName", "refs"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["aggregateName", "previewFields"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["aggregateFieldAssignments"],
                    "additionalProperties": False
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }
    
    # ==================== Workflow Nodes ====================
    
    def prepare_input(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """입력 데이터 준비"""
        LoggingUtil.info("PreviewFieldsGenerator", "Preparing input data...")
        
        # Add line numbers to requirements
        description = state.get('description', '')
        state['lineNumberedRequirements'] = self._add_line_numbers(description)
        
        # Build prompt
        state['prompt'] = self._build_prompt(state)
        
        state['logs'] = state.get('logs', [])
        state['logs'].append({
            "message": "Input data prepared",
            "timestamp": datetime.now().isoformat()
        })
        state['progress'] = 10
        
        return state
    
    def generate_fields(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """LLM을 사용하여 Preview Fields 생성"""
        LoggingUtil.info("PreviewFieldsGenerator", "Generating preview fields...")
        aggregates = state.get('aggregateDrafts', [])
        
        try:
            # LLM 초기화
            if not self.llm:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature
                )
            
            # Structured Output 설정
            structured_llm = self.llm.with_structured_output(
                self._get_response_schema(),
                method="json_schema",
                strict=True
            )
            
            # LLM 호출
            response = structured_llm.invoke(state['prompt'])
            
            # 응답 처리
            state['inference'] = response.get('inference', '')
            state['aggregateFieldAssignments'] = response.get('result', {}).get('aggregateFieldAssignments', [])
            
            # 생성된 결과 로깅
            assignments = state['aggregateFieldAssignments']
            LoggingUtil.info("PreviewFieldsGenerator", f"Completed: {len(assignments)} assignments")
            
            # 누락된 aggregate 검사
            input_aggregate_names = set(agg.get('name') for agg in aggregates)
            output_aggregate_names = set(a.get('aggregateName') for a in assignments)
            missing_aggregates = input_aggregate_names - output_aggregate_names
            
            if missing_aggregates:
                LoggingUtil.error(
                    "PreviewFieldsGenerator",
                    f"⚠️ Missing aggregates in output: {list(missing_aggregates)}"
                )
            
            state['logs'].append({
                "message": f"Generated fields for {len(state['aggregateFieldAssignments'])} aggregates",
                "timestamp": datetime.now().isoformat()
            })
            state['progress'] = 90
            
        except Exception as e:
            LoggingUtil.error("PreviewFieldsGenerator", f"Error generating fields: {str(e)}")
            state['isFailed'] = True
            state['logs'].append({
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def finalize_output(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """최종 출력 정리"""
        LoggingUtil.info("PreviewFieldsGenerator", "Finalizing output...")
        
        # Note: refs 변환은 프론트엔드에서 RefsTraceUtil을 사용하여 처리됨
        # 백엔드는 LLM이 생성한 refs를 그대로 반환
        
        # 검증: 각 aggregate가 최소한의 필드를 가지고 있는지 확인
        for assignment in state.get('aggregateFieldAssignments', []):
            if not assignment.get('previewFields') or len(assignment['previewFields']) == 0:
                error_msg = f"Aggregate '{assignment.get('aggregateName', 'Unknown')}' has no generated fields"
                LoggingUtil.error("PreviewFieldsGenerator", error_msg)
                state['isFailed'] = True
                state['logs'].append({
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                return state
        
        state['isCompleted'] = True
        state['progress'] = 100
        state['logs'].append({
            "message": "Preview fields generation completed",
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    # ==================== Workflow Construction ====================
    
    def create_workflow(self) -> StateGraph:
        """워크플로우 생성"""
        workflow = StateGraph(PreviewFieldsState)
        
        # 노드 추가
        workflow.add_node("prepare_input", self.prepare_input)
        workflow.add_node("generate_fields", self.generate_fields)
        workflow.add_node("finalize_output", self.finalize_output)
        
        # 엣지 추가
        workflow.set_entry_point("prepare_input")
        workflow.add_edge("prepare_input", "generate_fields")
        workflow.add_edge("generate_fields", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        return workflow.compile()
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""
        LoggingUtil.info("PreviewFieldsGenerator", "Starting workflow")
        
        # 초기 상태 설정
        initial_state: PreviewFieldsState = {
            'description': input_data.get('description', ''),
            'aggregateDrafts': input_data.get('aggregateDrafts', []),
            'generatorKey': input_data.get('generatorKey', 'default'),
            'traceMap': input_data.get('traceMap', {}),
            'lineNumberedRequirements': '',
            'prompt': '',
            'inference': '',
            'aggregateFieldAssignments': [],
            'logs': [],
            'progress': 0,
            'isCompleted': False,
            'isFailed': False
        }
        
        # 워크플로우 실행
        workflow = self.create_workflow()
        final_state = workflow.invoke(initial_state)
        
        # 결과 반환
        return {
            'inference': final_state.get('inference', ''),
            'aggregateFieldAssignments': final_state.get('aggregateFieldAssignments', []),
            'logs': final_state.get('logs', []),
            'progress': final_state.get('progress', 0),
            'isCompleted': final_state.get('isCompleted', False),
            'isFailed': final_state.get('isFailed', False)
        }

