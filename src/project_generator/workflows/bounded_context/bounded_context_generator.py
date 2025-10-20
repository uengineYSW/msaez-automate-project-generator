"""
BoundedContextGenerator - LangGraph 워크플로우
DDD 원칙 기반 Bounded Context 분할 및 관계 정의
"""
from typing import TypedDict, List, Dict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json
import re
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_generator.config import Config
from src.project_generator.utils.logging_util import LoggingUtil


class BoundedContextState(TypedDict):
    """Bounded Context 생성 상태 (camelCase for Frontend compatibility)"""
    # Inputs
    devisionAspect: str
    requirements: Dict  # { userStory: str, summarizedResult: Dict, analysisResult: Dict, pbcInfo: List[Dict] }
    generateOption: Dict  # { numberOfBCs: int, additionalOptions: str, aspectDetails: Dict, isProtocolMode: bool }
    feedback: Optional[str]
    previousAspectModel: Optional[Dict]
    
    # Outputs
    thoughts: str
    boundedContexts: List[Dict]
    relations: List[Dict]
    explanations: List[Dict]
    
    # Metadata
    progress: int
    logs: Annotated[List[Dict], "append"]
    isCompleted: bool
    error: str


class BoundedContextWorkflow:
    """
    Bounded Context 생성 워크플로우
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_LLM_MODEL,
            temperature=Config.DEFAULT_LLM_TEMPERATURE,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        workflow = StateGraph(BoundedContextState)
        
        # 노드 추가
        workflow.add_node("generate_bounded_contexts", self.generate_bounded_contexts)
        workflow.add_node("finalize", self.finalize)

        # 그래프 연결
        workflow.set_entry_point("generate_bounded_contexts")
        workflow.add_edge("generate_bounded_contexts", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run(self, inputs: Dict) -> Dict:
        """
        워크플로우 실행
        """
        initial_state: BoundedContextState = {
            "devisionAspect": inputs.get("devisionAspect", ""),
            "requirements": inputs.get("requirements", {}),
            "generateOption": inputs.get("generateOption", {}),
            "feedback": inputs.get("feedback"),
            "previousAspectModel": inputs.get("previousAspectModel"),
            "thoughts": "",
            "boundedContexts": [],
            "relations": [],
            "explanations": [],
            "progress": 0,
            "logs": [],
            "isCompleted": False,
            "error": ""
        }
        
        LoggingUtil.info("BoundedContextWorkflow", f"워크플로우 시작 (Aspect: {initial_state['devisionAspect']})")
        final_state = self.workflow.invoke(initial_state)
        LoggingUtil.info("BoundedContextWorkflow", f"워크플로우 완료 (Aspect: {initial_state['devisionAspect']})")
        
        return self.finalize_result(final_state)

    def generate_bounded_contexts(self, state: BoundedContextState) -> Dict:
        """
        Bounded Context 생성
        """
        devision_aspect = state["devisionAspect"]
        requirements = state["requirements"]
        generate_option = state["generateOption"]
        feedback = state.get("feedback")
        previous_aspect_model = state.get("previousAspectModel")
        
        LoggingUtil.info("BoundedContextWorkflow", f"📝 BC 생성 시작 (Aspect: {devision_aspect})")
        
        # 요구사항 언어 감지
        user_story = requirements.get("userStory", "")
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in user_story[:500])
        language = "Korean" if has_korean else "English"
        
        # 프롬프트 구성
        LoggingUtil.info("BoundedContextWorkflow", "📝 프롬프트 구성 시작...")
        prompt_dict = self._build_prompt(
            devision_aspect,
            requirements,
            generate_option,
            feedback,
            previous_aspect_model,
            language
        )
        
        LoggingUtil.info("BoundedContextWorkflow", "📝 프롬프트 구성 완료")
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # System Message
            messages = [SystemMessage(content=prompt_dict["system"])]
            
            # User Messages
            for user_msg in prompt_dict["user"]:
                messages.append(HumanMessage(content=user_msg))
            
            # 프롬프트 로깅 (디버깅용)
            LoggingUtil.info("BoundedContextWorkflow", f"📝 System Prompt 길이: {len(prompt_dict['system'])}")
            LoggingUtil.info("BoundedContextWorkflow", f"📝 User Prompt 개수: {len(prompt_dict['user'])}")
            for i, user_msg in enumerate(prompt_dict["user"]):
                LoggingUtil.info("BoundedContextWorkflow", f"📝 User Prompt {i+1} 길이: {len(user_msg)}")
            
            response_chunks = []
            
            # 스트리밍으로 LLM 응답 수신
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    response_chunks.append(chunk.content)
            
            response = "".join(response_chunks)
            
            # LLM 응답 로깅 (디버깅용)
            LoggingUtil.info("BoundedContextWorkflow", f"📝 LLM 응답 길이: {len(response)}")
            LoggingUtil.info("BoundedContextWorkflow", f"📝 LLM 응답 처음 500자: {response[:500]}")
            
            response_clean = self._extract_json(response)
            result_data = json.loads(response_clean)
            
            LoggingUtil.info("BoundedContextWorkflow", f"✅ BC LLM 응답 완료: {len(result_data.get('boundedContexts', []))}개 BC")
            LoggingUtil.info("BoundedContextWorkflow", f"📝 thoughts 길이: {len(result_data.get('thoughts', ''))}")
            LoggingUtil.info("BoundedContextWorkflow", f"📝 relations 개수: {len(result_data.get('relations', []))}")
            LoggingUtil.info("BoundedContextWorkflow", f"📝 explanations 개수: {len(result_data.get('explanations', []))}")
            
            return {
                "thoughts": result_data.get("thoughts", ""),
                "boundedContexts": result_data.get("boundedContexts", []),
                "relations": result_data.get("relations", []),
                "explanations": result_data.get("explanations", []),
                "progress": 50,
                "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": "BC 생성 완료"}]
            }
        except Exception as e:
            LoggingUtil.exception("BoundedContextWorkflow", "BC 생성 중 오류 발생", e)
            return {
                "thoughts": "",
                "boundedContexts": [],
                "relations": [],
                "explanations": [],
                "error": str(e),
                "progress": 50,
                "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": f"BC 생성 오류: {str(e)}"}]
            }

    def _build_prompt(self, devision_aspect, requirements, generate_option, feedback, previous_aspect_model, language):
        """프롬프트 구성 (기존 DevideBoundedContextGenerator.js와 동일)"""
        LoggingUtil.info("BoundedContextWorkflow", f"📝 _build_prompt 시작: aspect={devision_aspect}, language={language}")
        
        # Maximum BC count
        max_bcs = generate_option.get('numberOfBCs', 5)
        LoggingUtil.info("BoundedContextWorkflow", f"📝 max_bcs: {max_bcs}")
        
        # Task Guidelines (이미 <instruction> 태그 포함)
        LoggingUtil.info("BoundedContextWorkflow", "📝 Task Guidelines 구성 중...")
        task_guidelines = self._build_task_guidelines(language, max_bcs)
        LoggingUtil.info("BoundedContextWorkflow", f"📝 Task Guidelines 완료: {len(task_guidelines)}자")
        
        # End comment (기존 생성기와 동일)
        end_comment = "\n\n<request>This is the entire guideline. When you're ready, please output 'Approved.' Then I will begin user input.</request>"
        
        # User Input (JSON 형식으로 변환)
        LoggingUtil.info("BoundedContextWorkflow", "📝 User Input 구성 중...")
        user_input_dict = self._build_user_input_dict(
            devision_aspect,
            requirements,
            generate_option,
            feedback,
            previous_aspect_model
        )
        LoggingUtil.info("BoundedContextWorkflow", f"📝 User Input 완료: {len(str(user_input_dict))}자")
        
        # Persona 정보 (System Prompt)
        LoggingUtil.info("BoundedContextWorkflow", "📝 Persona 정보 구성 중...")
        persona_info = """<persona_and_role>
<persona>Expert Domain-Driven Design (DDD) Architect</persona>
<goal>To analyze functional requirements and divide them into appropriate Bounded Contexts following Domain-Driven Design principles, ensuring high cohesion and low coupling.</goal>
<backstory>I am a highly experienced domain architect specializing in system decomposition and bounded context design. I have extensive knowledge of domain-driven design principles and patterns, microservices architecture and context boundaries, business domain modeling and strategic design, and event-driven architecture and system integration patterns. My expertise lies in identifying natural boundaries within complex business domains and creating cohesive, loosely-coupled bounded contexts that align with organizational structure and business capabilities. I excel at analyzing actor interactions, event flows, and business capabilities to determine optimal context boundaries and integration patterns.</backstory>
</persona_and_role>"""

        # Language Guide
        language_guide = f"<language_guide>Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.</language_guide>"

        # 프롬프트 구성 (기존 생성기와 동일)
        # 1. System Prompt (Persona)
        system_prompt = persona_info
        
        # 2. User Prompt (Task Guidelines + End Comment)
        user_prompt_1 = task_guidelines + end_comment
        
        # 3. User Prompt (User Inputs)
        user_prompt_2 = json.dumps(user_input_dict, ensure_ascii=False, indent=2)
        
        # 4. User Prompt (Language Guide)
        user_prompt_3 = language_guide
        
        return {
            "system": system_prompt,
            "user": [user_prompt_1, user_prompt_2, user_prompt_3]
        }

    def finalize(self, state: BoundedContextState) -> Dict:
        """
        최종 결과 정리 (기존 DevideBoundedContextGenerator.js의 _processAIOutput과 동일)
        """
        LoggingUtil.info("BoundedContextWorkflow", "✨ BC 워크플로우 최종 정리")
        
        # Frontend의 _processAIOutput 로직 구현
        result = {
            "devisionAspect": state.get("devisionAspect", ""),
            "thoughts": state.get("thoughts", ""),
            "boundedContexts": state.get("boundedContexts", []),
            "relations": state.get("relations", []),
            "explanations": state.get("explanations", []),
            "progress": 100,
            "isCompleted": True,
            "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": "BC 워크플로우 완료"}]
        }
        
        # events와 requirements를 빈 배열로 초기화 (기존 생성기와 동일)
        for bc in result.get("boundedContexts", []):
            if not bc.get("events"):
                bc["events"] = []
            if not bc.get("requirements"):
                bc["requirements"] = []
        
        return result

    def finalize_result(self, state: BoundedContextState) -> Dict:
        """
        최종 결과를 반환하기 전에 필요한 후처리
        """
        return {
            "devisionAspect": state.get("devisionAspect", ""),
            "thoughts": state.get("thoughts", ""),
            "boundedContexts": state.get("boundedContexts", []),
            "relations": state.get("relations", []),
            "explanations": state.get("explanations", []),
            "progress": state.get("progress", 0),
            "logs": state.get("logs", []),
            "isCompleted": state.get("isCompleted", False),
            "error": state.get("error", "")
        }

    def _extract_json(self, text: str) -> str:
        """LLM 응답에서 JSON 부분만 추출"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _build_task_guidelines(self, language: str, max_bcs: int = 5) -> str:
        """Task Guidelines 프롬프트 구성 (기존 DevideBoundedContextGenerator.js와 동일)"""
        return f"""<instruction>
<core_instructions>
    <title>Bounded Context Division Task</title>
    <task_description>Your task is to analyze the provided requirements and divide them into multiple Bounded Contexts following Domain-Driven Design principles. You will identify natural boundaries within the business domain and create cohesive, loosely-coupled bounded contexts that align with organizational structure and business capabilities.</task_description>
    
    <input_description>
        <title>You will receive user inputs containing:</title>
        <item id="1">**Division Aspect:** The specific aspect to focus on when dividing contexts</item>
        <item id="2">**Maximum Number of Bounded Contexts:** The maximum number of bounded contexts to create</item>
        <item id="3">**Requirements Document:** Business requirements with actors and events</item>
        <item id="4">**Available Pre-Built Components (PBCs):** List of reusable components that can be leveraged</item>
        <item id="5">**Additional Rules:** Optional additional requirements and constraints to consider</item>
    </input_description>

    <guidelines>
        <title>Bounded Context Division Guidelines</title>
        
        <section id="core_principles">
            <title>Core Principles</title>
            <rule id="1">**High Cohesion, Low Coupling:** Group related behaviors and data together while minimizing inter-context dependencies</rule>
            <rule id="2">**Event Action Range:** Seize event's action range to create bounded context</rule>
            <rule id="3">**Event Flow:** Seize relation between events to create flow</rule>
            <rule id="4">**Actor Grouping:** Consider which actors are responsible for which events</rule>
            <rule id="5">**Business Capability Alignment:** Ensure bounded contexts align with business capabilities</rule>
        </section>

        <section id="domain_classification">
            <title>Domain Classification Strategy</title>
            
            <core_domain>
                <title>Core Domain</title>
                <characteristics>
                    <item>Direct impact on business competitive advantage</item>
                    <item>User-facing functionality</item>
                    <item>Strategic importance to business goals</item>
                </characteristics>
                <scoring>
                    <differentiation>Typically 0.8-1.0 (high business differentiation value)</differentiation>
                    <complexity>Can vary (0.4-1.0)</complexity>
                </scoring>
                <implementation_strategy>Rich Domain Model</implementation_strategy>
            </core_domain>

            <supporting_domain>
                <title>Supporting Domain</title>
                <characteristics>
                    <item>Enables core domain functionality</item>
                    <item>Internal business processes</item>
                    <item>Medium business impact</item>
                </characteristics>
                <scoring>
                    <differentiation>Typically 0.4-0.7 (medium business differentiation)</differentiation>
                    <complexity>Can vary (0.3-0.9)</complexity>
                </scoring>
                <implementation_strategy>Transaction Script or Active Record</implementation_strategy>
            </supporting_domain>

            <generic_domain>
                <title>Generic Domain</title>
                <characteristics>
                    <item>Common functionality across industries</item>
                    <item>Can be replaced by third-party solutions</item>
                    <item>Low differentiation but can have high complexity</item>
                </characteristics>
                <scoring>
                    <differentiation>0.0-0.3 (low business differentiation)</differentiation>
                    <complexity>Can vary (0.2-1.0, can be high despite low differentiation)</complexity>
                </scoring>
                <implementation_strategy>Active Record or PBC: (pbc-name)</implementation_strategy>
            </generic_domain>
        </section>

        <section id="scoring_instructions">
            <title>Scoring Instructions</title>
            <complexity>
                <description>Score from 0.0 to 1.0 indicating technical implementation difficulty</description>
                <considerations>
                    <item>Technical dependencies</item>
                    <item>Business rules complexity</item>
                    <item>Data consistency requirements</item>
                </considerations>
                <note>High score is possible even for Generic domains</note>
            </complexity>
            <differentiation>
                <description>Score from 0.0 to 1.0 indicating business differentiation value</description>
                <considerations>
                    <item>Competitive advantage</item>
                    <item>User interaction</item>
                    <item>Strategic importance</item>
                </considerations>
                <note>User-facing domains should have higher scores</note>
            </differentiation>
        </section>

        <section id="pbc_matching">
            <title>Pre-Built Component (PBC) Matching Rule</title>
            <importance>CRITICAL</importance>
            <rule id="1">**Priority:** Before creating any bounded context, first check if the functionality already exists in the available PBCs provided in user input</rule>
            <rule id="2">**If Match Found:** You MUST create it as a "Generic Domain" bounded context</rule>
            <rule id="3">**Implementation Strategy:** Set to "PBC: [pbc-name]"</rule>
            <rule id="4">**Naming:** Bounded context name of PBC must be written as is pbc name</rule>
            <rule id="5">**Precedence:** This rule takes precedence over all other domain classification rules</rule>
        </section>

        <section id="aggregate_extraction">
            <title>Aggregate Extraction</title>
            <rule id="1">**Identify Aggregates:** For each bounded context, extract aggregates that represent business entities and their consistency boundaries</rule>
            <rule id="2">**Naming:** Aggregates should be named in PascalCase</rule>
            <rule id="3">**Alias:** Provide alias in the same national language as the requirements</rule>
        </section>

        <section id="traceability">
            <title>Source Traceability Requirements</title>
            <rule id="1">**Mandatory Refs:** Every bounded context role, relation, and explanation MUST include refs linking back to specific requirement lines</rule>
            <rule id="2">**Refs Format:** Use format [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]</rule>
            <rule id="3">**Minimal Phrases:** Use 1-2 word phrases that uniquely identify the position in the line</rule>
            <rule id="4">**Valid Line Numbers:** Refs must reference valid line numbers from the requirements section</rule>
            <rule id="5">**Multiple References:** Include multiple ranges if a field references multiple parts of requirements</rule>
        </section>

        <section id="language_instructions">
            <title>Language Instructions</title>
            <rule id="1">**National Language Usage:** Use the same national language as the Requirements for: thoughts, explanations, alias, requirements</rule>
            <rule id="2">**Bounded Context Names:** Must be written in English PascalCase</rule>
            <rule id="3">**References in Explanations:** When referring to bounded context in explanations, use alias</rule>
        </section>
    </guidelines>

    <refs_format_example>
        <title>Example of refs Format</title>
        <description>If requirements contain:</description>
        <example_requirements>
<1>E-commerce Platform</1>
<2></2>
<3>Users can browse and purchase products</3>
<4>Payment processing with multiple providers</4>
<5>Order fulfillment and tracking system</5>
<6>Customer support chat functionality</6>
        </example_requirements>
        <example_refs>
- "ProductCatalog" BC with role "Manages product information and browsing" → roleRefs: [[[3, "Users"], [3, "products"]]]
- Relation between "Payment" and "Order" → refs: [[[4, "Payment"], [5, "Order"]]]
- Explanation about "CustomerSupport" interaction → refs: [[[6, "Customer"], [6, "functionality"]]]
        </example_refs>
    </refs_format_example>
</core_instructions>

<output_format>
    <title>JSON Output Format</title>
    <description>The output must be a JSON object structured as follows:</description>
    <schema>
{{
    "thoughts": "(Explanations of how Bounded Contexts were derived: cohesion & coupling analysis, domain expertise, technical cohesion, persona-based division, etc.)",
    "boundedContexts": [
        {{
            "name": "(Bounded Context name in PascalCase)",
            "alias": "(Alias of Bounded Context in national language of Requirements)",
            "importance": "Core Domain" || "Supporting Domain" || "Generic Domain",
            "complexity": (number: 0.0-1.0, technical implementation difficulty),
            "differentiation": (number: 0.0-1.0, business differentiation value),
            "implementationStrategy": "Event Sourcing" || "Rich Domain Model" || "Transaction Script" || "Active Record" || "PBC: (pbc-name)",
            "aggregates": [
                {{
                    "name": "(Aggregate name in PascalCase)",
                    "alias": "(Alias of Aggregate in language of Requirements)"
                }}
            ],
            "events": [], // All events that are composed from this Bounded Context
            "requirements": [], // Must be empty array
            "role": "(Explanation of what to do and how this Bounded Context works)",
            "roleRefs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ],
    "relations": [
        {{
            "name": "(Name of relation between Bounded Contexts)",
            "type": "(Relation type - refer to Additional Rules in user input for allowed types)",
            "upStream": {{
                "name": "(Name of upstream Bounded Context)",
                "alias": "(Alias of upstream Bounded Context in language of Requirements)"
            }},
            "downStream": {{
                "name": "(Name of downstream Bounded Context)",
                "alias": "(Alias of downstream Bounded Context in language of Requirements)"
            }},
            "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ],
    "explanations": [
        {{
            "sourceContext": "(Source Bounded Context alias)",
            "targetContext": "(Target Bounded Context alias)",
            "relationType": "(Relationship type)",
            "reason": "(Explanation of why this type was chosen)",
            "interactionPattern": "(Description of how these contexts interact)",
            "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ]
}}
    </schema>
    <field_requirements>
        <requirement id="1">All field names must match exactly as shown in the schema</requirement>
        <requirement id="2">Bounded Context names must be PascalCase</requirement>
        <requirement id="3">Alias must be in the same language as the requirements</requirement>
        <requirement id="4">All refs must use minimal phrases and valid line numbers</requirement>
    </field_requirements>
</output_format>
</instruction>"""

    def _build_user_input_dict(self, devision_aspect, requirements, generate_option, feedback, previous_aspect_model) -> dict:
        """User Input 딕셔너리 구성 (기존 DevideBoundedContextGenerator.js의 __buildJsonUserQueryInputFormat과 동일)"""
        user_input = {
            "division_aspect": devision_aspect,
            "maximum_number_of_bounded_contexts": generate_option.get('numberOfBCs', 5),
            "actors": requirements.get('analysisResult', {}).get('actors', []),
            "events": requirements.get('analysisResult', {}).get('events', []),
            "available_pre_built_components_pbcs": requirements.get('pbcInfo', []),
            "additional_rules": self._build_additional_rules(generate_option),
            "requirements": self._get_line_numbered_requirements(requirements)
        }
        
        # Feedback (if exists)
        if feedback:
            user_input["feedback"] = self._feedback_prompt(feedback, previous_aspect_model)
        
        return user_input

    def _build_user_input(self, devision_aspect, requirements, generate_option, feedback, previous_aspect_model) -> str:
        """User Input 프롬프트 구성"""
        user_input = f"""<user_input>
    <division_aspect>{devision_aspect}</division_aspect>
    <maximum_number_of_bounded_contexts>{generate_option.get('numberOfBCs', 5)}</maximum_number_of_bounded_contexts>
    <actors>{json.dumps(requirements.get('analysisResult', {}).get('actors', []), ensure_ascii=False)}</actors>
    <events>{json.dumps(requirements.get('analysisResult', {}).get('events', []), ensure_ascii=False)}</events>
    <available_pre_built_components_pbcs>{json.dumps(requirements.get('pbcInfo', []), ensure_ascii=False)}</available_pre_built_components_pbcs>
"""
        
        # Additional rules
        additional_rules = self._build_additional_rules(generate_option)
        if additional_rules:
            user_input += f"    <additional_rules>{additional_rules}</additional_rules>\n"
        
        # Requirements
        requirements_text = self._get_line_numbered_requirements(requirements)
        user_input += f"    <requirements>{requirements_text}</requirements>\n"
        
        # Feedback (if exists)
        if feedback:
            user_input += f"    <feedback>{feedback}</feedback>\n"
        
        # Previous aspect model (if exists)
        if previous_aspect_model:
            user_input += f"    <previous_aspect_model>{json.dumps(previous_aspect_model, ensure_ascii=False)}</previous_aspect_model>\n"
        
        user_input += "</user_input>"
        return user_input

    def _feedback_prompt(self, feedback, previous_aspect_model) -> str:
        """Feedback 프롬프트 구성 (기존 DevideBoundedContextGenerator.js의 _feedbackPrompt과 동일)"""
        return f"""<feedback_for_regeneration>
    <previous_model>
{json.dumps(previous_aspect_model, ensure_ascii=False, indent=2)}
    </previous_model>
    <instruction>Please refer to the added feedback below to create a new model that addresses the user's concerns while maintaining consistency with the requirements.</instruction>
    <user_feedback>
{feedback}
    </user_feedback>
</feedback_for_regeneration>"""

    def _build_additional_rules(self, generate_option) -> str:
        """Additional Rules 프롬프트 구성 (기존 DevideBoundedContextGenerator.js와 동일)"""
        rules = []
        is_protocol_mode = generate_option.get('isProtocolMode', False)
        aspect_details = generate_option.get('aspectDetails', {})
        
        # Relation type rules based on mode
        if is_protocol_mode:
            rules.append("""<relation_type_constraint>
    <allowed_types>
        <type>Request/Response</type>
        <type>Pub/Sub</type>
    </allowed_types>
    <requirements>
        <requirement>All Bounded Contexts must have at least one relation</requirement>
        <requirement>Event-driven architecture is preferred for loose coupling</requirement>
        <requirement>All relation types must use 'Pub/Sub' pattern. However, only Generic domains as downstream MUST use 'Request/Response' pattern</requirement>
    </requirements>
</relation_type_constraint>""")
        else:
            rules.append("""<relation_type_constraint>
    <allowed_types>
        <type>Conformist</type>
        <type>Shared Kernel</type>
        <type>Anti-corruption</type>
        <type>Separate Ways</type>
        <type>Customer-Supplier</type>
    </allowed_types>
</relation_type_constraint>""")
        
        # Aspect details
        if aspect_details:
            aspect_details_xml = """<specific_aspect_requirements>
    <description>When determining and explaining the bounded contexts, consider and reflect the following specific requirements:</description>"""
            
            if aspect_details.get('organizationalAspect'):
                aspect_details_xml += f"""
    <organizational_aspect>
        <details>{aspect_details['organizationalAspect']}</details>
        <instruction>Please reflect this team structure when separating bounded contexts</instruction>
    </organizational_aspect>"""
            
            if aspect_details.get('infrastructureAspect'):
                aspect_details_xml += f"""
    <infrastructure_aspect>
        <details>{aspect_details['infrastructureAspect']}</details>
        <instruction>Please consider these technical requirements when defining bounded contexts</instruction>
    </infrastructure_aspect>"""
            
            aspect_details_xml += """
    <important_note>In the "thoughts" section of your response, explicitly explain how these specific organizational and infrastructure requirements influenced your bounded context separation decisions.</important_note>
</specific_aspect_requirements>"""
            
            rules.append(aspect_details_xml)
        
        # Additional options from user
        additional_options = generate_option.get('additionalOptions', '')
        if additional_options:
            rules.append(f"<user_additional_requirements><content>{additional_options}</content></user_additional_requirements>")
        
        if not rules:
            return "<status>None</status>"
        
        return '\n'.join(rules)

    def _get_line_numbered_requirements(self, requirements) -> str:
        """라인 번호가 추가된 요구사항 텍스트 반환"""
        # summarizedResult가 있으면 사용, 없으면 userStory 사용
        summarized_result = requirements.get('summarizedResult', {})
        if summarized_result and summarized_result.get('summary'):
            text = summarized_result['summary']
        else:
            text = requirements.get('userStory', '')
        
        # 라인 번호 추가
        lines = text.split('\n')
        numbered_lines = [f"<{i+1}>{line}</{i+1}>" for i, line in enumerate(lines)]
        return '\n'.join(numbered_lines)

    def _build_output_format(self, language: str) -> str:
        """Output Format 프롬프트 구성"""
        return f"""<output_format>
<title>JSON Output Format</title>
<schema>
{{
    "thoughts": "(Explanations of how Bounded Contexts were derived in {language})",
    "boundedContexts": [
        {{
            "name": "(Bounded Context name in PascalCase)",
            "alias": "(Alias in {language})",
            "importance": "Core Domain" || "Supporting Domain" || "Generic Domain",
            "complexity": (number: 0.0-1.0),
            "differentiation": (number: 0.0-1.0),
            "implementationStrategy": "Event Sourcing" || "Rich Domain Model" || "Transaction Script" || "Active Record" || "PBC: (pbc-name)",
            "aggregates": [
                {{
                    "name": "(Aggregate name in PascalCase)",
                    "alias": "(Alias in {language})"
                }}
            ],
            "events": [],
            "requirements": [],
            "role": "(Explanation in {language})",
            "roleRefs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ],
    "relations": [
        {{
            "name": "(Name of relation)",
            "type": "(Relation type)",
            "upStream": {{
                "name": "(Name of upstream BC)",
                "alias": "(Alias in {language})"
            }},
            "downStream": {{
                "name": "(Name of downstream BC)",
                "alias": "(Alias in {language})"
            }},
            "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ],
    "explanations": [
        {{
            "sourceContext": "(Source BC alias)",
            "targetContext": "(Target BC alias)",
            "relationType": "(Relationship type)",
            "reason": "(Explanation in {language})",
            "interactionPattern": "(Description in {language})",
            "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
        }}
    ]
}}
</schema>
</output_format>"""

