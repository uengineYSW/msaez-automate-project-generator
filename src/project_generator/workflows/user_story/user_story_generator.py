"""
UserStoryGenerator - LangGraph 워크플로우
RAG를 활용한 User Story 자동 생성
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_generator.workflows.common.rag_retriever import RAGRetriever
from src.project_generator.config import Config
from src.project_generator.utils.logging_util import LoggingUtil


class UserStoryState(TypedDict):
    """UserStory 생성 상태 (camelCase for Frontend compatibility)"""
    # Inputs
    requirements: str                    # 입력 요구사항
    bounded_contexts: List[Dict]         # Bounded Context 정보 (선택)
    
    # RAG Context
    rag_context: Dict                   # RAG 검색 결과
    
    # Outputs
    userStories: List[Dict]             # 생성된 User Stories (camelCase)
    actors: List[Dict]                  # Actors
    businessRules: List[Dict]           # Business Rules
    boundedContexts: List[Dict]         # Bounded Contexts
    textResponse: str                   # 텍스트 모드 응답 (선택)
    
    # Metadata
    progress: int                       # 진행률 (0-100)
    logs: Annotated[List[Dict], "append"]  # 로그 (append only)
    isCompleted: bool                   # 완료 여부 (camelCase)
    error: str                         # 에러 메시지


class UserStoryWorkflow:
    """
    UserStory 생성 워크플로우
    
    Workflow:
    1. retrieve_rag: RAG로 유사 프로젝트 및 패턴 검색
    2. generate_user_stories: User Story 생성
    3. validate_user_stories: 검증
    4. finalize: 최종 결과 정리
    """
    
    def __init__(self):
        self.rag_retriever = RAGRetriever()
        # ✅ Frontend와 완전히 동일한 설정 (model_kwargs 대신 직접 파라미터로)
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_LLM_MODEL,  # gpt-4.1-2025-04-14
            temperature=Config.DEFAULT_LLM_TEMPERATURE,  # 0.2
            top_p=1.0,  # model_kwargs → 직접 파라미터 ✅
            frequency_penalty=0.0,  # model_kwargs → 직접 파라미터 ✅
            presence_penalty=0.0,  # model_kwargs → 직접 파라미터 ✅
            # max_tokens 설정 안 함 (Frontend도 없음)
            verbose=True  # 🔍 디버깅: API 요청 로깅
        )
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        workflow = StateGraph(UserStoryState)
        
        # 노드 추가
        workflow.add_node("retrieve_rag", self.retrieve_rag_context)
        workflow.add_node("generate_user_stories", self.generate_user_stories)
        workflow.add_node("validate_user_stories", self.validate_user_stories)
        workflow.add_node("finalize", self.finalize_result)
        
        # 엣지 설정
        workflow.set_entry_point("retrieve_rag")
        workflow.add_edge("retrieve_rag", "generate_user_stories")
        workflow.add_edge("generate_user_stories", "validate_user_stories")
        workflow.add_edge("validate_user_stories", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def retrieve_rag_context(self, state: UserStoryState) -> Dict:
        """
        Step 1: RAG로 유사 프로젝트 및 User Story 패턴 검색
        """
        requirements = state["requirements"]
        
        # User Story 패턴 검색
        user_story_patterns = self.rag_retriever.search_ddd_patterns(
            f"User Story patterns for: {requirements}",
            k=10
        )
        
        # 유사 프로젝트 검색
        similar_projects = self.rag_retriever.search_project_templates(
            requirements,
            k=5
        )
        
        # 도메인 용어 검색
        vocabulary = self.rag_retriever.search_vocabulary(requirements, k=20)
        
        return {
            "rag_context": {
                "user_story_patterns": user_story_patterns,
                "similar_projects": similar_projects,
                "vocabulary": vocabulary
            },
            "progress": 20,
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "level": "info",
                "message": f"RAG context retrieved: {len(user_story_patterns)} patterns, {len(similar_projects)} projects, {len(vocabulary)} terms"
            }]
        }
    
    def generate_user_stories(self, state: UserStoryState) -> Dict:
        """
        Step 2: RAG 컨텍스트를 활용한 User Story 생성
        """
        
        requirements = state["requirements"]
        rag_context = state.get("rag_context", {})
        bounded_contexts = state.get("bounded_contexts", [])
        
        # ✅ 기존 청크의 누적 결과 (Recursive 모드)
        existing_actors = state.get("existingActors", [])
        existing_user_stories = state.get("existingUserStories", [])
        existing_business_rules = state.get("existingBusinessRules", [])
        existing_bounded_contexts = state.get("existingBoundedContexts", [])
        
        # RAG 컨텍스트를 문자열로 포맷
        rag_context_str = self._format_rag_context(rag_context)
        
        # 요구사항 언어 감지 (간단한 휴리스틱)
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in requirements[:500])
        language = "Korean" if has_korean else "English"
        
        # ✅ 기존 데이터 프롬프트 생성 (중복 방지)
        existing_data_prompt = self._format_existing_data(
            existing_actors, existing_user_stories, 
            existing_business_rules, existing_bounded_contexts
        )
        
        # 기존 데이터가 있는지 확인 (첫 번째 청크인지 판별)
        has_existing_data = bool(existing_data_prompt.strip())
        
        # 가상 시나리오 모드 감지 (요구사항이 없을 때)
        is_virtual_scenario = self._is_virtual_scenario(requirements)
        
        # 요구사항 텍스트 포맷팅
        user_story_section = ""
        if is_virtual_scenario:
            # 가상 시나리오 모드: requirements 자체가 프롬프트
            user_story_section = requirements
        elif requirements and len(requirements) >= 100:
            user_story_section = f"""
The user story is: {requirements}

Please generate user stories and scenarios based on the above content, staying within the scope and context provided.
"""
        else:
            # userStory가 100자 미만이면 빈 문자열 (기존 생성기와 동일)
            user_story_section = ""
        
        # 가상 시나리오 모드 또는 userStory가 100자 미만일 때는 텍스트 형식 사용
        is_text_mode = is_virtual_scenario or not user_story_section
        
        if is_text_mode:
            # 가상 시나리오 모드: Frontend에서 이미 완성된 프롬프트를 전달하므로 그대로 사용
            if is_virtual_scenario:
                # requirements 자체가 이미 완성된 프롬프트 (Personas, Business Model 포함)
                prompt = f"""{requirements}

{existing_data_prompt}

LANGUAGE REQUIREMENT:
Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.

The response must:
- Ensure complete traceability between actors, stories
- Avoid any missing connections between components
- Provide a clear, well-structured text response
"""
            else:
                # 제목만 있는 경우: 기본 프롬프트 사용
                prompt = f"""Please generate a comprehensive analysis for the service with the following requests:

{existing_data_prompt}

1. Actors:
   - List all actors (users, external systems, etc.) that interact with the system
   - Describe each actor's role and responsibilities
   
2. User Stories (in detailed usecase spec):
   - Create detailed user stories for each actor
   - Each story should include "As a", "I want to", "So that" format
   - Include acceptance criteria for each story
   
3. Business Rules:
   - Define core business rules and constraints
   - Include validation rules and business logic
   - Specify any regulatory or compliance requirements

LANGUAGE REQUIREMENT:
Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.

The response must:
- Ensure complete traceability between actors, stories
- Avoid any missing connections between components
- Provide a clear, well-structured text response
"""
        else:
            prompt = f"""Please analyze the following requirements and extract ONLY the actors, user stories, and business rules that are EXPLICITLY mentioned or directly implied.

{user_story_section}

{existing_data_prompt}

REQUIREMENTS ANALYSIS RULES:
1. Actors:
   - Extract ONLY actors that are explicitly mentioned in the requirements
   - Do NOT create fictional or hypothetical actors
   - Focus on actual users, systems, or external entities mentioned in the text

2. User Stories:
   - Create user stories ONLY for features and functionality explicitly described in the requirements
   - Do NOT invent additional features or expand beyond what is specified
   - Each story must be directly traceable to specific requirements in the text
   - Use the format: "As a [role], I want [specific feature], so that [stated benefit]"

3. Business Rules:
   - Extract ONLY business rules that are explicitly stated or clearly implied in the requirements
   - Do NOT create generic or hypothetical business rules
   - Focus on actual constraints, validations, or processes mentioned in the text

OUTPUT FORMAT (JSON only, no markdown):
{{
  "title": "서비스 제목",
  "actors": [
    {{
      "title": "액터 제목",
      "description": "액터 설명",
      "role": "액터 역할"
    }}
  ],
  "userStories": [
    {{
      "title": "유저스토리 제목",
      "as": "As a [role]",
      "iWant": "I want [action]",
      "soThat": "so that [benefit]",
      "description": "상세 설명"
    }}
  ],
  "businessRules": [
    {{
      "title": "비즈니스 규칙 제목",
      "description": "규칙 설명"
    }}
  ],
  "boundedContexts": [
    {{
      "name": "컨텍스트 이름",
      "description": "컨텍스트 설명"
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Generate content ONLY from the provided requirements text
- Do NOT create fictional scenarios or hypothetical features
- Do NOT expand beyond what is explicitly stated or directly implied
- Extract ALL features, actors, and business rules mentioned in the requirements
- Be thorough and comprehensive in your extraction{" - The following topics are ALREADY COVERED in previous chunks. Do NOT duplicate them." if has_existing_data else ""}

The response must:
- Be directly traceable to the provided requirements text
- Avoid any fictional or hypothetical content
- Return ONLY the JSON object, no additional text

LANGUAGE REQUIREMENT:
Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.

FORMAT REQUIREMENT:
Please generate the json in valid json format and if there's a property its value is null, don't contain the property. Also, please return only the json without any natural language.
"""
        
        try:
            # LLM 호출 (스트리밍 사용 - 더 완전한 응답 생성)
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=prompt)]
            
            # 스트리밍으로 받아서 완전한 응답 확보
            response_chunks = []
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    response_chunks.append(chunk.content)
            
            response = "".join(response_chunks)
            
            # 텍스트 모드일 때는 JSON 파싱 건너뛰기
            if is_text_mode:
                return {
                    "userStories": [],
                    "actors": [],
                    "businessRules": [],
                    "boundedContexts": [],
                    "textResponse": response,  # 텍스트 응답 저장
                    "progress": 70,
                    "logs": [{
                        "timestamp": datetime.now().isoformat(),
                        "level": "info",
                        "message": "Generated text response (no JSON parsing)"
                    }]
                }
            
            # JSON 파싱
            response_clean = self._extract_json(response)
            
            result_data = json.loads(response_clean)
            
            # camelCase로 응답 받음
            user_stories = result_data.get("userStories", [])
            actors = result_data.get("actors", [])
            business_rules = result_data.get("businessRules", [])
            bounded_contexts = result_data.get("boundedContexts", [])
            
            return {
                "userStories": user_stories,
                "actors": actors,
                "businessRules": business_rules,
                "boundedContexts": bounded_contexts,
                "progress": 70,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "level": "info",
                    "message": f"Generated {len(user_stories)} user stories"
                }]
            }
            
        except Exception as e:
            error_msg = f"Failed to generate user stories: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "userStories": [],
                "actors": [],
                "businessRules": [],
                "boundedContexts": [],
                "error": error_msg,
                "progress": 70,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "level": "error",
                    "message": error_msg
                }]
            }
    
    def validate_user_stories(self, state: UserStoryState) -> Dict:
        """
        Step 3: 생성된 User Story 검증
        """
        
        user_stories = state.get("userStories", [])
        text_response = state.get("textResponse", None)
        
        # textResponse가 있으면 검증 건너뛰고 그대로 전달
        if text_response:
            return {
                "textResponse": text_response,
                "userStories": [],
                "progress": 90,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "level": "info",
                    "message": "Text mode response, skipping validation"
                }]
            }
        
        if not user_stories:
            return {
                "userStories": [],  # 빈 배열이라도 필드를 반환해야 함
                "progress": 90,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "level": "warning",
                    "message": "No user stories to validate"
                }]
            }
        
        # 검증 로직
        validated_stories = []
        issues = []
        
        for story in user_stories:
            # 필수 필드 확인 (camelCase)
            required_fields = ["as", "iWant", "soThat"]
            missing_fields = [f for f in required_fields if f not in story or not story[f]]
            
            if missing_fields:
                issues.append({
                    "story_id": story.get("id", "unknown"),
                    "issue": f"Missing required fields: {', '.join(missing_fields)}"
                })
            else:
                validated_stories.append(story)
        
        logs = [{
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "message": f"Validated {len(validated_stories)}/{len(user_stories)} user stories"
        }]
        
        if issues:
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "level": "warning",
                "message": f"Found {len(issues)} validation issues: {json.dumps(issues)}"
            })
        
        return {
            "userStories": validated_stories,
            "progress": 90,
            "logs": logs
        }
    
    def finalize_result(self, state: UserStoryState) -> Dict:
        """
        Step 4: 최종 결과 정리
        """
        
        # State의 모든 데이터를 유지하면서 완료 상태 추가 (camelCase)
        result = {
            "userStories": state.get("userStories", []),
            "actors": state.get("actors", []),
            "businessRules": state.get("businessRules", []),
            "boundedContexts": state.get("boundedContexts", []),
            "ragContext": state.get("rag_context", {}),
            "requirements": state.get("requirements", ""),
            "isCompleted": True,
            "progress": 100,
            "logs": state.get("logs", []) + [{
                "timestamp": datetime.now().isoformat(),
                "level": "info",
                "message": f"User story generation completed successfully. Total: {len(state.get('userStories', []))} stories"
            }]
        }
        
        # textResponse가 있으면 추가
        if state.get("textResponse"):
            result["textResponse"] = state.get("textResponse")
        
        return result
    
    def _is_virtual_scenario(self, requirements: str) -> bool:
        """
        가상 시나리오 모드 감지
        Frontend에서 생성한 가상 시나리오 프롬프트인지 확인
        """
        if not requirements:
            return False
        
        # 가상 시나리오 프롬프트의 특징적인 시작 부분 확인
        virtual_scenario_indicators = [
            "Please generate a comprehensive analysis",
            "Create pain points and possible solutions",
            "Persona definition and he(or her)'s painpoints"
        ]
        
        return any(indicator in requirements for indicator in virtual_scenario_indicators)
    
    def _format_existing_data(self, existing_actors, existing_user_stories, 
                              existing_business_rules, existing_bounded_contexts) -> str:
        """
        기존 청크의 누적 결과를 프롬프트에 포함 (중복 방지)
        RecursiveUserStoryGenerator의 createExistingDataPrompt와 동일
        """
        prompt = ""
        
        # 기존 액터 (title만)
        if existing_actors and len(existing_actors) > 0:
            actor_titles = [actor.get("title", "") for actor in existing_actors if actor.get("title")]
            if actor_titles:
                prompt += "Already Generated Actors: " + ", ".join(actor_titles) + "\n"
                prompt += "Create new actors if needed, but avoid duplicating existing ones.\n\n"
        
        # 기존 유저 스토리 (title만)
        if existing_user_stories and len(existing_user_stories) > 0:
            story_titles = [story.get("title", "") for story in existing_user_stories if story.get("title")]
            if story_titles:
                prompt += "Already Generated Topics (do not create related content):\n"
                prompt += ", ".join(story_titles) + "\n\n"
                prompt += "Focus on generating new, unrelated user stories and features.\n\n"
        
        # 기존 비즈니스 규칙 (title만)
        if existing_business_rules and len(existing_business_rules) > 0:
            rule_titles = [rule.get("title", "") for rule in existing_business_rules if rule.get("title")]
            if rule_titles:
                prompt += "Already Generated Business Rules: " + ", ".join(rule_titles) + "\n"
                prompt += "Create new rules if needed, but avoid duplicating existing ones.\n\n"
        
        # 기존 바운디드 컨텍스트 (name만)
        if existing_bounded_contexts and len(existing_bounded_contexts) > 0:
            bc_names = [bc.get("name", "") for bc in existing_bounded_contexts if bc.get("name")]
            if bc_names:
                prompt += "Already Generated Bounded Contexts: " + ", ".join(bc_names) + "\n"
                prompt += "Create new contexts if needed, but avoid duplicating existing ones.\n\n"
        
        if prompt:
            prompt += "IMPORTANT: When generating new content, ensure it complements and builds upon the existing data rather than duplicating it.\n\n"
        
        return prompt
    
    def _format_rag_context(self, rag_context: Dict) -> str:
        """RAG 컨텍스트를 프롬프트용 문자열로 포맷"""
        if not rag_context:
            return "NO RAG CONTEXT AVAILABLE (Knowledge base not initialized)"
        
        sections = []
        
        # User Story 패턴
        patterns = rag_context.get("user_story_patterns", [])
        if patterns:
            sections.append("USER STORY PATTERNS (from knowledge base):")
            for i, pattern in enumerate(patterns[:5], 1):  # 최대 5개
                sections.append(f"{i}. {pattern.get('content', '')[:200]}")
        
        # 유사 프로젝트
        projects = rag_context.get("similar_projects", [])
        if projects:
            sections.append("\nSIMILAR PROJECT EXAMPLES:")
            for i, project in enumerate(projects[:3], 1):  # 최대 3개
                sections.append(f"{i}. {project.get('content', '')[:300]}")
        
        # 도메인 용어
        vocab = rag_context.get("vocabulary", [])
        if vocab:
            sections.append("\nDOMAIN VOCABULARY:")
            for i, term in enumerate(vocab[:10], 1):  # 최대 10개
                sections.append(f"{i}. {term.get('content', '')[:100]}")
        
        return "\n".join(sections) if sections else "NO RAG CONTEXT AVAILABLE"
    
    def _extract_json(self, text: str) -> str:
        """텍스트에서 JSON 추출 (마크다운 제거)"""
        # ```json ... ``` 제거
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def run(self, inputs: Dict) -> Dict:
        """
        워크플로우 실행
        
        Args:
            inputs: {
                "requirements": str,
                "bounded_contexts": List[Dict] (optional)
            }
            
        Returns:
            {
                "user_stories": List[Dict],
                "progress": int,
                "logs": List[Dict],
                "is_completed": bool,
                "error": str
            }
        """
        initial_state: UserStoryState = {
            "requirements": inputs.get("requirements", ""),
            "bounded_contexts": inputs.get("bounded_contexts", []),
            "rag_context": {},
            "userStories": [],
            "actors": [],
            "businessRules": [],
            "boundedContexts": [],
            "progress": 0,
            "logs": [],
            "isCompleted": False,
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # 결과는 이미 camelCase 형식
            return result
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "userStories": [],
                "boundedContexts": [],
                "ragContext": {},
                "progress": 0,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "level": "error",
                    "message": error_msg
                }],
                "isCompleted": False,
                "isFailed": True,
                "error": error_msg,
                "requirements": inputs.get("requirements", "")
            }

