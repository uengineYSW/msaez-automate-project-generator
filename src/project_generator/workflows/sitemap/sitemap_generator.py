from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.project_generator.utils.logging_util import LoggingUtil
import json

class SiteMapState(TypedDict):
    """SiteMap 생성 워크플로우 상태"""
    job_id: str
    requirements: str
    bounded_contexts: list
    command_readmodel_data: dict
    existing_navigation: list
    site_map: dict
    logs: Annotated[list, lambda x, y: x + y]
    progress: int
    is_completed: bool
    is_failed: bool
    error: str

def generate_sitemap(state: SiteMapState) -> SiteMapState:
    """
    Command/ReadModel 데이터를 기반으로 SiteMap 생성
    """
    try:
        LoggingUtil.info(state["job_id"], "Starting SiteMap generation...")
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3
        )
        
        # 'ui' BC 제외
        filtered_bcs = [bc for bc in state["bounded_contexts"] if bc.get("name") != "ui"]
        bounded_contexts_json = json.dumps(filtered_bcs, ensure_ascii=False, indent=2)
        command_readmodel_json = json.dumps(state["command_readmodel_data"], ensure_ascii=False, indent=2)
        
        # 기존 Navigation이 있는 경우 프롬프트 추가
        existing_navigation_prompt = ""
        if state.get("existing_navigation") and len(state["existing_navigation"]) > 0:
            existing_titles = extract_existing_titles(state["existing_navigation"])
            existing_navigation_prompt = f"""
ALREADY GENERATED PAGES (DO NOT DUPLICATE):
{', '.join(existing_titles)}

IMPORTANT: 
- Only generate NEW pages and components from the current requirements chunk
- Do not regenerate pages that are already in the list above
- If a requirement relates to an existing page, skip it
- Focus only on extracting NEW content from the current chunk
"""
        
        prompt = f"""You are an expert UX designer and web architect. Generate a comprehensive website sitemap JSON structure based on user requirements.

REQUIREMENTS:
{state["requirements"]}

{existing_navigation_prompt}

BOUNDED CONTEXTS:
Bounded Contexts List: {bounded_contexts_json}
Command/ReadModel Data: {command_readmodel_json}

TASK:
Generate a user-friendly website sitemap that represents actual web pages and sections.
Focus on user experience and logical page organization, not technical implementation.
Create a hierarchical structure that makes sense to end users.

COMMAND/READMODEL INTEGRATION:
CRITICAL: You MUST use the provided Command/ReadModel data to create accurate UI components with proper references.

COMMAND/READMODEL TYPES:

Command Types (Business Logic):
- Creation Commands: Create[Entity], Register[Entity], Add[Entity]
- Update Commands: Update[Entity], Modify[Entity], Change[Entity]
- Deletion Commands: Delete[Entity], Remove[Entity], Cancel[Entity]
- Process Commands: Process[Action], Confirm[Entity], Validate[Entity]
- Integration Commands: Sync[Entity], Notify[Entity], Detect[Condition]

ReadModel Types (Query Operations):
- Entity Queries: [Entity]Detail, [Entity]Profile, [Entity]Info
- List Queries: [Entity]List, [Entity]History, [Entity]Search
- Admin Queries: [Entity]ListAdmin, [Entity]Statistics, [Entity]Report
- UI Support: [Entity]FormView, [Entity]SearchView, [Entity]Options
- Status Queries: [Entity]Status, [Entity]Availability, [Entity]Alert

REFERENCE RULES:
- Each component must reference exactly ONE Command or ReadModel from the provided data
- Use the actual Command/ReadModel names from the Command/ReadModel Data section
- Exclude pure UI components (Header, Footer, Navbar) - focus on business logic components only

STRUCTURE GUIDELINES:
- Home page as the root with main navigation sections as children
- Main sections (e.g., Rooms, Amenities, Contact Us) as direct children of Home
- Each main section contains business logic components only
- Page components are organized vertically under each main section
- Clear 3-level hierarchy: Home → Main Pages → Business Components
- Focus on actual website structure with Home as entry point
- Group business logic components under their parent page for better organization
- Each component should have only ONE reference (single Command or ReadModel)
- Exclude pure UI components (Header, Footer, Navbar) - focus on business logic components only
- Create separate components for different business operations (e.g., separate login form and register form)

OUTPUT FORMAT:
{{
  "siteMap": {{
    "title": "Website Title",
    "description": "Website purpose and description",
    "pages": [
      {{
        "id": "page-id",
        "title": "Page Title",
        "url": "/page-url",
        "description": "Page description",
        "children": [
          {{
            "id": "component-id",
            "title": "Component Title",
            "url": "",
            "description": "Component description",
            "reference": "CommandOrReadModelName"
          }}
        ]
      }}
    ]
  }}
}}

RULES:
- Create logical page structure based on requirements
- Each component must reference exactly ONE Command or ReadModel
- Exclude pure UI components (Header, Footer, Navbar)
- Return ONLY JSON, no explanations"""

        response = llm.invoke(prompt)
        response_text = response.content
        
        # JSON 파싱
        try:
            # JSON 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed_json = json.loads(response_text)
            
            # siteMap 필드 추출 (LLM이 { "siteMap": {...} } 형식으로 반환)
            site_map_data = parsed_json.get('siteMap', {})
            
            LoggingUtil.info(state["job_id"], "SiteMap generation completed successfully")
            
            return {
                **state,
                "site_map": site_map_data,
                "logs": [f"SiteMap generation completed"],
                "progress": 50
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            LoggingUtil.info(state["job_id"], error_msg)
            return {
                **state,
                "is_failed": True,
                "error": error_msg,
                "logs": [error_msg]
            }
            
    except Exception as e:
        error_msg = f"Error in generate_sitemap: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": [error_msg]
        }

def extract_existing_titles(navigation):
    """
    기존 Navigation에서 모든 제목 추출
    """
    titles = []
    
    def extract_recursive(nodes):
        for node in nodes:
            if node.get("title"):
                titles.append(node["title"])
            if node.get("children") and isinstance(node["children"], list):
                extract_recursive(node["children"])
    
    extract_recursive(navigation)
    return titles

def finalize_result(state: SiteMapState) -> SiteMapState:
    """
    최종 결과 구성
    """
    try:
        LoggingUtil.info(state["job_id"], "Finalizing SiteMap generation result...")
        
        return {
            **state,
            "site_map": state["site_map"],
            "logs": state["logs"] + ["Finalization completed"],
            "progress": 100,
            "is_completed": True
        }
        
    except Exception as e:
        error_msg = f"Error in finalize_result: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": state["logs"] + [error_msg]
        }

def create_sitemap_workflow():
    """
    SiteMap 생성 워크플로우 생성
    """
    workflow = StateGraph(SiteMapState)
    
    # 노드 추가
    workflow.add_node("generate_sitemap", generate_sitemap)
    workflow.add_node("finalize_result", finalize_result)
    
    # 엣지 추가
    workflow.set_entry_point("generate_sitemap")
    workflow.add_edge("generate_sitemap", "finalize_result")
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()

