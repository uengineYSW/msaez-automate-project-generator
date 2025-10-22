from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.project_generator.utils.logging_util import LoggingUtil
import json

class CommandReadModelState(TypedDict):
    """Command/ReadModel 추출 워크플로우 상태"""
    job_id: str
    requirements: str
    bounded_contexts: list
    extracted_data: dict
    logs: Annotated[list, lambda x, y: x + y]
    progress: int
    is_completed: bool
    is_failed: bool
    error: str

def extract_commands_and_readmodels(state: CommandReadModelState) -> CommandReadModelState:
    """
    요구사항에서 Command와 ReadModel 추출
    """
    try:
        LoggingUtil.info(state["job_id"], "Starting Command/ReadModel extraction...")
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3
        )
        
        bounded_contexts_json = json.dumps(state["bounded_contexts"], ensure_ascii=False, indent=2)
        
        prompt = f"""You are an expert DDD architect. Extract Commands and ReadModels (Views) from user requirements and organize them by Bounded Context.

REQUIREMENTS:
{state["requirements"]}

BOUNDED CONTEXTS:
{bounded_contexts_json}

TASK:
Extract all business operations (Commands) and query operations (ReadModels/Views) from the requirements and organize them by their corresponding Bounded Context.

EXTRACTION GUIDELINES:

## Command Extraction Rules (비즈니스 로직):
1. **상태 변경 작업**: 시스템의 상태나 데이터를 변경하는 모든 작업
   - 생성: CreateReservation, RegisterUser, CreateFlight
   - 수정: UpdateProfile, UpdateReservation, UpdateFlight
   - 삭제: CancelReservation, DeleteFlight, DeleteSeat
   - 처리: ProcessPayment, ConfirmReservation, VerifyEmail
2. **비즈니스 프로세스**: 특정 비즈니스 규칙을 수행하는 작업
   - 검증: ValidateReservation, AuthenticateUser
   - 계산: CalculatePrice, ComputeRefund
   - 통지: SendNotification, IssueAuthToken
3. **외부 시스템 연동**: 외부 시스템과의 상호작용
   - 동기화: SyncFlightInfo, SyncSeatInfo
   - 감지: DetectFraudulentReservation
4. **명명 규칙**: 동사 + 명사 (Verb + Noun)
5. **액터 식별**: user, admin, system, external

## ReadModel (View) Extraction Rules (조회 작업):
1. **데이터 조회**: 상태를 변경하지 않고 데이터를 가져오는 작업
   - 단일 조회: UserProfile, ReservationDetail, FlightDetail
   - 목록 조회: FlightList, ReservationHistory, InquiryList
2. **검색 및 필터링**: 조건에 따른 데이터 검색
   - 검색: FlightSearch, SearchReservations
   - 필터링: FilteredFlightList, AvailableSeats
3. **통계 및 보고서**: 집계 데이터나 요약 정보
   - 통계: ReservationStatistics, SalesReport
   - 현황: SeatAvailability, FlightStatus
4. **UI 지원 데이터**: 화면 구성에 필요한 데이터
   - 옵션 목록: AirportList, SeatClassOptions
   - 설정 정보: UserPreferences, SystemSettings
5. **명명 규칙**: 명사 + 목적 (Noun + Purpose)
6. **액터 식별**: user, admin, system

## Bounded Context Assignment:
1. **Domain Alignment**: Assign commands/views to the most appropriate Bounded Context based on domain responsibility
2. **Aggregate Alignment**: Consider which aggregates within each Bounded Context are most relevant
3. **Business Logic**: Group related operations within the same Bounded Context

OUTPUT FORMAT:
{{
  "extractedData": {{
    "boundedContexts": [
      {{
        "name": "BoundedContextName",
        "alias": "BoundedContextAlias",
        "commands": [
          {{
            "name": "CommandName",
            "alias": "CommandAlias",
            "description": "Command description",
            "actor": "user|admin|system",
            "aggregate": "AggregateName",
            "properties": [
              {{
                "name": "propertyName",
                "type": "String|Long|Integer|Boolean|Date",
                "description": "Property description"
              }}
            ]
          }}
        ],
        "readModels": [
          {{
            "name": "ReadModelName",
            "alias": "ReadModelAlias",
            "description": "ReadModel description",
            "actor": "user|admin|system",
            "aggregate": "AggregateName",
            "isMultipleResult": true|false,
            "queryParameters": [
              {{
                "name": "parameterName",
                "type": "String|Long|Integer|Boolean|Date",
                "description": "Parameter description"
              }}
            ]
          }}
        ]
      }}
    ]
  }}
}}

RULES:
- Extract ALL business operations from requirements
- Focus on domain-specific operations, not generic CRUD
- Ensure commands and views are properly categorized
- Use clear, descriptive names that reflect business intent
- Assign appropriate actors for each operation
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
            
            # extractedData 필드 추출 (LLM이 { "extractedData": {...} } 형식으로 반환)
            extracted_data = parsed_json.get('extractedData', {})
            
            LoggingUtil.info(state["job_id"], f"Successfully extracted {len(extracted_data.get('boundedContexts', []))} bounded contexts")
            
            return {
                **state,
                "extracted_data": extracted_data,
                "logs": [f"Command/ReadModel extraction completed"],
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
        error_msg = f"Error in extract_commands_and_readmodels: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": [error_msg]
        }

def finalize_result(state: CommandReadModelState) -> CommandReadModelState:
    """
    최종 결과 구성
    """
    try:
        LoggingUtil.info(state["job_id"], "Finalizing Command/ReadModel extraction result...")
        
        return {
            **state,
            "extracted_data": state["extracted_data"],
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

def create_command_readmodel_workflow():
    """
    Command/ReadModel 추출 워크플로우 생성
    """
    workflow = StateGraph(CommandReadModelState)
    
    # 노드 추가
    workflow.add_node("extract_commands_and_readmodels", extract_commands_and_readmodels)
    workflow.add_node("finalize_result", finalize_result)
    
    # 엣지 추가
    workflow.set_entry_point("extract_commands_and_readmodels")
    workflow.add_edge("extract_commands_and_readmodels", "finalize_result")
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()

