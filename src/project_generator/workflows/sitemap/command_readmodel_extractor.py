from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.project_generator.utils.logging_util import LoggingUtil
from src.project_generator.systems.firebase_system import FirebaseSystem
import json
import re

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
    current_chunk: int
    total_chunks: int
    chunks: list

def split_requirements_into_chunks(requirements: str, chunk_size: int = 12000) -> list:
    """
    요구사항을 청크로 분할 (문장 단위로 분할하여 문맥 유지)
    """
    # 문장 단위로 분할 (., !, ?, \n\n 기준)
    sentences = re.split(r'(?<=[.!?\n])\s+', requirements)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때 크기 확인
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # 마지막 청크 추가
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [requirements]

def merge_extracted_data(existing_data: dict, new_data: dict) -> dict:
    """
    기존 추출 데이터와 새로운 추출 데이터를 병합
    """
    if not existing_data or not existing_data.get("boundedContexts"):
        return new_data
    
    merged_bcs = {bc["name"]: bc for bc in existing_data.get("boundedContexts", [])}
    
    for new_bc in new_data.get("boundedContexts", []):
        bc_name = new_bc["name"]
        
        if bc_name in merged_bcs:
            # 기존 BC에 commands와 readModels 추가 (중복 제거)
            existing_bc = merged_bcs[bc_name]
            
            # Commands 병합
            existing_command_names = {cmd["name"] for cmd in existing_bc.get("commands", [])}
            new_commands = [cmd for cmd in new_bc.get("commands", []) if cmd["name"] not in existing_command_names]
            existing_bc["commands"] = existing_bc.get("commands", []) + new_commands
            
            # ReadModels 병합
            existing_readmodel_names = {rm["name"] for rm in existing_bc.get("readModels", [])}
            new_readmodels = [rm for rm in new_bc.get("readModels", []) if rm["name"] not in existing_readmodel_names]
            existing_bc["readModels"] = existing_bc.get("readModels", []) + new_readmodels
        else:
            # 새로운 BC 추가
            merged_bcs[bc_name] = new_bc
    
    return {"boundedContexts": list(merged_bcs.values())}

def extract_commands_and_readmodels(state: CommandReadModelState) -> CommandReadModelState:
    """
    요구사항에서 Command와 ReadModel 추출 (청크별 순차 처리)
    """
    try:
        # 첫 호출 시 청크 분할
        if state.get("current_chunk", 0) == 0:
            requirements = state["requirements"]
            
            # 요구사항 길이 체크 및 청크 분할 결정
            if len(requirements) > 12000:
                chunks = split_requirements_into_chunks(requirements, chunk_size=12000)
                LoggingUtil.info(state["job_id"], f"Requirements split into {len(chunks)} chunks")
                state["chunks"] = chunks
                state["total_chunks"] = len(chunks)
                state["current_chunk"] = 0
                state["extracted_data"] = {"boundedContexts": []}
            else:
                # 단일 청크로 처리
                state["chunks"] = [requirements]
                state["total_chunks"] = 1
                state["current_chunk"] = 0
                state["extracted_data"] = {"boundedContexts": []}
        
        current_chunk_index = state["current_chunk"]
        total_chunks = state["total_chunks"]
        chunk_text = state["chunks"][current_chunk_index]
        
        LoggingUtil.info(state["job_id"], f"Processing chunk {current_chunk_index + 1}/{total_chunks}...")
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3
        )
        
        bounded_contexts_json = json.dumps(state["bounded_contexts"], ensure_ascii=False, indent=2)
        
        # 기존 추출 데이터가 있으면 프롬프트에 추가
        existing_data_prompt = ""
        if state["extracted_data"].get("boundedContexts"):
            existing_data_prompt = f"""
IMPORTANT - ALREADY EXTRACTED DATA:
The following Commands and ReadModels have already been extracted from previous chunks.
DO NOT duplicate them. Only extract NEW operations from the current chunk.

Already Extracted:
{json.dumps(state["extracted_data"], ensure_ascii=False, indent=2)}
"""
        
        prompt = f"""You are an expert DDD architect. Extract Commands and ReadModels (Views) from user requirements and organize them by Bounded Context.

{existing_data_prompt}

CURRENT REQUIREMENTS CHUNK ({current_chunk_index + 1}/{total_chunks}):
{chunk_text}

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
            new_extracted_data = parsed_json.get('extractedData', {})
            
            # 기존 데이터와 병합
            merged_data = merge_extracted_data(state["extracted_data"], new_extracted_data)
            
            LoggingUtil.info(state["job_id"], f"Chunk {current_chunk_index + 1}/{total_chunks} processed: {len(new_extracted_data.get('boundedContexts', []))} BCs")
            
            # 다음 청크 처리 여부 결정
            next_chunk = current_chunk_index + 1
            progress = int((next_chunk / total_chunks) * 80)  # 80%까지는 추출 단계
            
            # Firebase에 중간 결과 업데이트 (UI에 실시간 반영)
            try:
                firebase = FirebaseSystem.instance()
                job_path = f"jobs/command_readmodel_extractor/{state['job_id']}"
                
                # 중간 결과를 Firebase에 업데이트 (outputs 경로에 저장)
                output_path = f"{job_path}/state/outputs"
                firebase.update_data(output_path, {
                    "extractedData": merged_data,
                    "progress": progress,
                    "logs": state["logs"] + [f"Chunk {current_chunk_index + 1}/{total_chunks} completed"]
                })
                LoggingUtil.info(state["job_id"], f"Firebase updated with chunk {current_chunk_index + 1}/{total_chunks} results")
            except Exception as e:
                LoggingUtil.info(state["job_id"], f"Failed to update Firebase: {str(e)}")
            
            if next_chunk < total_chunks:
                # 다음 청크 처리
                return {
                    **state,
                    "extracted_data": merged_data,
                    "current_chunk": next_chunk,
                    "logs": [f"Chunk {next_chunk}/{total_chunks} extraction in progress"],
                    "progress": progress
                }
            else:
                # 모든 청크 처리 완료
                return {
                    **state,
                    "extracted_data": merged_data,
                    "logs": [f"All chunks processed. Total {len(merged_data.get('boundedContexts', []))} bounded contexts"],
                    "progress": 80
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

def should_continue_extraction(state: CommandReadModelState) -> str:
    """
    청크 처리를 계속할지 결정
    
    IMPORTANT: current_chunk는 이미 처리된 청크의 인덱스
    다음 청크가 존재하는지 확인하려면 current_chunk + 1 < total_chunks
    """
    if state.get("is_failed"):
        return "finalize_result"
    
    # extract_commands_and_readmodels 함수가 다음 청크 인덱스를 이미 설정했는지 확인
    # 로그에서 "All chunks processed"가 있으면 완료된 것
    logs = state.get("logs", [])
    if logs and any("All chunks processed" in log for log in logs):
        return "finalize_result"
    
    current_chunk = state.get("current_chunk", 0)
    total_chunks = state.get("total_chunks", 1)
    
    # current_chunk는 다음에 처리할 청크의 인덱스
    # 예: chunk 0 처리 후 current_chunk = 1 설정됨
    if current_chunk < total_chunks:
        return "extract_commands_and_readmodels"
    else:
        return "finalize_result"

def create_command_readmodel_workflow():
    """
    Command/ReadModel 추출 워크플로우 생성 (루프 지원)
    """
    workflow = StateGraph(CommandReadModelState)
    
    # 노드 추가
    workflow.add_node("extract_commands_and_readmodels", extract_commands_and_readmodels)
    workflow.add_node("finalize_result", finalize_result)
    
    # 엣지 추가
    workflow.set_entry_point("extract_commands_and_readmodels")
    
    # 조건부 엣지: 다음 청크가 있으면 extract로, 없으면 finalize로
    workflow.add_conditional_edges(
        "extract_commands_and_readmodels",
        should_continue_extraction,
        {
            "extract_commands_and_readmodels": "extract_commands_and_readmodels",
            "finalize_result": "finalize_result"
        }
    )
    
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()

