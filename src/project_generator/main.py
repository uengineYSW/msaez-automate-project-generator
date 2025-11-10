import asyncio
import concurrent.futures
import threading
import json
import math
import copy
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from typing import List

from project_generator.utils import JobUtil, DecentralizedJobManager
from project_generator.systems.firebase_system import FirebaseSystem
from project_generator.config import Config
from project_generator.run_healcheck_server import run_healcheck_server
from project_generator.simple_autoscaler import start_autoscaler
from project_generator.utils.logging_util import LoggingUtil

# Workflow imports
from project_generator.workflows.user_story.user_story_generator import UserStoryWorkflow
from project_generator.workflows.summarizer.requirements_summarizer import RequirementsSummarizerWorkflow
from project_generator.workflows.bounded_context.bounded_context_generator import BoundedContextWorkflow
from project_generator.workflows.sitemap.command_readmodel_extractor import create_command_readmodel_workflow
from project_generator.workflows.sitemap.sitemap_generator import create_sitemap_workflow
from project_generator.workflows.aggregate_draft.requirements_mapper import RequirementsMappingWorkflow
from project_generator.workflows.aggregate_draft.aggregate_draft_generator import AggregateDraftGenerator
from project_generator.workflows.aggregate_draft.preview_fields_generator import PreviewFieldsGenerator
from project_generator.workflows.aggregate_draft.ddl_fields_generator import DDLFieldsGenerator
from project_generator.workflows.aggregate_draft.traceability_generator import TraceabilityGenerator
from project_generator.workflows.aggregate_draft.ddl_extractor import DDLExtractor
from project_generator.workflows.requirements_validation.requirements_validator import RequirementsValidator

# 전역 job_manager 인스턴스
_current_job_manager: DecentralizedJobManager = None


def _compute_intermediate_lengths(final_length: int, steps: int = 3) -> List[int]:
    """
    최종 생성 길이를 기반으로 중간 길이 리스트를 계산.
    스트리밍이 어려운 워크플로우에서 주기적 진행률 업데이트 용도로 사용.
    """
    if final_length <= 0 or steps <= 0:
        return []

    lengths = set()
    for idx in range(1, steps + 1):
        length = max(1, min(final_length - 1, (final_length * idx) // (steps + 1)))
        lengths.add(length)

    intermediate = sorted(lengths)
    return intermediate


async def main():
    """메인 함수 - Flask 서버, Job 모니터링, 자동 스케일러 동시 시작"""
    
    flask_thread = None
    restart_count = 0
    
    while True:
        tasks = []
        job_manager = None
        
        try:
            
            # Flask 서버 시작 (첫 실행시에만)
            if flask_thread is None:
                flask_thread = threading.Thread(target=run_healcheck_server, daemon=True)
                flask_thread.start()
                LoggingUtil.info("main", "Flask 서버가 포트 2024에서 시작되었습니다.")
                LoggingUtil.info("main", "헬스체크 엔드포인트: http://localhost:2024/ok")

            if restart_count > 0:
                LoggingUtil.info("main", f"메인 함수 재시작 중... (재시작 횟수: {restart_count})")

            pod_id = Config.get_pod_id()
            job_manager = DecentralizedJobManager(pod_id, process_job_async)
            
            # 전역 job_manager 설정
            global _current_job_manager
            _current_job_manager = job_manager
            
            # 감시할 namespace 목록
            monitored_namespaces = ['user_story_generator', 'summarizer', 'bounded_context', 'command_readmodel_extractor', 'sitemap_generator', 'requirements_mapper', 'aggregate_draft_generator', 'preview_fields_generator', 'ddl_fields_generator', 'traceability_generator', 'ddl_extractor', 'requirements_validator']
            
            if Config.is_local_run():
                tasks.append(asyncio.create_task(job_manager.start_job_monitoring(monitored_namespaces)))
                LoggingUtil.info("main", "작업 모니터링이 시작되었습니다.")
            else:
                tasks.append(asyncio.create_task(start_autoscaler()))
                tasks.append(asyncio.create_task(job_manager.start_job_monitoring(monitored_namespaces)))
                LoggingUtil.info("main", "자동 스케일러 및 작업 모니터링이 시작되었습니다.")
            
            
            # shutdown_event 모니터링 태스크 추가
            shutdown_monitor_task = asyncio.create_task(job_manager.shutdown_event.wait())
            tasks.append(shutdown_monitor_task)
            
            # 태스크들 중 하나라도 완료되면 종료 (shutdown_event 포함)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # shutdown_event가 설정되었는지 확인
            if shutdown_monitor_task in done:
                LoggingUtil.info("main", "Graceful shutdown 신호 수신. 메인 루프를 종료합니다.")
                
                # 나머지 실행 중인 태스크들 취소
                for task in pending:
                    if not task.done():
                        LoggingUtil.debug("main", f"태스크 취소 중: {task}")
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            LoggingUtil.debug("main", "태스크가 정상적으로 취소되었습니다.")
                        except Exception as cleanup_error:
                            LoggingUtil.exception("main", "태스크 정리 중 예외 발생", cleanup_error)
                
                LoggingUtil.info("main", "메인 함수 정상 종료")
                break  # while 루프 종료
            
        except Exception as e:
            restart_count += 1
            LoggingUtil.exception("main", f"메인 함수에서 예외 발생 (재시작 횟수: {restart_count})", e)
            
            # 실행 중인 태스크들 정리
            for task in tasks:
                if not task.done():
                    LoggingUtil.debug("main", f"태스크 취소 중: {task}")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        LoggingUtil.debug("main", "태스크가 정상적으로 취소되었습니다.")
                    except Exception as cleanup_error:
                        LoggingUtil.exception("main", "태스크 정리 중 예외 발생", cleanup_error)

            continue


async def process_summarizer_job(job_id: str, complete_job_func: callable):
    """Summarizer Job 처리 함수"""
    error_occurred = None
    try:
        LoggingUtil.info("main", f"🚀 Summarizer 처리 시작: {job_id}")
        
        # Job 데이터 로딩
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            job_path = f'jobs/summarizer/{job_id}'
            job_data = await loop.run_in_executor(
                executor,
                lambda: FirebaseSystem.instance().get_data(job_path)
            )
        
        if not job_data:
            LoggingUtil.warning("main", f"Job 데이터 없음: {job_id}")
            return
        
        inputs = job_data.get("state", {}).get("inputs", {})
        if not inputs:
            LoggingUtil.warning("main", f"Job inputs 없음: {job_id}")
            return
        
        # SummarizerWorkflow 실행
        workflow = RequirementsSummarizerWorkflow()
        result = await asyncio.to_thread(workflow.run, inputs)
        
        summaries = result.get('summarizedRequirements', [])
        LoggingUtil.info("main", f"✅ 요약 완료: {len(summaries)}개")
        
        # 결과를 Firebase에 저장
        output_path = f'jobs/summarizer/{job_id}/state/outputs'
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            result
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/summarizer/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        error_occurred = e
        LoggingUtil.exception("main", f"Summarizer Job 처리 오류: {job_id}", e)
        
        # 실패 상태 저장
        try:
            error_output = {
                "summarizedRequirements": [],
                "isCompleted": False,
                "error": str(e),
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": f"오류: {str(e)}"
                }]
            }
            
            output_path = f'jobs/summarizer/{job_id}/state/outputs'
            await asyncio.to_thread(
                FirebaseSystem.instance().set_data,
                output_path,
                error_output
            )
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        # 예외 발생 여부와 관계없이 complete_job_func 호출
        complete_job_func()

async def process_user_story_job(job_id: str, complete_job_func: callable):
    """UserStory Job 처리 함수"""
    try:
        LoggingUtil.info("main", f"🚀 UserStory 처리 시작: {job_id}")
        
        # Job 데이터 로딩 (user_story_generator namespace 사용)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            job_path = f'jobs/user_story_generator/{job_id}'
            job_data = await loop.run_in_executor(
                executor,
                lambda: FirebaseSystem.instance().get_data(job_path)
            )
        
        if not job_data:
            LoggingUtil.warning("main", f"Job 데이터 없음: {job_id}")
            return
        
        inputs = job_data.get("state", {}).get("inputs", {})
        if not inputs:
            LoggingUtil.warning("main", f"Job inputs 없음: {job_id}")
            return
        
        # UserStoryWorkflow 실행
        workflow = UserStoryWorkflow()
        result = await asyncio.to_thread(workflow.run, inputs)
        
        # 결과는 이미 camelCase로 변환되어 있음
        user_stories = result.get('userStories', [])
        actors = result.get('actors', [])
        business_rules = result.get('businessRules', [])
        LoggingUtil.info("main", f"✅ 생성 완료: Stories {len(user_stories)}, Actors {len(actors)}, Rules {len(business_rules)}")
        
        # 결과를 Firebase에 저장 (비동기 처리)
        output_path = f'jobs/user_story_generator/{job_id}/state/outputs'
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            result
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/user_story_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"처리 오류: {job_id}", e)
        
        # 실패 기록
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'userStories': [],  # camelCase
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/user_story_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        # 예외 발생 여부와 관계없이 complete_job_func 호출
        complete_job_func()

async def process_bounded_context_job(job_id: str, complete_job_func: callable):
    """Bounded Context 생성 Job 처리"""
    
    try:
        # Job 데이터 로드
        job_path = f'jobs/bounded_context/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출 (state.inputs에서 가져옴)
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'devisionAspect': inputs_data.get('devisionAspect', ''),
            'requirements': inputs_data.get('requirements', {}),
            'generateOption': inputs_data.get('generateOption', {}),
            'feedback': inputs_data.get('feedback'),
            'previousAspectModel': inputs_data.get('previousAspectModel')
        }
        
        # 워크플로우 실행
        workflow = BoundedContextWorkflow()
        result = await asyncio.to_thread(workflow.run, inputs)
        
        output_path = f'jobs/bounded_context/{job_id}/state/outputs'
        firebase = FirebaseSystem.instance()

        try:
            final_length = len(json.dumps(result, ensure_ascii=False))
        except Exception:
            final_length = 0

        intermediate_lengths = _compute_intermediate_lengths(final_length, steps=3)

        for idx, length in enumerate(intermediate_lengths):
            progress_value = max(1, min(95, int(((idx + 1) / (len(intermediate_lengths) + 1)) * 100)))
            update_payload = {
                'currentGeneratedLength': length,
                'progress': progress_value,
                'isCompleted': False
            }
            await firebase.update_data_async(
                output_path,
                firebase.sanitize_data_for_firebase(update_payload)
            )
            await asyncio.sleep(1)

        result_with_length = copy.deepcopy(result)
        result_with_length['currentGeneratedLength'] = final_length

        await asyncio.to_thread(
            firebase.set_data,
            output_path,
            firebase.sanitize_data_for_firebase(result_with_length)
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/bounded_context/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 BC 생성 완료: {job_id}, BCs: {len(result.get('boundedContexts', []))}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        error_occurred = e
        LoggingUtil.exception("main", f"BC 생성 오류: {job_id}", e)
        
        # 실패 기록
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'thoughts': '',
                'boundedContexts': [],
                'relations': [],
                'explanations': [],
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/bounded_context/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        # 예외 발생 여부와 관계없이 complete_job_func 호출
        complete_job_func()

async def process_command_readmodel_job(job_id: str, complete_job_func: callable):
    """Command/ReadModel 추출 Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 Command/ReadModel 추출 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/command_readmodel_extractor/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'job_id': job_id,
            'requirements': inputs_data.get('requirements', ''),
            'bounded_contexts': inputs_data.get('boundedContexts', []),
            'logs': [],
            'progress': 0,
            'is_completed': False,
            'is_failed': False,
            'error': '',
            'extracted_data': {}
        }
        
        # 워크플로우 실행 (recursion_limit 증가)
        workflow = create_command_readmodel_workflow()
        result = await asyncio.to_thread(
            workflow.invoke, 
            inputs,
            {"recursion_limit": 50}
        )
        
        # 결과 저장
        output_path = f'jobs/command_readmodel_extractor/{job_id}/state/outputs'
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            {
                'extractedData': result.get('extracted_data', {}),
                'logs': result.get('logs', []),
                'progress': result.get('progress', 0),
                'isCompleted': result.get('is_completed', False),
                'isFailed': result.get('is_failed', False),
                'error': result.get('error', '')
            }
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/command_readmodel_extractor/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 Command/ReadModel 추출 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        error_occurred = e
        LoggingUtil.exception("main", f"Command/ReadModel 추출 오류: {job_id}", e)
        
        # 실패 기록
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'extractedData': {},
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/command_readmodel_extractor/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()

async def process_sitemap_job(job_id: str, complete_job_func: callable):
    """SiteMap 생성 Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 SiteMap 생성 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/sitemap_generator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'job_id': job_id,
            'requirements': inputs_data.get('requirements', ''),
            'bounded_contexts': inputs_data.get('boundedContexts', []),
            'command_readmodel_data': inputs_data.get('commandReadModelData', {}),
            'existing_navigation': inputs_data.get('existingNavigation', []),
            'logs': [],
            'progress': 0,
            'is_completed': False,
            'is_failed': False,
            'error': '',
            'site_map': {}
        }
        
        # 워크플로우 실행
        workflow = create_sitemap_workflow()
        result = await asyncio.to_thread(
            workflow.invoke, 
            inputs,
            {"recursion_limit": 50}
        )
        
        output_path = f'jobs/sitemap_generator/{job_id}/state/outputs'
        firebase = FirebaseSystem.instance()

        try:
            final_length = len(json.dumps(result.get('site_map', {}), ensure_ascii=False))
        except Exception:
            final_length = 0

        intermediate_lengths = _compute_intermediate_lengths(final_length, steps=3)

        for idx, length in enumerate(intermediate_lengths):
            progress_value = max(1, min(95, int(((idx + 1) / (len(intermediate_lengths) + 1)) * 100)))
            update_payload = {
                'currentGeneratedLength': length,
                'progress': progress_value,
                'isCompleted': False
            }
            await firebase.update_data_async(
                output_path,
                firebase.sanitize_data_for_firebase(update_payload)
            )
            await asyncio.sleep(1)

        final_output = {
            'siteMap': result.get('site_map', {}),
            'logs': result.get('logs', []),
            'progress': result.get('progress', 0),
            'isCompleted': result.get('is_completed', False),
            'isFailed': result.get('is_failed', False),
            'error': result.get('error', ''),
            'currentGeneratedLength': final_length
        }

        await asyncio.to_thread(
            firebase.set_data,
            output_path,
            firebase.sanitize_data_for_firebase(final_output)
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/sitemap_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 SiteMap 생성 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        error_occurred = e
        LoggingUtil.exception("main", f"SiteMap 생성 오류: {job_id}", e)
        
        # 실패 기록
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'siteMap': {},
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/sitemap_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()

async def process_requirements_mapping_job(job_id: str, complete_job_func: callable):
    """Requirements Mapping Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 Requirements Mapping 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/requirements_mapper/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'bounded_context': inputs_data.get('boundedContext', {}),
            'requirement_chunk': inputs_data.get('requirementChunk', {}),
            'relevant_requirements': [],
            'progress': 0,
            'logs': [],
            'is_completed': False,
            'error': ''
        }
        
        # 워크플로우 실행
        workflow = RequirementsMappingWorkflow()
        result = workflow.run(inputs)
        
        # 결과를 Firebase에 저장
        bounded_context = inputs_data.get('boundedContext', {}) or {}
        bc_name = bounded_context.get('name', '')
        
        output = {
            'boundedContext': bc_name,
            'requirements': result.get('relevant_requirements', []),
            'isCompleted': result.get('is_completed', True),
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        output_path = f'{job_path}/state/outputs'
        # Firebase에 저장하기 전에 데이터 정제
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/requirements_mapper/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 Requirements Mapping 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"Requirements Mapping 오류: {job_id}", e)
        
        # 실패 기록
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'requirements': [],
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/requirements_mapper/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()

async def process_aggregate_draft_job(job_id: str, complete_job_func: callable):
    """Aggregate Draft Generation Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 Aggregate Draft 생성 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/aggregate_draft_generator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'bounded_context': inputs_data.get('boundedContext', {}),
            'description': inputs_data.get('description', ''),
            'accumulated_drafts': inputs_data.get('accumulatedDrafts', {}),
            'analysis_result': inputs_data.get('analysisResult', {})
        }
        
        # 워크플로우 실행
        generator = AggregateDraftGenerator()
        result = generator.run(inputs)
        
        # 결과를 Firebase에 저장
        output = {
            'inference': result.get('inference', ''),
            'options': result.get('options', []),
            'defaultOptionIndex': result.get('default_option_index', 1),
            'conclusions': result.get('conclusions', ''),
            'isCompleted': result.get('is_completed', True),
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        output_path = f'{job_path}/state/outputs'
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/aggregate_draft_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 Aggregate Draft 생성 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"Aggregate Draft 생성 오류: {job_id}", e)
        
        try:
            error_output = {
                'isFailed': True,
                'error': str(e),
                'progress': 0,
                'options': [],
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/aggregate_draft_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()


async def process_preview_fields_job(job_id: str, complete_job_func: callable):
    """Preview Fields Generation Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 Preview Fields 생성 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/preview_fields_generator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        inputs = {
            'description': inputs_data.get('description', ''),
            'aggregateDrafts': inputs_data.get('aggregateDrafts', []),
            'generatorKey': inputs_data.get('generatorKey', 'default'),
            'traceMap': inputs_data.get('traceMap', {})
        }
        
        # 워크플로우 실행
        generator = PreviewFieldsGenerator()
        result = generator.run(inputs)
        
        # 결과를 Firebase에 저장
        output = {
            'inference': result.get('inference', ''),
            'aggregateFieldAssignments': result.get('aggregateFieldAssignments', []),
            'isCompleted': result.get('isCompleted', True),
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        output_path = f'{job_path}/state/outputs'
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/preview_fields_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 Preview Fields 생성 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"Preview Fields 생성 오류: {job_id}", e)
        
        try:
            error_output = {
                'isFailed': True,
                'isCompleted': True,
                'progress': 100,
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/preview_fields_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()


async def process_ddl_fields_job(job_id: str, complete_job_func: callable):
    """DDL Fields Assignment Job 처리"""
    
    try:
        LoggingUtil.info("main", f"🚀 DDL Fields 할당 시작: {job_id}")
        
        # Job 데이터 로드
        job_path = f'jobs/ddl_fields_generator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        input_data = {
            'description': inputs_data.get('description', ''),
            'aggregate_drafts': inputs_data.get('aggregateDrafts', []),
            'all_ddl_fields': inputs_data.get('allDdlFields', []),
            'generator_key': inputs_data.get('generatorKey', 'default')
        }
        
        # 워크플로우 실행
        generator = DDLFieldsGenerator()
        result = generator.generate(input_data)
        
        # 결과를 Firebase에 저장
        output = {
            'inference': result.get('inference', ''),
            'aggregateFieldAssignments': result.get('result', {}).get('aggregateFieldAssignments', []),
            'isCompleted': True,
            'progress': 100,
            'logs': [{'timestamp': result.get('timestamp', ''), 'level': 'info', 'message': 'DDL fields assigned successfully'}]
        }
        
        output_path = f'{job_path}/state/outputs'
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/ddl_fields_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )
        
        LoggingUtil.info("main", f"🎉 DDL Fields 할당 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"DDL Fields 할당 오류: {job_id}", e)
        
        try:
            error_output = {
                'isFailed': True,
                'isCompleted': True,
                'progress': 100,
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/ddl_fields_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()


async def process_traceability_job(job_id: str, complete_job_func: callable):
    """Traceability Addition Job 처리"""
    try:
        LoggingUtil.info("main", f"🚀 Traceability 추가 시작: {job_id}")

        job_path = f'jobs/traceability_generator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )

        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return

        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})

        input_data = {
            'generatedDraftOptions': inputs_data.get('generatedDraftOptions', []),
            'boundedContextName': inputs_data.get('boundedContextName', ''),
            'description': inputs_data.get('description', ''),
            'functionalRequirements': inputs_data.get('functionalRequirements', ''),
            'traceMap': inputs_data.get('traceMap', {}),
        }

        generator = TraceabilityGenerator()
        result = generator.generate(input_data)

        output = {
            'inference': result.get('inference', ''),
            'draftTraceMap': result.get('draftTraceMap', {}),
            'isCompleted': True,
            'progress': 100,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Traceability mapping completed'}]
        }

        output_path = f'{job_path}/state/outputs'
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )

        req_path = f'requestedJobs/traceability_generator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )

        LoggingUtil.info("main", f"🎉 Traceability 추가 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")

    except Exception as e:
        LoggingUtil.exception("main", f"Traceability 추가 오류: {job_id}", e)
        try:
            error_output = {
                'isFailed': True,
                'isCompleted': True,
                'progress': 100,
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/traceability_generator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    finally:
        complete_job_func()


async def process_ddl_extractor_job(job_id: str, complete_job_func: callable):
    """DDL Extractor Job 처리"""
    try:
        LoggingUtil.info("main", f"🚀 DDL 필드 추출 시작: {job_id}")

        job_path = f'jobs/ddl_extractor/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )

        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return

        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})

        input_data = {
            'ddlRequirements': inputs_data.get('ddlRequirements', []),
            'boundedContextName': inputs_data.get('boundedContextName', ''),
        }

        generator = DDLExtractor()
        result = generator.generate(input_data)

        output = {
            'inference': result.get('inference', ''),
            'ddlFieldRefs': result.get('ddlFieldRefs', []),
            'isCompleted': True,
            'progress': 100,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'DDL extraction completed'}]
        }

        output_path = f'{job_path}/state/outputs'
        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )

        req_path = f'requestedJobs/ddl_extractor/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )

        LoggingUtil.info("main", f"🎉 DDL 필드 추출 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")

    except Exception as e:
        LoggingUtil.exception("main", f"DDL 추출 오류: {job_id}", e)
        try:
            error_output = {
                'isFailed': True,
                'isCompleted': True,
                'progress': 100,
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/ddl_extractor/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    finally:
        complete_job_func()


async def process_requirements_validator_job(job_id: str, complete_job_func: callable):
    """Requirements Validator Job 처리"""
    try:
        LoggingUtil.info("main", f"🚀 요구사항 검증 시작: {job_id}")

        job_path = f'jobs/requirements_validator/{job_id}'
        job_data = await asyncio.to_thread(
            FirebaseSystem.instance().get_data,
            job_path
        )

        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return

        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})

        input_data = {
            'requirements': inputs_data.get('requirements', {}),
            'previousChunkSummary': inputs_data.get('previousChunkSummary', {}),
            'currentChunkStartLine': inputs_data.get('currentChunkStartLine', 1),
        }

        generator = RequirementsValidator()
        result = generator.generate(input_data)

        output_path = f'{job_path}/state/outputs'
        firebase = FirebaseSystem.instance()

        content = result.get('content', {}) or {}
        final_length = 0
        try:
            final_length = len(json.dumps(content, ensure_ascii=False))
        except Exception:
            final_length = 0

        intermediate_lengths = _compute_intermediate_lengths(final_length, steps=3)

        for idx, length in enumerate(intermediate_lengths):
            progress_value = max(1, min(95, int(((idx + 1) / (len(intermediate_lengths) + 1)) * 100)))
            update_payload = {
                'currentGeneratedLength': length,
                'progress': progress_value,
                'isCompleted': False
            }
            await firebase.update_data_async(
                output_path,
                firebase.sanitize_data_for_firebase(update_payload)
            )
            await asyncio.sleep(1)

        output = {
            'type': result.get('type', 'ANALYSIS_RESULT'),
            'content': result.get('content', {}),
            'isCompleted': True,
            'progress': 100,
            'currentGeneratedLength': final_length,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Requirements validation completed'}]
        }

        sanitized_output = FirebaseSystem.instance().sanitize_data_for_firebase(output)
        await asyncio.to_thread(
            FirebaseSystem.instance().set_data,
            output_path,
            sanitized_output
        )

        req_path = f'requestedJobs/requirements_validator/{job_id}'
        await asyncio.to_thread(
            FirebaseSystem.instance().delete_data,
            req_path
        )

        LoggingUtil.info("main", f"🎉 요구사항 검증 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")

    except Exception as e:
        LoggingUtil.exception("main", f"요구사항 검증 오류: {job_id}", e)
        try:
            error_output = {
                'isFailed': True,
                'isCompleted': True,
                'progress': 100,
                'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'error', 'message': str(e)}]
            }
            output_path = f'jobs/requirements_validator/{job_id}/state/outputs'
            FirebaseSystem.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    finally:
        complete_job_func()

async def process_job_async(job_id: str, complete_job_func: callable):
    """비동기 Job 처리 함수 (Job ID prefix로 라우팅)"""
    
    try:
        LoggingUtil.debug("main", f"Job 시작: {job_id}")
        if not JobUtil.is_valid_job_id(job_id):
            LoggingUtil.warning("main", f"Job 처리 오류: {job_id}, 유효하지 않음")
            return
        
        # Job 타입별 라우팅 (각 함수에서 finally 블록으로 complete_job_func 호출)
        if job_id.startswith("usgen-"):
            await process_user_story_job(job_id, complete_job_func)
        elif job_id.startswith("summ-"):
            await process_summarizer_job(job_id, complete_job_func)
        elif job_id.startswith("bcgen-"):
            await process_bounded_context_job(job_id, complete_job_func)
        elif job_id.startswith("cmrext-"):
            await process_command_readmodel_job(job_id, complete_job_func)
        elif job_id.startswith("smapgen-"):
            await process_sitemap_job(job_id, complete_job_func)
        elif job_id.startswith("reqmap-"):
            await process_requirements_mapping_job(job_id, complete_job_func)
        elif job_id.startswith("aggr-draft-"):
            await process_aggregate_draft_job(job_id, complete_job_func)
        elif job_id.startswith("preview-fields-"):
            await process_preview_fields_job(job_id, complete_job_func)
        elif job_id.startswith("ddl-fields-"):
            await process_ddl_fields_job(job_id, complete_job_func)
        elif job_id.startswith("trace-add-"):
            await process_traceability_job(job_id, complete_job_func)
        elif job_id.startswith("ddl-extract-"):
            await process_ddl_extractor_job(job_id, complete_job_func)
        elif job_id.startswith("req-valid-"):
            await process_requirements_validator_job(job_id, complete_job_func)
        else:
            LoggingUtil.warning("main", f"지원하지 않는 Job 타입: {job_id}")
            
    except asyncio.CancelledError:
        LoggingUtil.debug("main", f"Job {job_id} 취소됨")
        return
        
    except Exception as e:
        LoggingUtil.exception("main", f"Job 처리 오류: {job_id}", e)

if __name__ == "__main__":
    asyncio.run(main())