import asyncio
import concurrent.futures
import threading
import json
import math
import copy
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from typing import List

from project_generator.utils import JobUtil, DecentralizedJobManager
from project_generator.systems.storage_system_factory import StorageSystemFactory
from project_generator.config import Config

# StorageSystem 별칭 (호환성)
StorageSystem = StorageSystemFactory
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
from project_generator.utils.trace_markdown_util import TraceMarkdownUtil
from project_generator.workflows.aggregate_draft.preview_fields_generator import PreviewFieldsGenerator
from project_generator.workflows.aggregate_draft.ddl_fields_generator import DDLFieldsGenerator
from project_generator.workflows.aggregate_draft.traceability_generator import TraceabilityGenerator
from project_generator.workflows.aggregate_draft.ddl_extractor import DDLExtractor
from project_generator.workflows.aggregate_draft.standard_transformer import AggregateDraftStandardTransformer
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
            # Storage 시스템 초기화
            StorageSystemFactory.initialize()
            
            # Flask 서버 시작 (첫 실행시에만)
            if flask_thread is None:
                flask_thread = threading.Thread(target=run_healcheck_server, daemon=True)
                flask_thread.start()
                flask_port = os.getenv('FLASK_PORT', '2025')
                flask_host = os.getenv('FLASK_HOST', 'localhost')
                LoggingUtil.info("main", f"Flask 서버가 포트 {flask_port}에서 시작되었습니다.")
                LoggingUtil.info("main", f"헬스체크 엔드포인트: http://{flask_host}:{flask_port}/ok")

            if restart_count > 0:
                LoggingUtil.info("main", f"메인 함수 재시작 중... (재시작 횟수: {restart_count})")

            pod_id = Config.get_pod_id()
            job_manager = DecentralizedJobManager(pod_id, process_job_async)
            
            # 전역 job_manager 설정
            global _current_job_manager
            _current_job_manager = job_manager
            
            # 감시할 namespace 목록
            monitored_namespaces = ['user_story_generator', 'summarizer', 'bounded_context', 'command_readmodel_extractor', 'sitemap_generator', 'requirements_mapper', 'aggregate_draft_generator', 'preview_fields_generator', 'ddl_fields_generator', 'traceability_generator', 'standard_transformer', 'ddl_extractor', 'requirements_validator']
            
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
                lambda: StorageSystemFactory.instance().get_data(job_path)
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
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'jobs/summarizer/{job_id}/state/outputs'
        
        # 1) isCompleted 제외한 데이터 먼저 저장
        result_without_completed = {k: v for k, v in result.items() if k != 'isCompleted'}
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            result_without_completed
        )
        
        # 2) 짧은 대기 후 isCompleted 저장
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/summarizer/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
                StorageSystemFactory.instance().set_data,
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
                lambda: StorageSystemFactory.instance().get_data(job_path)
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
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'jobs/user_story_generator/{job_id}/state/outputs'
        
        # 1) isCompleted 제외한 데이터 먼저 저장
        result_without_completed = {k: v for k, v in result.items() if k != 'isCompleted'}
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            result_without_completed
        )
        
        # 2) 짧은 대기 후 isCompleted 저장 (이벤트 순서 보장)
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/user_story_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
        storage = StorageSystemFactory.instance()

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
            await storage.update_data_async(
                output_path,
                storage.sanitize_data_for_storage(update_payload)
            )
            await asyncio.sleep(1)

        result_with_length = copy.deepcopy(result)
        result_with_length['currentGeneratedLength'] = final_length

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        # 1) isCompleted 제외한 데이터 먼저 저장
        result_without_completed = {k: v for k, v in result_with_length.items() if k != 'isCompleted'}
        await asyncio.to_thread(
            storage.set_data,
            output_path,
            storage.sanitize_data_for_storage(result_without_completed)
        )
        
        # 2) 짧은 대기 후 isCompleted 저장
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            storage.update_data,
            output_path,
            {'isCompleted': True}
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/bounded_context/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'jobs/command_readmodel_extractor/{job_id}/state/outputs'
        
        # 1) isCompleted 제외한 데이터 먼저 저장
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            {
                'extractedData': result.get('extracted_data', {}),
                'logs': result.get('logs', []),
                'progress': result.get('progress', 0),
                'isFailed': result.get('is_failed', False),
                'error': result.get('error', '')
            }
        )
        
        # 2) 짧은 대기 후 isCompleted 저장
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': result.get('is_completed', False)}
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/command_readmodel_extractor/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
        storage = StorageSystemFactory.instance()

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
            await storage.update_data_async(
                output_path,
                storage.sanitize_data_for_storage(update_payload)
            )
            await asyncio.sleep(1)

        final_output = {
            'siteMap': result.get('site_map', {}),
            'logs': result.get('logs', []),
            'progress': result.get('progress', 0),
            'isFailed': result.get('is_failed', False),
            'error': result.get('error', ''),
            'currentGeneratedLength': final_length
        }

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        await asyncio.to_thread(
            storage.set_data,
            output_path,
            storage.sanitize_data_for_storage(final_output)
        )
        
        # 짧은 대기 후 isCompleted 저장
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            storage.update_data,
            output_path,
            {'isCompleted': result.get('is_completed', False)}
        )
        
        # requestedJob 삭제
        req_path = f'requestedJobs/sitemap_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': result.get('is_completed', True)}
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/requirements_mapper/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        bounded_context = inputs_data.get('boundedContext', {})
        
        # traceMap 생성 (프론트엔드와 동일한 로직)
        # boundedContext.requirements 배열이 있으면 traceMap 생성
        if bounded_context.get('requirements') and isinstance(bounded_context['requirements'], list):
            relations = inputs_data.get('relations', [])
            explanations = inputs_data.get('explanations', [])
            analysis_result = inputs_data.get('analysisResult', {})
            events = analysis_result.get('events', []) if isinstance(analysis_result, dict) else []
            
            # 원본 요구사항 구성 (traceMap 생성 시 원본 라인 길이 계산용)
            # inputs_data에서 직접 가져오거나, requirements 배열에서 추출
            original_requirements = inputs_data.get('originalRequirements', '')
            if not original_requirements:
                # requirements 배열에서 userStory와 ddl 추출
                user_story_parts = []
                ddl_parts = []
                for req in bounded_context['requirements']:
                    req_type = req.get('type', '').lower()
                    req_text = req.get('text', '')
                    if req_type == 'userstory' and req_text:
                        user_story_parts.append(req_text)
                    elif req_type == 'ddl' and req_text:
                        ddl_parts.append(req_text)
            try:
                # 프론트엔드와 동일: 원본 요구사항을 전달하지 않음
                bc_description_with_mapping = TraceMarkdownUtil.get_description_with_mapping_index(
                    bounded_context,
                    relations,
                    explanations,
                    events
                )
                
                # traceMap을 requirements에 추가
                if not isinstance(bounded_context.get('requirements'), dict):
                    # requirements가 배열인 경우, dict로 변환
                    requirements_dict = {
                        'traceMap': bc_description_with_mapping['traceMap'],
                        'description': bc_description_with_mapping['markdown']
                    }
                    # 기존 requirements 배열 정보도 유지
                    if bounded_context['requirements']:
                        requirements_dict['userStory'] = ''
                        requirements_dict['ddl'] = ''
                        requirements_dict['event'] = ''
                        # requirements 배열을 타입별로 분류
                        for req in bounded_context['requirements']:
                            req_type = req.get('type', '').lower()
                            req_text = req.get('text', '')
                            if req_type == 'userstory' and req_text:
                                requirements_dict['userStory'] += req_text + '\n\n'
                            elif req_type == 'ddl' and req_text:
                                requirements_dict['ddl'] += req_text + '\n\n'
                            elif req_type == 'event' and req_text:
                                requirements_dict['event'] += req_text + '\n\n'
                    
                    bounded_context['requirements'] = requirements_dict
                else:
                    # requirements가 이미 dict인 경우
                    bounded_context['requirements']['traceMap'] = bc_description_with_mapping['traceMap']
                    if 'description' not in bounded_context['requirements']:
                        bounded_context['requirements']['description'] = bc_description_with_mapping['markdown']
                
                LoggingUtil.info("main", f"✅ traceMap 생성 완료: {len(bc_description_with_mapping['traceMap'])} lines")
            except Exception as e:
                LoggingUtil.warning("main", f"⚠️ traceMap 생성 실패 (계속 진행): {e}")
                # traceMap 생성 실패해도 계속 진행
                if not isinstance(bounded_context.get('requirements'), dict):
                    bounded_context['requirements'] = {'traceMap': {}}
                elif 'traceMap' not in bounded_context['requirements']:
                    bounded_context['requirements']['traceMap'] = {}
        
        inputs = {
            'bounded_context': bounded_context,
            'description': inputs_data.get('description', ''),
            'accumulated_drafts': inputs_data.get('accumulatedDrafts', {}),
            'analysis_result': inputs_data.get('analysisResult', {})
        }
        
        # 워크플로우 실행
        generator = AggregateDraftGenerator()
        result = generator.run(inputs)
        
        # 결과를 Firebase에 저장
        # defaultOptionIndex: 1-based (LLM) → 0-based (프론트엔드)
        llm_default_index = result.get('default_option_index', 1)
        frontend_default_index = max(0, llm_default_index - 1)  # 1-based → 0-based
        
        output = {
            'inference': result.get('inference', ''),
            'options': result.get('options', []),
            'defaultOptionIndex': frontend_default_index,
            'conclusions': result.get('conclusions', ''),
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': result.get('is_completed', True)}
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/aggregate_draft_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
            job_path
        )
        
        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return
        
        # 입력 데이터 추출
        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})
        
        trace_map = inputs_data.get('traceMap', {})
        
        # traceMap 복원 (Firebase가 배열로 변환한 경우 처리)
        if isinstance(trace_map, list):
            LoggingUtil.warning("main", f"⚠️ Preview Fields: traceMap이 배열 형태입니다! 복원 중...")
            temp_generator = PreviewFieldsGenerator()
            trace_map = temp_generator._restore_trace_map(trace_map)
            LoggingUtil.info("main", f"✅ Preview Fields: traceMap 복원 완료, keys={len(trace_map) if isinstance(trace_map, dict) else 0}")
        elif isinstance(trace_map, dict):
            LoggingUtil.info("main", f"✅ Preview Fields: traceMap 구조 확인 (dict), keys={len(trace_map)}")
        
        # 프론트엔드 에이전트 방식과 동일하게 description만 사용
        inputs = {
            'description': inputs_data.get('description', ''),
            'aggregateDrafts': inputs_data.get('aggregateDrafts', []),
            'generatorKey': inputs_data.get('generatorKey', 'default'),
            'traceMap': trace_map,
            'originalRequirements': inputs_data.get('originalRequirements', '')  # 원본 요구사항 (userStory + ddl)
        }
        
        # 워크플로우 실행
        generator = PreviewFieldsGenerator()
        result = generator.run(inputs)
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output = {
            'inference': result.get('inference', ''),
            'aggregateFieldAssignments': result.get('aggregateFieldAssignments', []),
            'progress': result.get('progress', 100),
            'logs': result.get('logs', [])
        }
        
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': result.get('isCompleted', True)}
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/preview_fields_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
            'progress': 100,
            'logs': [{'timestamp': result.get('timestamp', ''), 'level': 'info', 'message': 'DDL fields assigned successfully'}]
        }
        
        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )
        
        # 요청 Job 제거
        req_path = f'requestedJobs/ddl_fields_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
        except Exception as save_error:
            LoggingUtil.exception("main", f"실패 저장 오류: {job_id}", save_error)
    
    finally:
        complete_job_func()


async def process_standard_transformation_job(job_id: str, complete_job_func: callable):
    """Standard Transformation Job 처리"""
    transformer = None  # 변수 스코프를 위해 함수 시작 부분에서 초기화
    try:
        LoggingUtil.info("main", f"🚀 표준 변환 시작: {job_id}")

        job_path = f'jobs/standard_transformer/{job_id}'
        job_data = await asyncio.to_thread(
            StorageSystemFactory.instance().get_data,
            job_path
        )

        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return

        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})

        draft_options = inputs_data.get('draftOptions', [])
        bounded_context = inputs_data.get('boundedContext', {})
        transformation_session_id = inputs_data.get('transformationSessionId', None)
        user_id = inputs_data.get('userId', None)

        # Storage 업데이트 콜백 함수 정의
        output_path = f'{job_path}/state/outputs'
        storage = StorageSystemFactory.instance()
        
        # 표준 변환기 실행
        # transformationSessionId가 있으면 디렉토리명으로 사용, 없으면 job_id 사용
        result_dir_name = transformation_session_id if transformation_session_id else job_id
        transformer = AggregateDraftStandardTransformer(enable_rag=True, user_id=user_id)
        
        async def storage_update_callback(update_data: dict):
            """Storage에 진행 상황 업데이트 (Firebase/AceBase 공통)"""
            try:
                sanitized_data = storage.sanitize_data_for_storage(update_data)
                await storage.update_data_async(output_path, sanitized_data)
            except Exception as e:
                LoggingUtil.warning("main", f"Storage 업데이트 실패: {e}")
        
        # 동기 함수로 Storage 업데이트 (transform 내부에서 호출)
        def sync_storage_update(update_data: dict):
            """동기 함수로 Storage 업데이트 (transform 내부에서 호출)"""
            try:
                sanitized_data = storage.sanitize_data_for_storage(update_data)
                storage.update_data(output_path, sanitized_data)
            except Exception as e:
                LoggingUtil.warning("main", f"Storage 업데이트 실패: {e}")
        
        result = transformer.transform(
            draft_options, 
            bounded_context, 
            job_id=result_dir_name,
            firebase_update_callback=sync_storage_update,  # Storage 업데이트 콜백 (Firebase/AceBase 공통)
            transformation_session_id=transformation_session_id  # 세션 ID 전달
        )

        # error가 None이거나 빈 문자열이면 제외
        # transformedOptions 또는 transformed_options 둘 다 확인 (호환성)
        transformed_options = result.get('transformedOptions') or result.get('transformed_options') or draft_options
        transformation_log = result.get('transformationLog') or result.get('transformation_log') or ''
        is_completed = result.get('isCompleted') if 'isCompleted' in result else result.get('is_completed', True)
        
        output = {
            'transformedOptions': transformed_options,
            'transformationLog': transformation_log,
            'progress': 100
        }
        
        # error가 실제로 있을 때만 추가
        error = result.get('error')
        if error:
            output['error'] = error

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': is_completed}
        )

        req_path = f'requestedJobs/standard_transformer/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
            req_path
        )

        LoggingUtil.info("main", f"🎉 표준 변환 완료: {job_id}")
        LoggingUtil.info("main", "────────────────────────────────────────────────────────────────")
        
    except Exception as e:
        LoggingUtil.exception("main", f"표준 변환 오류: {job_id}", e)
        
        # 에러 상태 저장
        try:
            job_path = f'jobs/standard_transformer/{job_id}'
            output_path = f'{job_path}/state/outputs'
            error_output = {
                'transformedOptions': inputs_data.get('draftOptions', []),  # 원본 반환
                'transformationLog': f'변환 실패: {str(e)}',
                'isCompleted': False,
                'progress': 0,
                'error': str(e)
            }
            sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(error_output)
            await asyncio.to_thread(
                StorageSystemFactory.instance().set_data,
                output_path,
                sanitized_output
            )
        except Exception as save_error:
            LoggingUtil.exception("main", f"에러 상태 저장 실패: {job_id}", save_error)
    finally:
        # 프로세스 종료 시 사용자별 임시 문서 정리
        if transformer:
            try:
                if not transformation_session_id:
                    # transformation_session_id가 없으면 각 job이 독립적이므로 즉시 정리
                    transformer.cleanup_user_standards()
                else:
                    # transformation_session_id가 있으면 같은 세션의 다른 BC가 남아있는지 확인
                    # requestedJobs/standard_transformer에서 같은 세션의 다른 job 확인
                    requested_jobs = await asyncio.to_thread(
                                StorageSystemFactory.instance().get_children_data,
                        'requestedJobs/standard_transformer'
                    )
                    
                    # 같은 세션의 다른 job이 있는지 확인
                    has_other_session_jobs = False
                    if requested_jobs:
                        for other_job_id, other_job_data in requested_jobs.items():
                            if other_job_id == job_id:
                                continue  # 현재 job은 제외
                            
                            # 다른 job의 transformationSessionId 확인
                            other_job_path = f'jobs/standard_transformer/{other_job_id}'
                            other_job_data_full = await asyncio.to_thread(
                                StorageSystemFactory.instance().get_data,
                                other_job_path
                            )
                            
                            if other_job_data_full:
                                other_state = other_job_data_full.get('state', {})
                                other_inputs = other_state.get('inputs', {})
                                other_session_id = other_inputs.get('transformationSessionId', None)
                                
                                if other_session_id == transformation_session_id:
                                    has_other_session_jobs = True
                                    break
                    
                    # 같은 세션의 다른 job이 없으면 cleanup 수행
                    if not has_other_session_jobs:
                        LoggingUtil.info("main", f"🧹 세션({transformation_session_id})의 모든 BC 처리 완료, 표준 문서 정리 시작")
                        transformer.cleanup_user_standards()
                    else:
                        LoggingUtil.debug("main", f"⏳ 세션({transformation_session_id})의 다른 BC가 아직 처리 중, cleanup 대기")
            except Exception as cleanup_error:
                LoggingUtil.warning("main", f"사용자 표준 문서 정리 중 오류: {cleanup_error}")
        complete_job_func()


async def process_traceability_job(job_id: str, complete_job_func: callable):
    """Traceability Addition Job 처리"""
    try:
        LoggingUtil.info("main", f"🚀 Traceability 추가 시작: {job_id}")

        job_path = f'jobs/traceability_generator/{job_id}'
        job_data = await asyncio.to_thread(
            StorageSystemFactory.instance().get_data,
            job_path
        )

        if not job_data:
            LoggingUtil.error("main", f"Job 데이터 없음: {job_id}")
            return

        state = job_data.get('state', {})
        inputs_data = state.get('inputs', {})

        trace_map = inputs_data.get('traceMap', {})
        
        # traceMap 복원 (Firebase가 배열로 변환한 경우 처리)
        if isinstance(trace_map, list):
            LoggingUtil.warning("main", f"⚠️ Traceability: traceMap이 배열 형태입니다! 복원 중... (배열 길이: {len(trace_map)})")
            # 원본 배열에서 키 샘플 확인 (복원 전) - 전체 확인
            original_keys = []
            for item in trace_map:  # 전체 확인
                if isinstance(item, dict) and 'key' in item:
                    try:
                        key = int(item['key'])
                        original_keys.append(key)
                    except (ValueError, TypeError):
                        pass
            if original_keys:
                original_odd = sorted([k for k in original_keys if k % 2 == 1])[:20]
                original_even = sorted([k for k in original_keys if k % 2 == 0])[:20]
                odd_count = len([k for k in original_keys if k % 2 == 1])
                even_count = len([k for k in original_keys if k % 2 == 0])
                LoggingUtil.info("main", f"📋 원본 배열 키 분석 - 총 키 수: {len(original_keys)}, "
                    f"홀수 키 수: {odd_count}, 짝수 키 수: {even_count}, "
                    f"홀수 샘플: {original_odd[:10]}, 짝수 샘플: {original_even[:10]}")
            else:
                LoggingUtil.warning("main", f"⚠️ 원본 배열에서 키를 찾을 수 없습니다! 배열 구조 확인 필요")
                if trace_map and len(trace_map) > 0:
                    # 배열 구조 상세 분석
                    first_item = trace_map[0]
                    LoggingUtil.info("main", f"🔍 배열 첫 번째 항목 타입: {type(first_item)}, "
                        f"내용: {str(first_item)[:200] if first_item else 'None'}")
                    if isinstance(first_item, dict):
                        LoggingUtil.info("main", f"🔍 첫 번째 항목의 키들: {list(first_item.keys()) if first_item else []}")
                    # 여러 항목 샘플 확인
                    sample_items = []
                    for i, item in enumerate(trace_map[:5]):
                        if isinstance(item, dict):
                            sample_items.append(f"항목{i}: keys={list(item.keys())}")
                        else:
                            sample_items.append(f"항목{i}: type={type(item).__name__}")
                    if sample_items:
                        LoggingUtil.info("main", f"🔍 배열 샘플 (처음 5개): {'; '.join(sample_items)}")
            
            temp_generator = TraceabilityGenerator()
            trace_map = temp_generator._restore_trace_map(trace_map)
            if isinstance(trace_map, dict):
                # 복원된 키 샘플 확인 (홀수/짝수 모두 확인)
                # 키를 정수로 변환하여 정렬 (문자열과 정수 혼합 정렬 방지)
                numeric_keys = []
                for k in trace_map.keys():
                    try:
                        if isinstance(k, int):
                            numeric_keys.append(k)
                        elif isinstance(k, str) and k.isdigit():
                            numeric_keys.append(int(k))
                    except (ValueError, TypeError):
                        pass
                sample_keys = sorted(numeric_keys)[:20]
                odd_keys = [k for k in sample_keys if k % 2 == 1]
                even_keys = [k for k in sample_keys if k % 2 == 0]
                LoggingUtil.info("main", f"✅ Traceability: traceMap 복원 완료, keys={len(trace_map)}, "
                    f"샘플 키 (짝수): {even_keys[:10]}, 샘플 키 (홀수): {odd_keys[:10]}")
            else:
                LoggingUtil.warning("main", f"⚠️ Traceability: traceMap 복원 실패, 타입={type(trace_map)}")
        elif isinstance(trace_map, dict):
            # dict인 경우도 키 샘플 확인
            # 키를 정수로 변환하여 정렬 (문자열과 정수 혼합 정렬 방지)
            numeric_keys = []
            for k in trace_map.keys():
                try:
                    if isinstance(k, int):
                        numeric_keys.append(k)
                    elif isinstance(k, str) and k.isdigit():
                        numeric_keys.append(int(k))
                except (ValueError, TypeError):
                    pass
            sample_keys = sorted(numeric_keys)[:20]
            odd_keys = [k for k in sample_keys if k % 2 == 1]
            even_keys = [k for k in sample_keys if k % 2 == 0]
            LoggingUtil.info("main", f"✅ Traceability: traceMap 구조 확인 (dict), keys={len(trace_map)}, "
                f"샘플 키 (짝수): {even_keys[:10]}, 샘플 키 (홀수): {odd_keys[:10]}")

        input_data = {
            'generatedDraftOptions': inputs_data.get('generatedDraftOptions', []),
            'boundedContextName': inputs_data.get('boundedContextName', ''),
            'description': inputs_data.get('description', ''),
            'functionalRequirements': inputs_data.get('functionalRequirements', ''),
            'traceMap': trace_map,
        }

        generator = TraceabilityGenerator()
        result = generator.generate(input_data)

        output = {
            'inference': result.get('inference', ''),
            'draftTraceMap': result.get('draftTraceMap', {}),
            'progress': 100,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Traceability mapping completed'}]
        }

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )

        req_path = f'requestedJobs/traceability_generator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
            'progress': 100,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'DDL extraction completed'}]
        }

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        output_path = f'{job_path}/state/outputs'
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )

        req_path = f'requestedJobs/ddl_extractor/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
            StorageSystemFactory.instance().get_data,
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
        storage = StorageSystemFactory.instance()

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
            await storage.update_data_async(
                output_path,
                storage.sanitize_data_for_storage(update_payload)
            )
            await asyncio.sleep(1)

        output = {
            'type': result.get('type', 'ANALYSIS_RESULT'),
            'content': result.get('content', {}),
            'progress': 100,
            'currentGeneratedLength': final_length,
            'logs': [{'timestamp': datetime.now().isoformat(), 'level': 'info', 'message': 'Requirements validation completed'}]
        }

        # ★ isCompleted를 마지막에 별도로 저장하여 이벤트 순서 보장
        # 1) isCompleted 제외한 데이터 먼저 저장
        sanitized_output = StorageSystemFactory.instance().sanitize_data_for_storage(output)
        await asyncio.to_thread(
            StorageSystemFactory.instance().set_data,
            output_path,
            sanitized_output
        )
        
        # 2) 짧은 대기 후 isCompleted 저장 (이벤트 순서 보장)
        await asyncio.sleep(0.1)
        await asyncio.to_thread(
            StorageSystemFactory.instance().update_data,
            output_path,
            {'isCompleted': True}
        )

        req_path = f'requestedJobs/requirements_validator/{job_id}'
        await asyncio.to_thread(
            StorageSystemFactory.instance().delete_data,
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
            StorageSystemFactory.instance().set_data(output_path, error_output)
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
        elif job_id.startswith("std-trans-"):
            await process_standard_transformation_job(job_id, complete_job_func)
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