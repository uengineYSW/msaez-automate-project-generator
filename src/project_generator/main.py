import asyncio
import concurrent.futures
import threading
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from project_generator.utils import JobUtil, DecentralizedJobManager
from project_generator.systems.firebase_system import FirebaseSystem
from project_generator.config import Config
from project_generator.run_healcheck_server import run_healcheck_server
from project_generator.simple_autoscaler import start_autoscaler
from project_generator.utils.logging_util import LoggingUtil

# UserStory Workflow import
from project_generator.workflows.user_story.user_story_generator import UserStoryWorkflow

# 전역 job_manager 인스턴스
_current_job_manager: DecentralizedJobManager = None

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
            
            # 감시할 namespace 목록 (UserStory Generator만)
            monitored_namespaces = ['user_story_generator']
            
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
        complete_job_func()
        
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

async def process_job_async(job_id: str, complete_job_func: callable):
    """비동기 UserStory Job 처리 함수"""
    
    try:
        LoggingUtil.debug("main", f"Job 시작: {job_id}")
        if not JobUtil.is_valid_job_id(job_id):
            LoggingUtil.warning("main", f"Job 처리 오류: {job_id}, 유효하지 않음")
            return
        
        # UserStory Job만 처리
        if job_id.startswith("usgen-"):
            await process_user_story_job(job_id, complete_job_func)
        else:
            LoggingUtil.warning("main", f"지원하지 않는 Job 타입: {job_id}")
            
    except asyncio.CancelledError:
        LoggingUtil.debug("main", f"Job {job_id} 취소됨")
        return
        
    except Exception as e:
        LoggingUtil.exception("main", f"Job 처리 오류: {job_id}", e)

if __name__ == "__main__":
    asyncio.run(main())