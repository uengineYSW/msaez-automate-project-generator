import asyncio
import time
import signal
import os
import sys
from typing import Optional, List, Tuple

from ..systems import FirebaseSystem
from ..config import Config
from .logging_util import LoggingUtil

class DecentralizedJobManager:
    def __init__(self, pod_id: str, job_processing_func: callable):
        self.pod_id = pod_id
        self.job_processing_func = job_processing_func
        self.current_job_id: Optional[str] = None  # 단일 Job 처리
        self.is_processing = False
        self.current_task: Optional[asyncio.Task] = None  # 현재 실행 중인 작업 태스크
        self.shutdown_requested = False  # Graceful shutdown 플래그
        self.shutdown_event = asyncio.Event()  # Graceful shutdown 완료 이벤트
        self.job_removal_requested = False  # 현재 작업 제거 요청 플래그
        self.job_cancellation_flags = {}  # 작업별 취소 플래그 {job_id: asyncio.Event}
    
    @staticmethod
    def _get_namespace_from_job_id(job_id: str) -> str:
        """Job ID prefix로 namespace 결정"""
        if job_id.startswith("usgen-"):
            return "user_story_generator"
        elif job_id.startswith("summ-"):
            return "summarizer"
        elif job_id.startswith("bcgen-"):
            return "bounded_context"
        elif job_id.startswith("cmrext-"):
            return "command_readmodel_extractor"
        elif job_id.startswith("smapgen-"):
            return "sitemap_generator"
        else:
            return "project_generator"
    
    def _get_job_path(self, job_id: str, subpath: str = "") -> str:
        """Job ID에 맞는 Firebase 경로 반환"""
        namespace = self._get_namespace_from_job_id(job_id)
        base = f"jobs/{namespace}/{job_id}"
        return f"{base}/{subpath}" if subpath else base
    
    def _get_requested_job_path(self, job_id: str) -> str:
        """RequestedJob Firebase 경로 반환"""
        namespace = self._get_namespace_from_job_id(job_id)
        return f"requestedJobs/{namespace}/{job_id}"
    
    def _get_job_state_path(self, job_id: str) -> str:
        """JobState Firebase 경로 반환"""
        namespace = self._get_namespace_from_job_id(job_id)
        return f"jobStates/{namespace}/{job_id}"
    
    def setup_signal_handlers(self):
        """Graceful shutdown을 위한 신호 핸들러 설정"""
        def signal_handler(signum, frame):
            LoggingUtil.info("decentralized_job_manager", f"종료 신호 수신 ({signum}). Graceful shutdown 시작...")
            self.shutdown_requested = True
        
        # SIGTERM (Kubernetes가 Pod 종료 시 보내는 신호)과 SIGINT 처리
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def start_job_monitoring(self, namespaces=None):
        """각 Pod가 독립적으로 작업 모니터링 - 순차 처리 (Graceful Shutdown 지원)
        
        Args:
            namespaces: 감시할 namespace 리스트. None이면 Config의 기본 namespace 사용
        """
        if namespaces is None:
            namespaces = [Config.get_namespace()]
        
        LoggingUtil.info("decentralized_job_manager", f"Job 모니터링 시작 (순차 처리 모드, Graceful Shutdown 지원)")
        LoggingUtil.info("decentralized_job_manager", f"감시 중인 namespaces: {namespaces}")
        
        while not self.shutdown_requested:
            try:
                LoggingUtil.debug("decentralized_job_manager", f"Job 모니터링 중...")

                # 여러 namespace의 Job을 모두 수집
                all_requested_jobs = {}
                for namespace in namespaces:
                    namespace_path = f"requestedJobs/{namespace}"
                    jobs = await FirebaseSystem.instance().get_children_data_async(namespace_path)
                    if jobs:
                        all_requested_jobs.update(jobs)
                
                requested_jobs = all_requested_jobs
                
                # 작업 삭제 요청 확인 및 처리
                await self.check_and_handle_removal_requests(requested_jobs)
                
                # 완료된 작업 확인 및 정리
                if self.current_task and self.current_task.done():
                    await self._handle_completed_task()
                
                # Graceful shutdown 요청이 있고 현재 작업이 없으면 종료
                if self.shutdown_requested and not self.is_processing:
                    LoggingUtil.info("decentralized_job_manager", "Graceful shutdown: 현재 처리 중인 작업이 없어 즉시 종료")
                    break
                
                # Graceful shutdown 요청이 없을 때만 새 작업 수락
                if not self.shutdown_requested and not self.is_processing:
                    # 현재 처리 중인 Job이 없을 때만 새 Job 검색
                    await self.find_and_process_next_job(requested_jobs)
                
                # 현재 처리 중인 Job의 heartbeat 전송
                if self.current_job_id:
                    await self.send_heartbeat()
                    LoggingUtil.debug("decentralized_job_manager", f"Job {self.current_job_id} heartbeat 전송 완료")
                
                # 대기 중인 작업들의 waitingJobCount 업데이트
                await self.update_waiting_job_counts(requested_jobs)
                
                # 실패한 작업 복구 (다른 Pod의 실패 작업)
                await self.recover_failed_jobs(requested_jobs)
                
                # 이벤트 루프 양보 - 다른 태스크들이 실행될 수 있도록 함
                await asyncio.sleep(0.1)
                
                await asyncio.sleep(15)  # 15초마다 체크
                
            except Exception as e:
                LoggingUtil.exception("decentralized_job_manager", f"작업 모니터링 오류", e)
                await asyncio.sleep(15)
        
        # Graceful shutdown 처리
        await self._handle_graceful_shutdown()

    async def _handle_graceful_shutdown(self):
        """Graceful shutdown 처리 - 현재 작업 완료를 기다림"""
        if self.is_processing and self.current_task:
            LoggingUtil.info("decentralized_job_manager", f"Graceful shutdown: 현재 작업 {self.current_job_id} 완료를 기다리는 중...")
            
            # 현재 작업이 완료될 때까지 기다림
            while self.is_processing and not self.current_task.done():
                await self.send_heartbeat()  # 작업이 살아있음을 알림
                await asyncio.sleep(30)  # 30초마다 확인
            
            # 작업이 완료되면 정리
            if self.current_task and self.current_task.done():
                await self._handle_completed_task()
            
            LoggingUtil.info("decentralized_job_manager", f"Graceful shutdown: 모든 작업 완료. 안전하게 종료합니다.")
        else:
            LoggingUtil.info("decentralized_job_manager", f"Graceful shutdown: 처리 중인 작업이 없어 즉시 종료합니다.")
        
        
        # Graceful shutdown 완료 이벤트 설정
        self.shutdown_event.set()
        
        # 프로세스 강제 종료를 위한 추가 로직
        LoggingUtil.info("decentralized_job_manager", "프로세스를 안전하게 종료합니다.")
        
        # 짧은 지연 후 프로세스 종료
        await asyncio.sleep(1)
        
        # 현재 이벤트 루프의 모든 태스크 취소
        try:
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                LoggingUtil.info("decentralized_job_manager", f"{len(tasks)}개의 실행 중인 태스크를 취소합니다.")
                for task in tasks:
                    task.cancel()
                
                # 취소된 태스크들이 완료될 때까지 잠시 대기
                await asyncio.sleep(0.5)
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", "태스크 취소 중 오류", e)
        
        # 프로세스 종료
        LoggingUtil.info("decentralized_job_manager", "프로세스를 종료합니다.")
        os._exit(0)

    def is_job_cancelled(self, job_id: str) -> bool:
        """특정 작업이 취소되었는지 확인"""
        if job_id in self.job_cancellation_flags:
            return self.job_cancellation_flags[job_id].is_set()
        return False

    def get_job_cancellation_event(self, job_id: str) -> Optional[asyncio.Event]:
        """특정 작업의 취소 이벤트 반환"""
        return self.job_cancellation_flags.get(job_id)

    async def _handle_completed_task(self):
        """완료된 작업 태스크 처리"""
        if self.current_task:
            try:
                # 태스크에서 예외가 발생했는지 확인
                await self.current_task
            except Exception as e:
                LoggingUtil.exception("decentralized_job_manager", f"Job {self.current_job_id} 처리 중 오류", e)
            finally:
                self.current_task = None


    async def find_and_process_next_job(self, requested_jobs: dict):
        """사용 가능한 다음 Job 찾기 및 처리 시작 (FIFO 순서)"""
        
        if not requested_jobs:
            LoggingUtil.debug("decentralized_job_manager", f"대기 중인 Job 없음 (requested_jobs: {requested_jobs})")
            return
        
        LoggingUtil.info("decentralized_job_manager", f"🔍 대기 중인 Job {len(requested_jobs)}개 발견: {list(requested_jobs.keys())}")
        
        # createdAt 기준으로 정렬 (FIFO)
        sorted_jobs = self._sort_jobs_by_created_at(requested_jobs)
        
        # 할당되지 않은 Job 찾기 (시간순으로)
        for job_id, job_data in sorted_jobs:
            # assignedPodId가 없고, status가 'failed'가 아닌 작업만 고려
            assigned_pod = job_data.get('assignedPodId')
            status = job_data.get('status')
            LoggingUtil.debug("decentralized_job_manager", f"Job {job_id} 확인 - assignedPodId: {assigned_pod}, status: {status}")
            
            if assigned_pod is None and status != 'failed':
                LoggingUtil.info("decentralized_job_manager", f"🎯 Job {job_id} claim 시도...")
                success = await self.atomic_claim_job(job_id)
                if success:
                    # 성공적으로 클레임한 경우 해당 Job 처리 시작
                    LoggingUtil.info("decentralized_job_manager", f"✅ Job {job_id} claim 성공! 처리 시작...")
                    await self.start_job_processing(job_id)
                    LoggingUtil.info("decentralized_job_manager", f"🚀 Job {job_id} 처리 시작 완료")
                    break  # 하나만 처리하고 종료
                else:
                    LoggingUtil.warning("decentralized_job_manager", f"❌ Job {job_id} claim 실패")
            else:
                LoggingUtil.debug("decentralized_job_manager", f"⏭️  Job {job_id} 스킵 (이미 할당됨 또는 실패)")

    async def atomic_claim_job(self, job_id: str) -> bool:
        """원자적 작업 클레임"""
        
        def update_function(current_data):
            if current_data is None:
                return current_data

            restored_data = FirebaseSystem.instance().restore_data_from_firebase(current_data)
            
            if restored_data.get('assignedPodId') is not None:
                return current_data

            restored_data['assignedPodId'] = self.pod_id
            restored_data['claimedAt'] = time.time()
            restored_data['status'] = 'processing'
            restored_data['lastHeartbeat'] = time.time()
            return FirebaseSystem.instance().sanitize_data_for_firebase(restored_data)
        
        try:
            ref = FirebaseSystem.instance().database.reference(self._get_requested_job_path(job_id))
            transaction_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ref.transaction(update_function)
            )
            
            if transaction_result is None:
                LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 클레임 시도했으나, 해당 경로에 데이터가 없음.")
                return False

            final_data = FirebaseSystem.instance().restore_data_from_firebase(transaction_result)
            if final_data.get('assignedPodId') == self.pod_id:
                LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 클레임 성공")
                return True
            else:
                LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id}은 다른 Pod에 의해 선점되었거나 이미 처리 중입니다.")
                return False
            
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"작업 클레임 실패", e)
        
        return False
    
    async def start_job_processing(self, job_id: str):
        """Job 처리 시작"""
        self.current_job_id = job_id
        self.is_processing = True
        
        # 작업별 취소 플래그 생성
        self.job_cancellation_flags[job_id] = asyncio.Event()
        
        LoggingUtil.debug("decentralized_job_manager", f"Job {job_id} 처리 시작")
        
        # 실제 작업 수행 - execute_job_logic은 태스크만 생성하고 즉시 리턴하므로 await 불필요
        await self.execute_job_logic(job_id)
        
        LoggingUtil.debug("decentralized_job_manager", f"Job {job_id} 백그라운드 태스크 생성 완료")
    
    async def execute_job_logic(self, job_id: str):
        """실제 Job 로직 실행 - 비동기로 백그라운드에서 실행"""
        LoggingUtil.debug("decentralized_job_manager", f"Job {job_id} 로직 실행 중...")
        
        # 작업을 백그라운드 태스크로 실행하여 heartbeat가 블록되지 않도록 함
        self.current_task = asyncio.create_task(
            self.job_processing_func(job_id, self.complete_job)
        )

    def complete_job(self):
        """Job 완료 처리"""
        LoggingUtil.debug("decentralized_job_manager", f"Job {self.current_job_id} 처리 완료")

        # 취소 플래그 정리
        if self.current_job_id in self.job_cancellation_flags:
            del self.job_cancellation_flags[self.current_job_id]

        self.current_job_id = None
        self.is_processing = False
        self.job_removal_requested = False  # 제거 요청 플래그 초기화
        # current_task는 _handle_completed_task에서 정리됨
    

    async def send_heartbeat(self):
        """현재 처리 중인 Job의 heartbeat 전송"""
        if not self.current_job_id:
            return
        
        try:
            heartbeat_data = {'lastHeartbeat': time.time()}
            
            # Graceful shutdown 중이면 상태 정보 추가
            if self.shutdown_requested:
                heartbeat_data['shutdownRequested'] = True
                heartbeat_data['acceptingNewJobs'] = False
            
            await FirebaseSystem.instance().update_data_async(
                self._get_requested_job_path(self.current_job_id),
                heartbeat_data
            )
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"Heartbeat 실패", e)
    

    async def update_waiting_job_counts(self, requested_jobs: dict):
        """대기 중인 작업들의 waitingJobCount 업데이트"""
        try:
            if not requested_jobs:
                return
            
            # createdAt 기준으로 정렬
            sorted_jobs = self._sort_jobs_by_created_at(requested_jobs)
            
            # 대기 중인 작업들만 필터링 (assignedPodId가 없고, status가 'failed'가 아닌 것들)
            waiting_jobs = []
            for job_id, job_data in sorted_jobs:
                if job_data.get('assignedPodId') is None and job_data.get('status') != 'failed':
                    waiting_jobs.append((job_id, job_data))
            
            # 각 대기 중인 작업의 waitingJobCount 계산 및 업데이트
            for index, (job_id, job_data) in enumerate(waiting_jobs):
                waiting_count = index + 1  # 앞에 있는 대기 작업의 개수(대기중인 상태이기 때문에 기본적으로 1개 추가)
                current_waiting_count = job_data.get('waitingJobCount')
                
                # waitingJobCount가 없거나 기존 값과 다를 경우에만 업데이트
                if current_waiting_count != waiting_count:
                    await FirebaseSystem.instance().update_data_async(
                        self._get_requested_job_path(job_id),
                        {'waitingJobCount': waiting_count}
                    )
                    LoggingUtil.debug("decentralized_job_manager", f"Job {job_id} waitingJobCount 업데이트: {waiting_count}")
                    
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"waitingJobCount 업데이트 오류", e)
    

    async def recover_failed_jobs(self, requested_jobs: dict):
        """다른 Pod의 실패한 작업 복구 및 영구 실패 처리"""
        try:     
            if not requested_jobs:
                return
            
            current_time = time.time()
            for job_id, job_data in requested_jobs.items():
                assigned_pod = job_data.get('assignedPodId')
                last_heartbeat = job_data.get('lastHeartbeat', 0)
                
                # 다른 Pod가 할당했지만 5분간 heartbeat 없으면 실패로 간주
                if (assigned_pod and 
                    current_time - last_heartbeat > 300 and
                    job_data.get('status') == 'processing' and 
                    (not job_data.get('shutdownRequested'))):
                    
                    recovery_count = job_data.get('recoveryCount', 0)
                    LoggingUtil.warning("decentralized_job_manager", f"실패한 작업 감지: {job_id} (Pod: {assigned_pod}), 복구 시도 횟수: {recovery_count}")

                    if recovery_count >= 1:
                        # 복구 횟수 초과 시 영구 실패 처리
                        await self.mark_job_as_failed(job_id)
                    else:
                        # 작업 복구 시도
                        await self.reset_failed_job(job_id, recovery_count)
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"실패 작업 복구 오류", e)

    async def mark_job_as_failed(self, job_id: str):
        """영구적으로 실패한 작업을 'failed' 상태로 표시"""
        try:
            await FirebaseSystem.instance().update_data_async(
                self._get_requested_job_path(job_id),
                {
                    'status': 'failed',
                    'assignedPodId': None,
                    'lastHeartbeat': None,
                    'failedAt': time.time(),
                }
            )
            LoggingUtil.error("decentralized_job_manager", f"작업 {job_id}가 영구 실패 처리되었습니다.")
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"작업 {job_id} 실패 처리 중 오류", e)
    
    async def reset_failed_job(self, job_id: str, current_recovery_count: int):
        """실패한 작업 초기화 및 복구 횟수 증가"""
        try:
            await FirebaseSystem.instance().update_data_async(
                self._get_requested_job_path(job_id),
                {
                    'assignedPodId': None,
                    'status': 'pending',
                    'lastHeartbeat': None,
                    'recoveryCount': current_recovery_count + 1,
                }
            )
            LoggingUtil.info("decentralized_job_manager", f"실패 작업 {job_id} 초기화 완료 (복구 시도: {current_recovery_count + 1})")
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"실패 작업 초기화 오류", e)
    
    
    def _sort_jobs_by_created_at(self, jobs_dict: dict) -> List[Tuple[str, dict]]:
        """createdAt 기준으로 작업들을 시간순(FIFO)으로 정렬"""
        jobs_list = [(job_id, job_data) for job_id, job_data in jobs_dict.items()]
        
        # createdAt 기준으로 정렬 (오래된 것부터)
        jobs_list.sort(key=lambda x: x[1].get('createdAt', 0))
        
        return jobs_list


    async def check_and_handle_removal_requests(self, requested_jobs: dict):
        """작업 삭제 요청 확인 및 처리"""
        try:
            # jobStates에서 삭제 요청된 작업들 조회
            job_states = await FirebaseSystem.instance().get_children_data_async(Config.get_job_state_root_path())
            
            if not job_states:
                return
            
            # isRemoveRequested가 true인 작업들 찾기
            removal_requests = {}
            for job_id, state_data in job_states.items():
                if state_data and state_data.get('isRemoveRequested') == True:
                    removal_requests[job_id] = state_data
            
            if not removal_requests:
                return
            
            LoggingUtil.debug("decentralized_job_manager", f"삭제 요청된 작업 {len(removal_requests)}개 발견")
            
            # 각 삭제 요청 처리
            for job_id, state_data in removal_requests.items():
                await self.handle_job_removal_request(job_id, requested_jobs)
                
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"삭제 요청 처리 오류", e)

    async def handle_job_removal_request(self, job_id: str, requested_jobs: dict):
        """개별 작업 삭제 요청 처리"""
        try:
            # 현재 진행 중인 작업인지 확인
            if self.current_job_id == job_id:
                await self.handle_current_job_removal(job_id)
                return
            
            if requested_jobs and job_id in requested_jobs:
                # 다른 Pod가 진행 중인 작업인지 확인
                job_data = requested_jobs[job_id]
                assigned_pod = job_data.get('assignedPodId')
                
                if assigned_pod == self.pod_id:
                    # 자신이 할당받았지만 아직 처리하지 않은 작업
                    await self.handle_current_job_removal(job_id)
                elif assigned_pod:
                    # 다른 Pod가 처리 중인 작업 - 해당 Pod가 처리해야 함
                    LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id}은 Pod {assigned_pod}가 처리 중이므로 건너뜀")
                    return
                else:
                    # 할당되지 않은 요청 작업 - 삭제 처리
                    await self.handle_unassigned_job_removal(job_id)
                return
            
            # jobs에서 해당 작업 확인
            job = FirebaseSystem.instance().get_data(self._get_job_path(job_id))
            
            if job:
                # 완료된 작업 삭제 처리
                await self.handle_completed_job_removal(job_id)
            else:
                # orphan jobState 삭제 처리
                await self.handle_orphan_job_state_removal(job_id)
                
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"작업 {job_id} 삭제 요청 처리 오류", e)

    async def handle_current_job_removal(self, job_id: str):
        """현재 진행 중인 작업 삭제 처리"""
        try:
            LoggingUtil.debug("decentralized_job_manager", f"현재 진행 중인 작업 {job_id} 삭제 요청 처리 시작")
            
            # 작업 중단 플래그 설정
            self.job_removal_requested = True
            
            # 취소 플래그 설정 (process_job_async에서 확인할 수 있도록)
            if job_id in self.job_cancellation_flags:
                self.job_cancellation_flags[job_id].set()
                LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 취소 플래그 설정")
            
            # 현재 실행 중인 태스크가 있으면 취소
            if self.current_task and not self.current_task.done():
                LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 태스크 취소 중...")
                self.current_task.cancel()
                
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 태스크가 정상적으로 취소됨")
                except Exception as e:
                    LoggingUtil.exception("decentralized_job_manager", f"작업 {job_id} 태스크 취소 중 오류", e)
            
            # 순차적으로 데이터 삭제: requestedJobs → jobs → jobStates
            await self.delete_job_data_sequentially(job_id, include_requested=True)
            
            # 상태 초기화 및 취소 플래그 정리
            if job_id in self.job_cancellation_flags:
                del self.job_cancellation_flags[job_id]
            
            self.current_job_id = None
            self.is_processing = False
            self.current_task = None
            self.job_removal_requested = False
            
            LoggingUtil.debug("decentralized_job_manager", f"작업 {job_id} 삭제 완료")
            
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"현재 작업 {job_id} 삭제 처리 오류", e)

    async def handle_unassigned_job_removal(self, job_id: str):
        """할당되지 않은 요청 작업 삭제 처리"""
        try:
            LoggingUtil.debug("decentralized_job_manager", f"할당되지 않은 요청 작업 {job_id} 삭제 처리")
            
            # requestedJobs → jobs → jobStates 순차 삭제
            await self.delete_job_data_sequentially(job_id, include_requested=True)
            
            LoggingUtil.debug("decentralized_job_manager", f"할당되지 않은 작업 {job_id} 삭제 완료")
            
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"할당되지 않은 작업 {job_id} 삭제 처리 오류", e)

    async def handle_completed_job_removal(self, job_id: str):
        """완료된 작업 삭제 처리"""
        try:
            LoggingUtil.debug("decentralized_job_manager", f"완료된 작업 {job_id} 삭제 처리")
            
            # jobs → jobStates 순차 삭제
            await self.delete_job_data_sequentially(job_id, include_requested=False)
            
            LoggingUtil.debug("decentralized_job_manager", f"완료된 작업 {job_id} 삭제 완료")
            
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"완료된 작업 {job_id} 삭제 처리 오류", e)

    async def handle_orphan_job_state_removal(self, job_id: str):
        """orphan jobState 삭제 처리"""
        try:
            LoggingUtil.debug("decentralized_job_manager", f"orphan jobState {job_id} 삭제 처리")
            
            # jobStates만 삭제
            job_state_path = Config.get_job_state_path(job_id)
            success = await FirebaseSystem.instance().delete_data_async(job_state_path)
            
            if success:
                LoggingUtil.debug("decentralized_job_manager", f"orphan jobState {job_id} 삭제 완료")
            else:
                LoggingUtil.warning("decentralized_job_manager", f"orphan jobState {job_id} 삭제 실패")
                
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"orphan jobState {job_id} 삭제 처리 오류", e)

    async def delete_job_data_sequentially(self, job_id: str, include_requested: bool = True):
        """작업 데이터 순차적 삭제 (requestedJobs → jobs → jobStates)"""
        try:
            # 1. requestedJobs 삭제 (필요한 경우)
            if include_requested:
                requested_job_path = self._get_requested_job_path(job_id)
                success = await FirebaseSystem.instance().delete_data_async(requested_job_path)
                if success:
                    LoggingUtil.debug("decentralized_job_manager", f"requestedJobs에서 {job_id} 삭제 완료")
                else:
                    LoggingUtil.warning("decentralized_job_manager", f"requestedJobs에서 {job_id} 삭제 실패")
                
                # 삭제 간격 (Firebase 부하 방지)
                await asyncio.sleep(0.5)
            
            # 2. jobs 삭제
            job_path = self._get_job_path(job_id)
            success = await FirebaseSystem.instance().delete_data_async(job_path)
            if success:
                LoggingUtil.debug("decentralized_job_manager", f"jobs에서 {job_id} 삭제 완료")
            else:
                LoggingUtil.warning("decentralized_job_manager", f"jobs에서 {job_id} 삭제 실패")
            
            # 삭제 간격
            await asyncio.sleep(0.5)
            
            # 3. jobStates 삭제
            job_state_path = Config.get_job_state_path(job_id)
            success = await FirebaseSystem.instance().delete_data_async(job_state_path)
            if success:
                LoggingUtil.debug("decentralized_job_manager", f"jobStates에서 {job_id} 삭제 완료")
            else:
                LoggingUtil.warning("decentralized_job_manager", f"jobStates에서 {job_id} 삭제 실패")
                
        except Exception as e:
            LoggingUtil.exception("decentralized_job_manager", f"작업 {job_id} 순차 삭제 오류", e)