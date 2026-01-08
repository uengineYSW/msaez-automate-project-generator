from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable


class StorageSystem(ABC):
    """Storage 시스템 추상 클래스 (Strategy Pattern)"""
    
    _instance: Optional['StorageSystem'] = None
    _initialized: bool = False
    
    @classmethod
    @abstractmethod
    def initialize(cls, **kwargs) -> 'StorageSystem':
        """Storage 시스템 초기화"""
        pass
    
    @classmethod
    @abstractmethod
    def instance(cls) -> 'StorageSystem':
        """싱글톤 인스턴스 반환"""
        pass
    
    # =============================================================================
    # 데이터 설정 메서드들
    # =============================================================================
    
    @abstractmethod
    def set_data(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로에 딕셔너리 데이터를 업로드"""
        pass
    
    @abstractmethod
    async def set_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로에 딕셔너리 데이터를 비동기로 업로드"""
        pass
    
    @abstractmethod
    def set_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """데이터를 업로드하되 결과를 기다리지 않음 (Fire and Forget)"""
        pass
    
    # =============================================================================
    # 데이터 업데이트 메서드들
    # =============================================================================
    
    @abstractmethod
    def update_data(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로의 데이터를 부분 업데이트"""
        pass
    
    @abstractmethod
    async def update_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로의 데이터를 비동기로 부분 업데이트"""
        pass
    
    @abstractmethod
    def update_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """데이터를 업데이트하되 결과를 기다리지 않음 (Fire and Forget)"""
        pass
    
    # =============================================================================
    # 조건부 업데이트 메서드들
    # =============================================================================
    
    @abstractmethod
    def conditional_update_data(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """두 데이터를 비교하여 변경된 부분만 효율적으로 업데이트"""
        pass
    
    @abstractmethod
    async def conditional_update_data_async(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """두 데이터를 비교하여 변경된 부분만 비동기로 효율적으로 업데이트"""
        pass
    
    @abstractmethod
    def conditional_update_data_fire_and_forget(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> None:
        """데이터를 조건부로 업데이트하되 결과를 기다리지 않음 (Fire and Forget)"""
        pass
    
    # =============================================================================
    # 데이터 조회 메서드들
    # =============================================================================
    
    @abstractmethod
    def get_data(self, path: str) -> Optional[Dict[str, Any]]:
        """특정 경로에서 데이터를 딕셔너리 형태로 조회"""
        pass
    
    @abstractmethod
    def get_children_data(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """특정 경로의 모든 자식 노드 데이터를 조회"""
        pass
    
    @abstractmethod
    async def get_children_data_async(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """특정 경로의 모든 자식 노드 데이터를 비동기로 조회"""
        pass
    
    # =============================================================================
    # 데이터 삭제 메서드들
    # =============================================================================
    
    @abstractmethod
    def delete_data(self, path: str) -> bool:
        """특정 경로의 데이터 삭제"""
        pass
    
    @abstractmethod
    async def delete_data_async(self, path: str) -> bool:
        """특정 경로의 데이터를 비동기로 삭제"""
        pass
    
    @abstractmethod
    def delete_data_fire_and_forget(self, path: str) -> None:
        """데이터를 삭제하되 결과를 기다리지 않음 (Fire and Forget)"""
        pass
    
    # =============================================================================
    # 데이터 감시 메서드들
    # =============================================================================
    
    @abstractmethod
    def watch_data(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """특정 경로의 데이터 변화를 감시하고 콜백 함수 호출"""
        pass
    
    @abstractmethod
    async def watch_data_async(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """특정 경로의 데이터 변화를 비동기로 감시하고 콜백 함수 호출"""
        pass
    
    @abstractmethod
    def unwatch_data(self, path: str) -> bool:
        """특정 경로의 데이터 감시 중단"""
        pass
    
    @abstractmethod
    async def unwatch_data_async(self, path: str) -> bool:
        """특정 경로의 데이터 감시를 비동기로 중단"""
        pass
    
    @abstractmethod
    def unwatch_all(self) -> bool:
        """모든 경로의 데이터 감시 중단"""
        pass
    
    @abstractmethod
    async def unwatch_all_async(self) -> bool:
        """모든 경로의 데이터 감시를 비동기로 중단"""
        pass
    
    @abstractmethod
    def get_active_watchers(self) -> list[str]:
        """현재 감시 중인 경로들의 목록을 반환"""
        pass
    
    # =============================================================================
    # 트랜잭션 메서드들
    # =============================================================================
    
    @abstractmethod
    def transaction(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 실행"""
        pass
    
    @abstractmethod
    async def transaction_async(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 비동기 실행"""
        pass
    
    # =============================================================================
    # 데이터 정제 메서드들
    # =============================================================================
    
    @abstractmethod
    def sanitize_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Storage 업로드를 위해 데이터 정제"""
        pass
    
    @abstractmethod
    def restore_data_from_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Storage에서 가져온 데이터를 원본 형태로 복원"""
        pass
    
    # =============================================================================
    # 참조 객체 반환 (Firebase 전용이지만 호환성을 위해 포함)
    # =============================================================================
    
    @property
    @abstractmethod
    def database(self):
        """데이터베이스 참조 객체 (호환성용)"""
        pass

