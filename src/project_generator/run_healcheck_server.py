from flask import Flask, jsonify, request
import logging
import os
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

class HealthCheckFilter(logging.Filter):
    """헬스체크 요청을 로그에서 제외하는 필터"""
    def filter(self, record):
        # 로그 메시지에 '/ok'가 포함되어 있으면 필터링 (로그 출력 안함)
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            if 'GET /ok HTTP' in message:
                return False
        return True

@app.after_request
def after_request(response):
    """모든 응답에 CORS 헤더 추가"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/ok', methods=['GET', 'OPTIONS'])
def health_check():
    """헬스체크 엔드포인트"""
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        return '', 200
    
    return jsonify({
        'status': 'ok',
        'message': 'EventStorming Generator 서버가 정상 작동 중입니다.'
    })

@app.route('/api/standard-documents/upload', methods=['POST', 'OPTIONS'])
def upload_standard_documents():
    """표준 문서 업로드 API (AceBase 로컬 환경용)"""
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        return '', 200
    
    try:
        # user_id 확인
        user_id = request.form.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        # 파일 확인
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Config import (순환 참조 방지를 위해 함수 내부에서 import)
        from project_generator.config import Config
        
        # 사용자별 디렉토리 경로
        user_standards_dir = Config.COMPANY_STANDARDS_PATH / user_id
        user_standards_dir.mkdir(parents=True, exist_ok=True)
        
        # 지원하는 파일 형식
        allowed_extensions = {'.xlsx', '.xls', '.pptx', '.ppt'}
        
        uploaded_files = []
        errors = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # 파일명 보안 처리
            filename = secure_filename(file.filename)
            file_ext = Path(filename).suffix.lower()
            
            # 파일 형식 검증
            if file_ext not in allowed_extensions:
                errors.append(f'{filename}: 지원하지 않는 파일 형식입니다. (.xlsx, .xls, .pptx, .ppt만 가능)')
                continue
            
            # 파일 저장
            file_path = user_standards_dir / filename
            try:
                file.save(str(file_path))
                # 파일 권한 설정 (non-root 사용자를 위해)
                try:
                    os.chmod(file_path, 0o666)
                except (OSError, PermissionError):
                    pass  # 권한 설정 실패해도 계속 진행
                
                uploaded_files.append({
                    'name': filename,
                    'size': file_path.stat().st_size,
                    'path': str(file_path)
                })
            except Exception as e:
                errors.append(f'{filename}: 저장 실패 - {str(e)}')
        
        if uploaded_files:
            return jsonify({
                'success': True,
                'message': f'{len(uploaded_files)}개 파일이 업로드되었습니다.',
                'uploadedFiles': uploaded_files,
                'errors': errors if errors else None
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '파일 업로드에 실패했습니다.',
                'errors': errors
            }), 400
            
    except Exception as e:
        logging.error(f'Standard documents upload error: {e}', exc_info=True)
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/standard-documents/list', methods=['GET', 'OPTIONS'])
def list_standard_documents():
    """표준 문서 목록 조회 API (AceBase 로컬 환경용)"""
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        return '', 200
    
    try:
        # user_id 확인
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        # Config import
        from project_generator.config import Config
        
        # 사용자별 디렉토리 경로
        user_standards_dir = Config.COMPANY_STANDARDS_PATH / user_id
        
        if not user_standards_dir.exists():
            return jsonify({'files': []}), 200
        
        # 지원하는 파일 형식
        allowed_extensions = {'.xlsx', '.xls', '.pptx', '.ppt'}
        
        files = []
        for file_path in user_standards_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'size': stat.st_size,
                    'uploadedAt': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'path': str(file_path)
                })
        
        return jsonify({'files': files}), 200
        
    except Exception as e:
        logging.error(f'Standard documents list error: {e}', exc_info=True)
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/standard-documents/delete', methods=['DELETE', 'OPTIONS'])
def delete_standard_document():
    """표준 문서 삭제 API (AceBase 로컬 환경용)"""
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        return '', 200
    
    try:
        # user_id와 filename 확인
        user_id = request.args.get('userId')
        filename = request.args.get('filename')
        
        if not user_id or not filename:
            return jsonify({'error': 'userId and filename are required'}), 400
        
        # Config import
        from project_generator.config import Config
        
        # 파일 경로
        file_path = Config.COMPANY_STANDARDS_PATH / user_id / secure_filename(filename)
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # 파일 삭제
        file_path.unlink()
        
        return jsonify({
            'success': True,
            'message': f'{filename} 파일이 삭제되었습니다.'
        }), 200
        
    except Exception as e:
        logging.error(f'Standard documents delete error: {e}', exc_info=True)
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

def run_healcheck_server():
    """Flask 서버를 별도 스레드에서 실행"""
    # Werkzeug 로거에 헬스체크 필터 적용
    werkzeug_logger = logging.getLogger('werkzeug')
    health_filter = HealthCheckFilter()
    werkzeug_logger.addFilter(health_filter)
    
    app.run(host='0.0.0.0', port=2024, debug=False, use_reloader=False)