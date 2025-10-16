from flask import Flask, jsonify, request
import logging

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

def run_healcheck_server():
    """Flask 서버를 별도 스레드에서 실행"""
    # Werkzeug 로거에 헬스체크 필터 적용
    werkzeug_logger = logging.getLogger('werkzeug')
    health_filter = HealthCheckFilter()
    werkzeug_logger.addFilter(health_filter)
    
    app.run(host='0.0.0.0', port=2024, debug=False, use_reloader=False)