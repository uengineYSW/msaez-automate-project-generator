"""
XmlUtil - Python implementation of frontend XmlUtil.js
Converts dict/list data to XML format for LLM prompts
"""
import json
from typing import Any, Dict, List, Union


class XmlUtil:
    """XML 변환 유틸리티 (프론트엔드 XmlUtil.js와 동일)"""
    
    @staticmethod
    def from_dict(data: Any, is_use_escape_xml: bool = False, to_snake_case: bool = False) -> str:
        """
        딕셔너리나 리스트를 XML 형식으로 변환
        
        Args:
            data: 변환할 데이터 (dict, list, or primitive)
            is_use_escape_xml: XML 특수문자 이스케이프 여부
            to_snake_case: 키를 snake_case로 변환할지 여부
            
        Returns:
            XML 형식 문자열
        """
        def _convert_value_to_xml(value: Any, indent_level: int = 1) -> str:
            indent = "  " * indent_level
            
            if value is None:
                return f"{indent}null"
            
            # 리스트인 경우
            if isinstance(value, list):
                if not value:
                    return f"{indent}<empty_list />"
                
                result = []
                for i, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        result.append(f"{indent}<item>")
                        result.append(_convert_value_to_xml(item, indent_level + 1))
                        result.append(f"{indent}</item>")
                    else:
                        result.append(f"{indent}<item>{_escape_xml(item)}</item>")
                return '\n'.join(result)
            
            # 딕셔너리인 경우
            elif isinstance(value, dict):
                result = []
                for key, val in value.items():
                    # snake_case 변환 (필요시)
                    if to_snake_case:
                        # 간단한 camelCase -> snake_case 변환
                        import re
                        key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
                    
                    if isinstance(val, (list, dict)):
                        # 리스트나 딕셔너리인 경우 재귀적으로 처리
                        result.append(f"{indent}<{key}>")
                        result.append(_convert_value_to_xml(val, indent_level + 1))
                        result.append(f"{indent}</{key}>")
                    else:
                        # 일반 값인 경우
                        result.append(f"{indent}<{key}>{_escape_xml(val)}</{key}>")
                return '\n'.join(result)
            
            # 일반 값인 경우
            else:
                return f"{indent}{_escape_xml(value)}"
        
        def _escape_xml(value: Any) -> str:
            """XML 특수문자 이스케이프"""
            if value is None:
                return ""
            
            # Boolean 값을 문자열로 변환
            if isinstance(value, bool):
                return str(value).lower()
            
            # 문자열인 경우 XML 특수문자 이스케이프
            if isinstance(value, str):
                if is_use_escape_xml:
                    return (value
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                        .replace("'", "&#39;"))
                else:
                    return value
            
            return str(value)
        
        return _convert_value_to_xml(data)

