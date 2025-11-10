"""
DDL Fields Assignment Generator for LangGraph
Assigns DDL fields to appropriate aggregates based on domain semantics
"""

from typing import TypedDict, List, Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ...utils.logging_util import LoggingUtil


# ===== State Definition =====
class DDLFieldsGeneratorState(TypedDict):
    """State for DDL Fields Assignment workflow"""
    # Input
    description: str
    aggregate_drafts: List[Dict[str, str]]
    all_ddl_fields: List[str]
    generator_key: str
    
    # Output
    inference: str
    result: Dict[str, Any]
    
    # Metadata
    error: str
    timestamp: str


# ===== Main Generator Class =====
class DDLFieldsGenerator:
    """
    Assigns DDL fields to aggregates using LangGraph workflow
    """
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=False
        )
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(DDLFieldsGeneratorState)
        
        # Add nodes
        workflow.add_node("assign_fields", self.assign_fields_node)
        workflow.add_node("validate_assignments", self.validate_assignments_node)
        workflow.add_node("finalize_output", self.finalize_output_node)
        
        # Add edges
        workflow.set_entry_point("assign_fields")
        workflow.add_edge("assign_fields", "validate_assignments")
        workflow.add_edge("validate_assignments", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        return workflow.compile()
    
    def assign_fields_node(self, state: DDLFieldsGeneratorState) -> DDLFieldsGeneratorState:
        """Node: Assign DDL fields to aggregates using LLM"""
        LoggingUtil.info("DDLFieldsGenerator", "Starting field assignment")
        
        try:
            # Define JSON schema for structured output
            schema = {
                "type": "object",
                "title": "DDLFieldAssignments",
                "description": "Assignment of DDL fields to aggregates",
                "properties": {
                    "inference": {
                        "type": "string",
                        "description": "Detailed reasoning for the field assignments"
                    },
                    "result": {
                        "type": "object",
                        "properties": {
                            "aggregateFieldAssignments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "aggregateName": {"type": "string"},
                                        "ddl_fields": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["aggregateName", "ddl_fields"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["aggregateFieldAssignments"],
                        "additionalProperties": False
                    }
                },
                "required": ["inference", "result"],
                "additionalProperties": False
            }
            
            # Create structured output LLM
            structured_llm = self.llm.with_structured_output(schema, strict=True)
            
            # Build prompt
            prompt = self._build_prompt(state)
            
            # Call LLM
            response = structured_llm.invoke(prompt)
            
            LoggingUtil.info("DDLFieldsGenerator", "Field assignment completed")
            
            return {
                **state,
                "inference": response.get("inference", ""),
                "result": response.get("result", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Field assignment failed: {str(e)}"
            LoggingUtil.error("DDLFieldsGenerator", error_msg)
            return {
                **state,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_assignments_node(self, state: DDLFieldsGeneratorState) -> DDLFieldsGeneratorState:
        """Node: Validate field assignments"""
        LoggingUtil.info("DDLFieldsGenerator", "Validating field assignments")
        
        if state.get("error"):
            return state
        
        try:
            input_fields = state["all_ddl_fields"]
            assignments = state["result"].get("aggregateFieldAssignments", [])
            
            # Collect all assigned fields
            assigned_fields = []
            for assignment in assignments:
                assigned_fields.extend(assignment.get("ddl_fields", []))
            
            # Check for extra fields (not in input)
            extra_fields = [f for f in assigned_fields if f not in input_fields]
            if extra_fields:
                # remove extra fields (debug-only log removed)
                # Remove extra fields from assignments
                for assignment in assignments:
                    assignment["ddl_fields"] = [f for f in assignment["ddl_fields"] if f in input_fields]
            
            # Re-collect assigned fields after removal
            final_assigned_fields = []
            for assignment in assignments:
                final_assigned_fields.extend(assignment.get("ddl_fields", []))
            
            # Check for missing fields
            missing_fields = [f for f in input_fields if f not in final_assigned_fields]
            if missing_fields:
                error_msg = f"Missing field assignments: {', '.join(missing_fields)}. All DDL fields must be assigned."
                LoggingUtil.error("DDLFieldsGenerator", error_msg)
                return {
                    **state,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            
            LoggingUtil.info("DDLFieldsGenerator", f"Validation passed: {len(input_fields)} fields assigned")
            
            return {
                **state,
                "result": {"aggregateFieldAssignments": assignments},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            LoggingUtil.error("DDLFieldsGenerator", error_msg)
            return {
                **state,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def finalize_output_node(self, state: DDLFieldsGeneratorState) -> DDLFieldsGeneratorState:
        """Node: Finalize output format"""
        LoggingUtil.info("DDLFieldsGenerator", "Finalizing output")
        
        if state.get("error"):
            return state
        
        try:
            # Output structure is already correct
            return {
                **state,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Output finalization failed: {str(e)}"
            LoggingUtil.error("DDLFieldsGenerator", error_msg)
            return {
                **state,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_prompt(self, state: DDLFieldsGeneratorState) -> str:
        """Build the prompt for field assignment"""
        
        # Format aggregate drafts
        aggregates_str = "\n".join([
            f"  - {agg['name']} ({agg.get('alias', agg['name'])})"
            for agg in state["aggregate_drafts"]
        ])
        
        # Format DDL fields
        fields_str = "\n".join([f"  - {field}" for field in state["all_ddl_fields"]])
        
        return f"""Role: Domain-Driven Design (DDD) Data Modeling Specialist

Goal: To analyze the functional requirements and aggregate draft structures within a bounded context, then intelligently assign DDL fields to the most appropriate aggregates based on domain semantics and data relationships. Your primary function is to ensure that every field from the DDL is assigned to exactly one aggregate, creating a complete and semantically correct mapping.

Backstory: With extensive experience in database design and domain modeling, I specialize in bridging the gap between technical database schemas and business domain models. I understand how to interpret DDL field names, data types, and relationships to determine which aggregate should own which data. My expertise lies in analyzing business context, understanding aggregate boundaries, and making intelligent decisions about data ownership that maintain both technical correctness and domain coherence.

Operational Guidelines:
* **Analyze Domain Context:** Thoroughly understand the functional requirements and business context to make informed decisions about data ownership.
* **Respect Aggregate Boundaries:** Ensure that each field is assigned to the aggregate that naturally owns that data from a business perspective.
* **Maintain Completeness:** Every DDL field must be assigned to exactly one aggregate - no field should be left unassigned.
* **Consider Relationships:** Analyze field naming patterns (e.g., foreign keys, timestamps, status fields) to determine logical ownership.
* **Prioritize Semantic Cohesion:** Group related fields together and assign them to aggregates where they form a cohesive concept.
* **Handle Edge Cases:** When field ownership is ambiguous, prioritize the aggregate that would most directly use or modify that data.
* **Provide Clear Reasoning:** Document your decision-making process for each assignment to ensure transparency and maintainability.

Your task is to analyze a bounded context's DDL fields and assign each field to the most appropriate aggregate draft.

Assignment Rules:
1. **Complete Coverage:** Every field in the "All DDL Fields" list must be assigned to exactly one aggregate.
2. **No Duplicates:** Each field should appear only once across all aggregate assignments.
3. **Semantic Alignment:** Assign fields to aggregates based on business logic and domain semantics, not just naming patterns.
4. **Primary Entity Focus:** Core identifying fields (IDs, primary keys) should typically go to their corresponding aggregate.

Field Analysis Guidelines:
5. **ID Fields:** Fields ending with "_id" or containing "id" typically belong to the aggregate they identify or reference.
6. **Timestamp Fields:** Created/updated timestamps usually belong to the main entity they track.
7. **Status/State Fields:** Status and state fields belong to the aggregate whose lifecycle they describe.
8. **Descriptive Fields:** Name, description, and similar fields belong to the entity they describe.
9. **Foreign Key Analysis:** Foreign key fields should be assigned based on the relationship direction and ownership.

Domain Context Considerations:
10. **Business Logic:** Consider which aggregate would naturally create, modify, or validate each field.
11. **Transaction Boundaries:** Fields that change together in business operations should typically be in the same aggregate.
12. **Data Lifecycle:** Consider the lifecycle of data - which aggregate controls when this data is created, updated, or deleted.

Inference Guidelines:
1. **Start with Context Analysis:** Begin by understanding the overall business domain and the role of each planned aggregate.
2. **Group Related Fields:** Identify clusters of related fields that naturally belong together.
3. **Apply Domain Logic:** Use business understanding to determine which aggregate should own each field or field group.
4. **Handle Ambiguous Cases:** For fields that could belong to multiple aggregates, explain your reasoning and prioritize based on:
   - Which aggregate would most directly use the field
   - Which aggregate controls the field's lifecycle
   - Business transaction boundaries
5. **Verify Completeness:** Ensure every DDL field is assigned and no field is duplicated across aggregates.
6. **Document Edge Cases:** Clearly explain any assignments where the field might not perfectly fit the aggregate but represents the best available option.

# Input:

## Functional Requirements:
{state['description']}

## Aggregate Drafts:
{aggregates_str}

## All DDL Fields:
{fields_str}

# Expected Output Format:

{{
    "inference": "<Detailed reasoning for the field assignments, including analysis of the domain context and explanation of assignment decisions>",
    "result": {{
        "aggregateFieldAssignments": [
            {{
                "aggregateName": "<name_of_aggregate>",
                "ddl_fields": [
                    "<field_name_1>",
                    "<field_name_2>"
                ]
            }}
        ]
    }}
}}

# Example:

Input:
- Functional Requirements: "CourseManagement context handles course lifecycle, instructor assignments, pricing, and student enrollments."
- Aggregate Drafts: [Course, Enrollment]
- All DDL Fields: [course_id, title, description, instructor_id, status, price_amount, price_currency, created_at, updated_at, enrollment_id, student_id, enrollment_date, completion_status]

Output:
{{
    "inference": "I analyzed the CourseManagement domain and assigned fields based on aggregate ownership principles. Course-related fields (course_id, title, description, instructor_id, status, price_amount, price_currency, created_at, updated_at) naturally belong to the Course aggregate as they describe course properties and lifecycle. Enrollment-related fields (enrollment_id, student_id, enrollment_date, completion_status) belong to the Enrollment aggregate as they track the student-course relationship and enrollment lifecycle.",
    "result": {{
        "aggregateFieldAssignments": [
            {{
                "aggregateName": "Course",
                "ddl_fields": ["course_id", "title", "description", "instructor_id", "status", "price_amount", "price_currency", "created_at", "updated_at"]
            }},
            {{
                "aggregateName": "Enrollment",
                "ddl_fields": ["enrollment_id", "student_id", "enrollment_date", "completion_status"]
            }}
        ]
    }}
}}

Now please assign the DDL fields to the appropriate aggregates based on the provided context.
"""
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for generating DDL field assignments
        
        Args:
            input_data: Dictionary containing:
                - description: Functional requirements
                - aggregate_drafts: List of aggregate drafts
                - all_ddl_fields: List of DDL field names
                - generator_key: Identifier for this generation
        
        Returns:
            Dictionary containing inference and result
        """
        LoggingUtil.info("DDLFieldsGenerator", "Starting DDL fields assignment")
        
        initial_state = {
            "description": input_data.get("description", ""),
            "aggregate_drafts": input_data.get("aggregate_drafts", []),
            "all_ddl_fields": input_data.get("all_ddl_fields", []),
            "generator_key": input_data.get("generator_key", "unknown"),
            "inference": "",
            "result": {},
            "error": "",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            final_state = self.workflow.invoke(initial_state)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            LoggingUtil.info("DDLFieldsGenerator", "DDL fields assignment completed")
            
            return {
                "inference": final_state.get("inference", ""),
                "result": final_state.get("result", {}),
                "timestamp": final_state.get("timestamp", "")
            }
            
        except Exception as e:
            error_msg = f"DDL fields assignment failed: {str(e)}"
            LoggingUtil.error("DDLFieldsGenerator", error_msg)
            raise Exception(error_msg)


# ===== Convenience function for testing =====
def generate_ddl_field_assignments(
    description: str,
    aggregate_drafts: List[Dict[str, str]],
    all_ddl_fields: List[str],
    generator_key: str = "test",
    model_name: str = "gpt-4o-2024-08-06"
) -> Dict[str, Any]:
    """
    Convenience function to generate DDL field assignments
    """
    generator = DDLFieldsGenerator(model_name=model_name)
    
    input_data = {
        "description": description,
        "aggregate_drafts": aggregate_drafts,
        "all_ddl_fields": all_ddl_fields,
        "generator_key": generator_key
    }
    
    return generator.generate(input_data)

