#########################################################################################
# Basic
import os
import pandas as pd
import traceback
# Fastapi
from fastapi import APIRouter, HTTPException, status, File, UploadFile

# LangChain
from langchain_core.runnables import RunnableConfig

# Graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

## Module
# State
from state.summary_state import SummaryState, extractionState
# Node
from node.summary_node import resume_load, resume_summary, fact_checking, retrieve_document, combine_prompt
# etc
from etc.etcc import summary_is_fact
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.summary_prompt import summary_prompt, extraction_prompt
# DTO
from dto.summary_dto import SummaryDTO, ExtractionDTO
#########################################################################################
summary = APIRouter(prefix='/summary')

# 요약 Prompt
@summary.post("", status_code = status.HTTP_200_OK, tags=['summary'])
def summary_graph(item: SummaryDTO):
    print('\n\033[36m[AI-API] \033[32m 자소서 요약')
    try:
        workflow = StateGraph(SummaryState)

        # 1. Node 추가
        workflow.add_node(
            "resume_document",
            lambda state: {"resume": resume_load(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "summary_document",
            lambda state: {"summary_result": resume_summary(state, summary_prompt)},
        )
        workflow.add_node("fact_checking", fact_checking)
        
        # 2. Edge 연결
        workflow.add_edge('resume_document','summary_document')
        workflow.add_edge('summary_document','fact_checking')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "fact_checking",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            summary_is_fact,
            {"fact": END,
                "not_fact": "summary_document",  # 사실이 아니면 다시 요약합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("resume_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'tech_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})


        # 9. 질문 입력
        inputs = SummaryState(job=item.job, 
                            applicant_id = item.applicant_id)

        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'applicant_id:\n{outputs["applicant_id"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'yes_or_no:\n{outputs["yes_or_no"]}')
        print(f'summary_result:\n{outputs["summary_result"]}')
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "자소서 DB 추출 완료",  # 응답 메시지 
            "item" : outputs["summary_result"]
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }


# 요약 Prompt
@summary.post("/extraction", status_code = status.HTTP_200_OK, tags=['summary'])
def tech_prompt(item: ExtractionDTO):
    print('\n\033[36m[AI-API] \033[32m 자소서 인적사항')
    try:
        workflow = StateGraph(extractionState)

        # 워크플로우에 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_result": combine_prompt(state, extraction_prompt)},
        )

        # Edge
        workflow.add_edge('retrieve_1_document','combine_prompt')
        
        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'extraction_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_output_form = """
        {
            "name": ,
            "phone": ,
            "email": ,
            "birth": ,
            "address": ,
            "else_summary":
            출신학교(University or School) 및 전공(Major) 정보를 포함하고, 경력 항목에서 이전 직무 경험을 자연스럽게 연결하여 기술해야 합니다. 
            또한, 자격증 및 어학 능력, 대외활동 및 기타 정보를 서술식으로 작성하되, 문장 간 개행 없이 하나의 단락으로 연결하여 표현해야 합니다. 
            마지막으로, 주요 내용에서는 핵심 역량과 차별점을 요약하여 지원하는 직무와의 적합성을 효과적으로 설명해야 합니다. 
            출력 시 줄바꿈('\n') 없이 한 문단으로 서술하고, 간결하면서도 논리적으로 연결된 문장으로 작성해야 합니다.
        }
        """

        # 질문 입력
        inputs = extractionState(
            applicant_id = item.applicant_id,
            query_main=f'이력서 요약을 해주세요.',
            output_form=input_output_form)

        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'final_result:\n{outputs["final_result"]}')
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "자소서 DB 추출 완료",  # 응답 메시지 
            "item": outputs["final_result"]
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }