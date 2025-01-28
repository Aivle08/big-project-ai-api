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
from state.summary_state import SummaryState, yonghunState
# Node
from node.summary_node import resume_load, resume_summary, fact_checking, retrieve_document, combine_prompt
# etc
from etc.etcc import is_fact
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.summary_prompt import summary_prompt
#########################################################################################
summary = APIRouter(prefix='/summary')

# 요약 Prompt
@summary.post("", status_code = status.HTTP_200_OK, tags=['summary'])
def summary_graph():
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
            is_fact,
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
        input_job = 'IT영업'
        inputs = SummaryState(job=input_job, 
                            applicant_id = 1)

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
        
        return outputs["summary_result"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }


# 요약 Prompt
@summary.post("/yonghun", status_code = status.HTTP_200_OK, tags=['summary'])
def tech_prompt():
    print('\n\033[36m[AI-API] \033[32m 자소서 인적사항')
    try:
        workflow = StateGraph(yonghunState)

        # 워크플로우에 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        #workflow.add_node("relevance_check", relevance_check)
        workflow.add_node("combine_prompt", combine_prompt)

        # Edge
        workflow.add_edge('retrieve_1_document','combine_prompt')
        
        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'younhun_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_output_form = """
        {
            "name": ,
            "phone": ,
            "email": ,
            "birth": ,
            "address": ,
            "else_summary": {
                "university": ,
                "major": ,
                "second_major": ,
                "score": ,
                "경력": [
                ],
                "자격증/어학": [
                ],
                "대외활동 및 기타": [
                ],
            }
        }
        """

        # 질문 입력
        inputs = yonghunState(
            applicant_id = 1,
            query_main=f'이력서 요약을 해주세요.',
            output_form=input_output_form)

        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'final_result:\n{outputs["final_result"]}')
        
        return outputs["final_result"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }