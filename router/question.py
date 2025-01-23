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
from state.question_state import GraphState
# Node
from node.question_chain import retrieve_document, relevance_check, combine_prompt
# etc
from etc.etcc import is_relevant
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
#########################################################################################
question = APIRouter(prefix='/question')

path = './'

# 기술 중심 Prompt
@question.post("/tech", status_code = status.HTTP_200_OK, tags=['question'])
async def tech_prompt():
    print('\n\033[36m[AI-API] \033[32m 질문 추출')
    try:
        workflow = StateGraph(GraphState)

        # # retriever 노드 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "retrieve_2_document",
            lambda state: {"evaluation": retrieve_document(state, "evaluation", 'company_id')},
        )
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node("combine_prompt", combine_prompt)

        # Edge
        workflow.add_edge('retrieve_1_document','retrieve_2_document')
        workflow.add_edge('retrieve_2_document','combine_prompt')
        workflow.add_edge('combine_prompt','relevance_check')
        
        # 조건부 엣지를 추가합니다.
        workflow.add_conditional_edges(
            "combine_prompt",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            is_relevant,
            {
                "relevant": END,  # 관련성이 있으면 답변을 생성합니다.
                "not relevant": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 체크포인터 설정
        memory = MemorySaver()

        # 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        visualize_graph(app)
            
        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_job = 'IT영업'

        # 질문 입력
        inputs = GraphState(job=input_job, 
                            company_id = 1,
                            applicant_id = 2,
                            relevance='no',
                            query_main=f'{input_job}의 기술 중심으로 생성해줘' )


        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'query_main:\n{outputs["query_main"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance:\n{outputs["relevance"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return outputs["final_question"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
