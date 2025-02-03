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
from state.score_state import ScoreState
# Node
from node.score_node import retrieve_document, score_resume, fact_checking, relevance_check
# etc
from etc.etcc import score_is_fact, score_is_relevant
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.score_prompt import score_prompt
# DTO
from dto.score_dto import ScoreDTO
#########################################################################################
score = APIRouter(prefix='/score')

# 요약 Prompt
@score.post("", status_code = status.HTTP_200_OK, tags=['score'])
def summary_graph(item: ScoreDTO):
    print('\n\033[36m[AI-API] \033[32m 점수 측정')
    try:
        workflow = StateGraph(ScoreState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "score_resume",
            lambda state: {"eval_resume": score_resume(state, score_prompt)},
        )
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node("fact_checking", fact_checking)
        
        # 2. Edge 연결
        workflow.add_edge('retrieve_document','relevance_check')
        workflow.add_edge('score_resume','fact_checking')
        
        #3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            score_is_relevant,
            {"relevant": 'score_resume',
            "not_relevant": "retrieve_document",  # 사실이 아니면 다시 요약합니다.
            },
        )
        workflow.add_conditional_edges(
            "fact_checking",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            score_is_fact,
            {"fact": END,
            "not_fact": "score_resume",  # 사실이 아니면 다시 요약합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'eval_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        
        # 9. 질문 입력
        inputs = ScoreState(job = item.job, 
                            applicant_id = item.applicant_id,
                            eval_item = item.eval_item,
                            eval_item_content = item.eval_item_content,
                            query_main=f'{item.job} 직무에 대한 {item.eval_item} 뽑아주세요.')

        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'applicant_id:\n{outputs["applicant_id"]}')
        print(f'resume:\n{outputs["resume"]}')
        #print(f'yes_or_no:\n{outputs["yes_or_no"]}')
        print(f'eval_resume:\n{outputs["eval_resume"]}')
        
        return {
            f'{item.eval_item}':int(outputs["eval_resume"]["eval_resume"][0]),
            f'{item.eval_item}평가이유':outputs["eval_resume"]["eval_resume"][1]
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }

