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
# workflow
from workflow.score_workflow import score_stategraph
# etc
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
# DTO
from dto.score_dto import ScoreDTO
#########################################################################################
score = APIRouter(prefix='/score')

# 요약 Prompt
@score.post("", status_code = status.HTTP_200_OK, tags=['score'])
def summary_graph(item: ScoreDTO):
    '''
    지원자의 자기소개서를 평가하여 점수를 부여하는 LangGraph 기반 워크플로우 실행.
    - 자기소개서와 평가 기준을 비교하여 관련성을 분석하고 점수를 매김.
    - 관련성이 없으면 0점 처리.
    - 사실 검증을 수행하여 평가의 신뢰성을 확보
    '''
    print('\n\033[36m[AI-API] \033[32m 점수 측정')
    try:
        workflow = score_stategraph(StateGraph(ScoreState))

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
                            query_main=f"저는 {item.job} 직무를 수행할 지원자의 {item.eval_item}에 대해 분석하고 싶습니다. 이 기준에 맞는 지원자의 자소서 내용을 뽑아주세요.",
                            resume_chunk=[])

        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'applicant_id:\n{outputs["applicant_id"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'eval_item:\n{outputs["eval_item"]}')
        print(f'eval_item_content:\n{outputs["eval_item_content"]}')
        print(f'eval_resume:\n{outputs["eval_resume"]}')
        print(f'resume_chunk:\n{outputs["resume_chunk"]}')
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "질문 측정 완료",  # 응답 메시지
            'item':{
                f'score':int(outputs["eval_resume"]["eval_resume"][0]),
                f'reason':outputs["eval_resume"]["eval_resume"][1],
                'chunk':outputs['resume_chunk']
            }
        }
    except RecursionError:  # 재귀 한도 초과 시 예외 처리
        print("\033[31m[재귀 한도 초과]\033[0m")
        return {
            "status": "success",  
            "code": 204,  
            "message": "재귀 한도를 초과하여 판단 불가.",  
            'item': {
                f'score':None,
                f'reason':None,
                'chunk':None,
            }
        }
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }

