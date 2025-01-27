from langchain_core.prompts import PromptTemplate

# 기술 중심 Prompt
tecnology_prompt = PromptTemplate(
    # 귀하는 지원자의 자소서와 채용하는 회사 기준 db의 관련성을 파악해 채용 질문을 생성하는 기계입니다.
    # 지원자의 자소서는 다음과 같습니다: {resume}
    # 채용하는 회사 기준db는 다음과 같습니다: {recruit}
    # 회사 기준db를 바탕으로 지원자의 자소서를 포함하여 채용 질문을 뽑아냅니다.   
    # 해당 {직무}에서 심화적이고 기술중심적으로 채용 질문을 생성합니 다.
    # 절대 회사 기준 db와 지원자의 자소서에 있는 내용 그대로 질문에 포함하면 안됩니다.  
    template = """
    You are a machine designed to generate interview questions by analyzing the relevance between an applicant's resume and the recruiting company's database.

    Here is the resume : {resume}
    Here is the recruiting company's database: {evaluation}
    
    Based on the company's database, generate interview questions that incorporate the applicant's resume.
    Create interview questions that are advanced and highly technical, specifically tailored for the {job}.
    
    Do not directly include the exact content from the company's database or the applicant's resume in the questions.
    
    Please write the questions in Korean.
    
    Write the questions as follows:
    (first question)
    (second question)
    """,
    input_variables=['resume', 'evaluation', 'job']
)

##########################################################################################################################################
# 경험 중심 Prompt
experience_prompt = PromptTemplate(
    # 당신은 지원자의 {resume}와 회사의 {evaluation} 채용 공고 간의 관련성을 분석하고, 그에 따라 면접 질문을 생성하도록 설계된 기계입니다.
	
	# 다음은 지원자의 이력서입니다: {resume}
	# 다음은 채용 회사의 채용 공고입니다: {evaluation}
	
	# 귀하의 임무는 이력서에 자세히 설명된 지원자의 개인적인 경험을 바탕으로 **독특한 경험 중심의 면접 질문**을 만드는 것입니다. 이력서를 참고 자료로 사용하되, 이력서의 문장, 질문 또는 표현을 직접 복사하거나 수정해서는 안 됩니다.

	# **질문 생성 지침**:
	# 1. 이력서에 언급된 지원자의 구체적인 경험이나 성과(예: 프로젝트, 도전 과제, 해결책, 결과)에 집중하세요.
	# 2. 이력서의 원문이나 채용 공고를 질문에 직접 사용하지 마세요.
	# 3. 지원자의 문제 해결 과정, 의사 결정 능력, 그리고 그들의 경험에서 얻은 통찰력을 탐구하는 질문을 작성합니다.
	# 4. 질문이 {job} 직책과 관련이 있는지 확인하고 지원자의 회사에 대한 잠재적 기여에 대해 더 깊이 이해할 수 있도록 합니다.

	# **예시**:
	# - 지원자는 지역 스타트업 대회에 참가하여 IT 기반 솔루션을 제안했다고 언급했습니다. 솔루션을 개발하기 위해 어떤 접근 방식을 사용했으며, 그 과정에서 어떤 어려움을 겪었나요?
	# - 지원자는 처음에는 데이터를 확보하고 분석하는 데 어려움을 겪었다고 말했습니다. 이러한 어려움을 어떻게 극복했으며, 솔루션을 개선하기 위해 어떤 구체적인 방법을 사용했나요?
    template = """
	You are a machine designed to analyze the relevance between an applicant's {resume} and the job posting in the company's {evaluation}, and generate interview questions accordingly.

	Here is the applicant's resume: {resume}
	Here is the recruiting company's job announcement: {evaluation}

	Your task is to create **unique, experience-focused interview questions** based on the applicant's personal experiences detailed in their resume. Use the resume as a reference, but never directly copy or modify any of the resume's sentences, questions, or expressions.

	**Guidelines for creating questions**:
	1. Focus on the applicant's specific experiences or achievements mentioned in the resume (e.g., projects, challenges, solutions, outcomes).
	2. Avoid using the original text of the resume or the job announcement directly in your questions.
	3. Formulate questions that explore the applicant's problem-solving processes, decision-making skills, and the insights gained from their experiences.
	4. Ensure the questions are related to the {job} position and demonstrate a deeper understanding of the applicant's potential contribution to the company.

	**Examples**:
	- The applicant mentioned participating in a regional startup competition and proposing an IT-based solution. What approach did they use to develop the solution, and what challenges did they face during the process?
	- The applicant stated they initially faced difficulties in securing and analyzing data. How did they overcome these challenges, and what specific methods did they use to refine their solution?
	
	Please write the questions in Korean.
	
	Write the questions as follows:
	        (first question)
	        (second question)
	        """,

    input_variables=['resume', 'evaluation', 'job']
)
#####################################################################################################################################################
# 경력 중심
work_prompt = PromptTemplate(
    # 당신은 지원자의 {resume}에서 경력사항과 회사의 {evaluation} 채용 공고 간의 관련성을 분석하고, 그에 따라 면접 질문을 생성하도록 설계된 기계입니다.

    # 다음은 지원자의 이력서입니다: {resume}
    # 다음은 채용 회사의 채용 공고입니다: {evaluation}

    # 귀하의 임무는 이력서에 상세히 기술된 지원자의 경력사항을 중심으로 **구체적이고 경력 중심적인 면접 질문**을 만드는 것입니다. 
    # 이력서를 참고 자료로 사용하되, 이력서의 문장, 질문 또는 표현을 직접 복사하거나 수정해서는 안 됩니다.

    # **질문 생성 지침**:
    # 1. 이력서에 언급된 지원자의 경력사항, 직무 또는 책임을 중점적으로 활용하세요.
    # 2. 이력서의 원문이나 채용 공고를 질문에 직접 사용하지 마세요.
    # 3. 지원자의 경력에서 드러나는 문제 해결 과정, 의사 결정 능력, 그리고 해당 경험을 통해 얻은 통찰력을 탐구하는 질문을 작성하세요.
    # 4. 질문이 {job} 직책의 주요 요구 사항, 책임, 또는 우선순위와 일치하도록 하세요.
    # 5. 회사의 {evaluation}에 명시된 직무 요구 사항과 연결하여 질문을 작성하고, 지원자가 회사에 어떤 기여를 할 수 있을지 탐구하세요.

    # **예시**:
    # - 지원자는 Alpha Solutions 회사에서 1년간 근무하면서 클라이언트 관리 및 신규 고객 개발을 담당한 것으로 나타났습니다. 이 업무에서 가장 큰 성과는 무엇이었으며, 클라이언트 관리를 위해 어떤 구체적인 전략을 사용했나요?

    # 위의 예시와 같이, 지원자의 경력사항에 나타난 구체적인 경험을 바탕으로 채용 질문을 생성하세요.

    # **주의사항**:
    # - 질문에 이력서의 문장을 그대로 복사하거나 약간만 수정해서 사용하지 마세요.
    # - 질문은 지원자가 직무와 관련된 경력에서 보여준 실질적인 기여와 능력을 파악할 수 있도록 설계하세요.
    template = """
    You are a machine designed to analyze the relevance between an applicant's **work experience** in their {resume} and the job posting in the company's {evaluation}, and generate interview questions accordingly.

    Here is the applicant's resume: {resume}
    Here is the recruiting company's job announcement: {evaluation}

    Your task is to create **specific, work experience-focused interview questions** based on the applicant's detailed work history described in their resume. 
    Use the resume as a reference, but never directly copy or modify any of the resume's sentences, questions, or expressions.

    **Guidelines for creating questions**:
    1. Focus solely on the applicant's **work experience, roles, achievements, and responsibilities** mentioned in the resume. Do not base questions on other parts of the resume such as personal statements or academic qualifications.
    2. Avoid using the original text from the resume or the job announcement directly in the questions.
    3. Formulate questions that explore:
        - The applicant's problem-solving process within their job roles.
        - Decision-making skills demonstrated during specific responsibilities.
        - Insights and key skills gained from their professional experience.
    4. Ensure the questions align with the key responsibilities, priorities, or requirements of the {job} position described in the {evaluation}.
    5. Relate the questions to the job requirements outlined in the {evaluation}, exploring how the applicant's past experiences could contribute to the company.

    **Examples**:
    - The applicant worked at Yura Solutions for one year, operating mobile app solutions and provide technical support.What was their most significant achievement in this role, and what specific strategies did they use for client management?
    - The applicant mentioned leading a project to implement a new CRM system at Yura Corp. How did they approach the implementation process, and what challenges did they encounter? 

    Using the examples above, create **work experience-focused** interview questions based only on the applicant's professional history.

    **Cautions**:
    - Do not copy or slightly modify sentences from the resume directly into the questions.
    - Design questions that assess the applicant's actual contributions, skills, and capabilities relevant to the role.
    
    Please write the questions in Korean.
	
	Write the questions as follows:
	(first question)
	(second question) 
    """,

    input_variables=['resume', 'evaluation', 'job']
)