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
    # 귀하는 지원자의 자소서와 채용하는 회사 기준 db의 {공고}를 보고 관련성을 파악해 채용 질문을 생성하는 기계입니다.
    # 지원자의 자소서는 다음과 같습니다: {resume}
    # 채용하는 회사 기준db의 공고는 다음과 같습니다: {evaluation}
    # 회사 기준db의 공고{evaluation}를 참고 지원자의 자소서를 중심으로 채용 질문을 뽑아냅니다.
    # 지원자의 자소서{resume}에 있는 구체적인 경험을 중심으로 질문을 생성합니다. 구체적인 예시는 다음과 같습니다.

    # 예시: 면접자 자소서를 보면 지역 축제 홍보 캠페인을 주도하셨다고 했는데 그 활동에서 어떤 부분을 담당하였고, 거기서 중요하게 생각하는 점은 무엇인지?

    # 절대 회사 기준 db와 지원자의 자소서에 있는 내용 그대로 질문에 포함하면 안됩니다.
    template = """
        You are a machine designed to generate interview questions by analyzing the relevance between an applicant's resume and the recruiting company's database.

        Here is the applicant's resume: {resume}
        Here is the recruiting company's job announcement: {evaluation}

        Based on the company's job announcement, generate unique interview questions that focus on the applicant's specific achievements or experiences. Avoid rephrasing or reusing the exact sentences, questions, or expressions found in the applicant's resume.

        Focus on extracting implicit ideas or details from the resume that are relevant to the {job} position. Formulate advanced and insightful questions that explore the applicant's problem-solving abilities, technical expertise, and potential contributions to the role.

        Rules:
        DO NOT directly use or modify sentences, phrases, or questions from the resume or job announcement.
        Avoid surface-level questions and instead create questions that dig deeper into the applicant's unique experiences.
        Ensure the questions are aligned with the {job} role but do not explicitly reference resume text.
        
        Please write the questions in Korean.
        
        Write the questions as follows:
        (first question)
        (second question)
        """,

    input_variables=['resume', 'evaluation', 'job']
)