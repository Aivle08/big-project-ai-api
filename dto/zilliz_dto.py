from pydantic import Field, BaseModel

class ResumeInsertDTO(BaseModel):
    pdf_name: str = Field('IT영업', description='PDF 파일 이름')
    applicant_id: int = Field(1, description='지원자 id')

class EvalInsertDTO(BaseModel):
    recruitment_id: int = Field(1, description='공고 id')
    detail_list: list = Field(['a', 'b', 'c'], description='평가 항목 상세')

class ResumeDeleteDTO(BaseModel):
    applicant_id: int = Field(1, description='지원자 id')

class EvalDeleteDTO(BaseModel):
    recruitment_id: int = Field(1, description='공고 id')