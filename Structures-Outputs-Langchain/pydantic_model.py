from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=5)
    
new_student={'name':'niraj','email':'nirj@gm.com','cgpa':5.0}

student=Student(**new_student)

print(student)
print(type(student))
    
    