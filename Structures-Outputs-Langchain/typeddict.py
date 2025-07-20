from typing import TypedDict

class Person(TypedDict):
    
    name:str
    age:int
    
new_person:Person={'name':'niraj','age':'36'}

print(new_person)

#output : {'name': 'niraj', 'age': '36'}