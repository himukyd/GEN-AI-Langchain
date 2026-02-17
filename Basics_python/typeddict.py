from typing import TypedDict
## Tuped Dictionary is diffetent of dicytionary in this we already mentioned that what is data type and all 
## But if we don't give values as a data type then it won't give any errors 
class dict_1(TypedDict):
    name : str
    id : int

student : dict_1=({'name':'Himanshu' ,'id':12})

print(type(student))