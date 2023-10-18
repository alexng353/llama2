import json

my_dict = {"name": "John", "age": 30, "city": "New York"}

json_str = json.dumps(my_dict, separators=(',', ':'))

print(json_str)
