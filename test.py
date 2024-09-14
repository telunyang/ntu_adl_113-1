import json

with open('./context.json', "r", encoding="utf-8") as f:
    print(len(json.loads(f.read())))
with open('./train.json', "r", encoding="utf-8") as f:
    print(len(json.loads(f.read())))
with open('./valid.json', "r", encoding="utf-8") as f:
    print(len(json.loads(f.read())))
with open('./test.json', "r", encoding="utf-8") as f:
    print(len(json.loads(f.read())))


'''
context.json: 9013
train.json: 21714
valid.json: 3009
test.json: 2213
'''