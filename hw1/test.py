# import json

# with open('./context.json', "r", encoding="utf-8") as f:
#     print(len(json.loads(f.read())))
# with open('./train.json', "r", encoding="utf-8") as f:
#     print(len(json.loads(f.read())))
# with open('./valid.json', "r", encoding="utf-8") as f:
#     print(len(json.loads(f.read())))
# with open('./test.json', "r", encoding="utf-8") as f:
#     print(len(json.loads(f.read())))


# '''
# context.json: 9013
# train.json: 21714
# valid.json: 3009
# test.json: 2213
# '''

from transformers import AutoTokenizer
# model_path = 'bert-base-chinese'
model_path = './models_span_selection'
tokenizer = AutoTokenizer.from_pretrained(model_path)
s = '南海是「南」的一部分，是廣府民系的重要聚居地。6000年前，南海開始出現「西樵山文化」。隋開皇十年設置南海縣，縣署駐廣州。唐屬廣州都督府。北宋太祖開寶四年屬廣南東路，五年屬廣州。元世祖至元十五年屬廣東道廣州路。明洪武元年屬廣州府。清屬廣東省廣州府，為廣州府兩個附郭縣之一，其縣署置於廣州城內。今天廣州仍有舊南海縣街，今天北京路以西及整個荔灣區，俱是昔日南海縣管豁地域。終清一代，雖然與番禺同屬附郭縣，但舉凡兩廣總督官署、廣東巡撫衙門、廣州將軍衙門、廣東水師提督衙門、廣州府衙、廣東省學宮都位於南海縣界內，而清廷沒有一個府級官署置在番禺境內，故省會實屬南海縣。 1911年辛亥革命後屬粵海道。1912年縣署遷佛山鎮。1920年廢道後直屬省。民國21年屬中區綏靖公署，25年屬第一區行政督察專員公署。抗日戰爭期間，縣治曾遷九江西岸。1950年3月成立縣人民政府。1951年1月經政務院批准，佛山撤鎮設市，南海縣與之分治，縣人民政府仍駐佛山市城區。1950年1月至1952年11月屬珠江專員公署；1952年11月至1956年2月屬粵中行政公署；1956年3月至1958年11月屬佛山專員公署；1958年11月至1959年1月屬廣州專員公署；1959年1月至1967年3月屬佛山專員公署；1967年3月至1968年3月屬佛山地區軍事管制委員會軍管；1968年3月至1979年3月為佛山專區革命委員會管轄；1979年3月至1983年6月屬佛山地區行政公署；1983年6月，廣東省實行市領導縣的體制，佛山地、市合併，南海縣隸屬於佛山市。'
tokenized_tokens = tokenizer.encode(
    s, 
    padding=True, 
    truncation=False, 
    max_length=1000, 
    add_special_tokens=True
)
print(len(tokenized_tokens))