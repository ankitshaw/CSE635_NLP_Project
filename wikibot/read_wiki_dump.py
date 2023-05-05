
import time
import json
start = time.time()
count = 0
# with open('D:\\NLP\\enwiki-latest-pages-articles-xml\\enwiki-latest-pages-articles.xml', encoding="utf8") as file:
with open('D:\\NLP\\enwiki-20230501-cirrussearch-content.json', encoding="utf8") as file:
    for line in file:
    #    print(line[0:20000])
        if count%2==1:
            print(json.loads(line)['text'])
        count = count + 1
        if count == 2:
            break
end =  time.time()
print("Execution time in seconds: ",(end-start))
print("No of lines printed: ",count)