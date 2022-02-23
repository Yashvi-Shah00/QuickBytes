import newspaper
import random

news_paper1 = newspaper.build('https://www.ndtv.com/',memoize_articles=False)
news_paper2 = newspaper.build('https://indianexpress.com/',memoize_articles=False)
news_paper3 = newspaper.build('https://www.hindustantimes.com/',memoize_articles=False)
# news_paper3 = newspaper.build('https://www.thehindu.com/',memoize_articles=False)

urlData1 = []
titleData1 = []
urlData2 = []
urlData3 = []
final_url_list = []
final_title_list = []

for article in news_paper1.articles:
  urlData1.append(article.url)
  titleData1.append(article.title)
  # print("Title",article.title)
  # print("Article",article)


for article in news_paper2.articles:
  urlData2.append(article.url)

for article in news_paper3.articles:
  urlData3.append(article.url)

# print(urlData2)
# print(urlData3)
def crawl_website(urlList,titleList):
  for i in range(0,5):
    num = random.randint(20, 100)
    print(num)
    required_url = urlList[num]
    required_title = titleList[num]
    final_url_list.append(required_url)
    final_title_list.append(required_title)

crawl_website(urlData1,titleData1)
# crawl_website(urlData2)
# crawl_website(urlData3)

print(final_url_list)
print(final_title_list)