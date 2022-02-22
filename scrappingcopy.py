import newspaper
cnn_paper = newspaper.build('http://cnn.com')
urlData = []
for article in cnn_paper.articles:
  urlData.push(article)
  # print(article.url)
print(urlData)