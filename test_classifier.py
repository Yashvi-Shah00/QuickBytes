import pickle
from pickle import load

article=["A lot is happening in the world of entertainment today. From the highly-anticipated Gehraiyaan trailer release starring Deepika Padukone, Siddhant Chaturvedi, Dhairya Karwa and Ananya Panday to the new posters reveal of Robert Pattinson’s The Batman, there is a number of things to unpack. The Gehraiyaan trailer caught a number of eyeballs, as celebrities rushed to speak highly of the first full promo of the Amazon Prime movie. The Shakun Batra directorial is being pitched as a mature relationship drama. It is being co-produced by Karan Johar’s Dharma Productions. The official synopsis of Gehraiyaan reads, “Directed by the very talented Shakun Batra, the much-awaited movie looks beneath the surface of complex modern relationships, adulting, letting go and taking control of ones’ life path.” As far as Hollywood is concerned, the makers released two brand new posters of the upcoming The Batman movie. While one featured the cowled face of the Caped Crusader, the other starred Pattinson and Zoe Kravtiz as Batman and Catwoman, respectively. Meanwhile, in the world of Indian television, there was some unfortunate news as TV star Shaheer Sheikh’s father passed away due to Covid-19 complications. Shaheer had earlier taken to social media to share that his father is suffering from the disease and is very critical."]
Tfidf_from_pickle = load(open('models/tfidf_model.pkl', 'rb'))
SVM_from_pickle = load(open('models/svm_model.pkl', 'rb'))

tfid_test_features1 = Tfidf_from_pickle.transform(article)
category_output = SVM_from_pickle.predict(tfid_test_features1)

print("Output",category_output)