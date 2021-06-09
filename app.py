from flask import Flask, request, redirect, url_for, render_template
from tensorflow.python.ops.variables import global_variables_initializer

from werkzeug import datastructures #werkzeug라이브러리로 또 import해야하나?
from werkzeug.utils import redirect

# 모델에 필요한 모듈들을 import

import numpy as np
import pandas as pd
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
np.random.seed(seed=0)



#지금은 /result 루트를 통해 원문을 받았고 result페이지에 원문, 요약문 넣어주는 부분에 넣었거든.
#/result에 모델코드 다 집어 넣어서 다 여기서 처리하도록 
# base.html(POST-원문)-> /result함수에서 processing(원문)-modeling-predict(요약문)-> ->result.html에 함께 띄우기
#
# 총 2개(인코딩, 디코딩) 모델이 있는데 그러면 각각 model1, 2에다가 sess1, 2 그리고 save파일도 각각 필요한 건가
# 아니면 py코드에 있는 Model부분을 다 복붙하면 되려나?


app = Flask(__name__)#static_url_path='/static'이 필요하나?

#사전에 학습된 모델(model폴더에 있음)을 아래에 불러옴 

#사용할 변수, hypothesis를 선언?


saver = tf.train.Saver()
model = global_variables_initializer()

sess = tf.Session() #세션 객체 생성, 사용자 요청이 들어올 때마다 모델을 학습시킬수 있게 함
sess.run(model)

#저정된 모델이 세션에 적용될 수 있게 모델 저장할 곳(save_path) 지정	
save_path = 'model\review_summaization_new.h5'
#방금 돌린 모델(세션)을 페스에 restore	
saver.restore(sess, save_path)



@app.route('/')
def base():
	
	return render_template('base.html')#render로 base 템플릿을 띄워줌



@app.route('/result', methods=['POST'])#form으로 받은 원문이 전달되는 주소.(여기서 모델을 실행시켜야..) 
def result():
	if request.method == 'POST':
		
		text_max_len = 60 #변수 선언 여기가 맞나?
		summary_max_len = 6

		input_stc =  request.form["original"] #사용자가 입력한 값을 전달받아 input_stc에 저장
		
		#입력 데이터 전처리파트를 아래에 작성
		token_stc = input_stc.split()
		encode_stc = tokenizer_from_json.texts_to_sequences([token_stc])
		pad_stc = pad_sequences(encode_stc, maxlen=text_max_len, padding="POST")


################################
#save한 모델은 1개인데 인코더 디코더 모델을 어떻게 선언(?)하지?

		#인코딩 부분
		#불러온 모델로 이부분(encoder_model)을 대체해야할까?
		en_out, en_hidden, en_cell = encoder_model.predict(pad_stc)

		predicted_seq = np.zeros((1,1))
		predicted_seq[0, 0] = su_to_index['sostoken']


		#디코딩 부분
		decoded_stc = []

		while True:
			output_words, h, c = decoder_model.predict([predicted_seq, en_out, en_hidden, en_cell])#디코딩용 모델 불러와서 예측

			predicted_word = index_to_su[np.argmax(output_words[0,0])]

			if predicted_word == 'eostoken':
				break

			decoded_stc.append(predicted_word)

			predicted_seq = np.zeros((1,1))
			predicted_seq[0, 0] = np.argmax(output_words[0, 0])

			en_hidden = h
			en_cell = c

		print('\n')
		summary = ' '.join(decoded_stc)#생성된 요약문을 담는 변수

		model.save('/content/gdrive/My Drive/capstone_design_yoon/')

		

		
		#위에 전처리한 거(feed_dict같은 부분에 넣어주면 될듯) 가지고 sess.run하는 거지
		
		

		


#load the model

	#예상 시나리오

	#저장한? 모델 불러오기(이게 처음 맞나?)
	#(전달 받은) 원문을 전처리

	#모델에 전처리가 끝난 원문을 넣고 예측돌려(아무거나 = model.predict(전처리된원문))
	#예측한 걸 디코더가 처리(summary = decode_predict(아무거나))


	return render_template('result.html', origins = input_stc, summ = summary)#render할 때 원문, 요약문도 함께 전달하기









if __name__ == '__main__':
	app.run(debug=True)#포트 설정은 다음에