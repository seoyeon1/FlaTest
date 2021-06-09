from flask import Flask, request, redirect, url_for, render_template
from werkzeug import datastructures
from werkzeug.utils import redirect
# from werkzeug.wrappers import 

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def base():
	#user = request.args.get("original")
	#return redirect(url_for("result", original = user))
	return render_template('base.html')#render로 다른 템플릿에 접근

@app.route('/result', methods=['GET'])
def result():
	if request.method == 'GET':#지금은 get인데 시간 나면 post로 바꿀 예정

		origin = request.args.get("original") #사용자가 입력한 값을 user에 저장
		summary = request.args.get("original") + '요약결과가 담길 부분이야' #요약 결과를 담을 변수

	#return redirect(url_for("result", original = user))
	return render_template('result.html', origins = origin, summ= summary)

#지금은 /result 루트를 통해 원문을 받았고 result페이지에 원문, 요약문 넣어주는 부분에 넣었거든.
#최종 목표는 /processing이라는 루트를 세로 만들고 거기에 전처리 코드랑 저장한 모델을 넣어서 값을 처리하고 그 결과를 /result 경로로 보내주면 어떨까
#아니면 /result에 모델코드 다 집어 넣어서 다 여기서 처리하게 하는 거지 
# 
#  


#flask url을 통한 값 전달 공부용 코드
@app.route('/method', methods=['GET', 'POST']) 
def method(): 
	if request.method == 'GET': 
		
		name = request.args.get("original") 
		return "GET으로 전달된 데이터({})".format(name) 
	else: 
		 
		name = request.form["original"] 
		return "POST로 전달된 데이터({})".format(name)



if __name__ == '__main__':
	app.run(debug=True)