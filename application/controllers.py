
from operator import truediv
from sqlalchemy import select
from flask import Flask, flash, redirect, render_template, \
    request,url_for,Response
from flask import current_app as app
from  flask import make_response,json,request,jsonify,send_file,Response
from application.models import Lists,Users
from .database import db
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltt
import io,os
import csv
from werkzeug.utils import secure_filename
import pandas as pd
from application.script import ALLOWED_EXTENSIONS ,allowed_file,Valid
import pickle
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image 
import shutil,io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet


os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0)

def get_full_answer(pages,query):
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer=chain.run(input_documents=pages, question=query)
    return answer









engine = create_engine("sqlite:///db_directory/kb.sqlite3", echo=True, future=True)

@app.route('/', methods = ["GET","POST"] )
def login():
    if request.method == "GET":
        return render_template("login.html")
    else:
        user_name = request.form["username"]
        
        pass_word = request.form["password"]
        
        if Users.query.filter_by(username = user_name).first() == None:
            flash(' INCORRECT USERNAME ')
            return redirect("/")
        elif Users.query.filter_by(username = user_name,password = pass_word).first() == None:
            flash(' INCORRECT PASSWORD ')
            return redirect("/")
        elif Users.query.filter_by(username = user_name,password = pass_word).first() != None:
            user= Users.query.filter_by(username=user_name).first()
            user_id=user.user_id
            link="/" + str(user_id) +"/board"
            return redirect(link)

@app.route('/register', methods = ["GET","POST"] )
def register():
    if request.method == "GET":
        return render_template("register.html")
    else:
        user_name = request.form["username"]
        pass_word = request.form["password"]
        NAME = request.form["name"]
        
        print(user_name, pass_word)
        user= Users.query.filter_by(username=user_name).first()
        if user != None:
            # the query has returned a user
            flash('Username already exist please use difffernt username')
            return render_template("register.html")

        u1 = Users(username =user_name,password = pass_word,name = NAME)
        db.session.add(u1)
        db.session.commit()
        
        return redirect("/")

@app.route("/<user_id>/board", methods=["GET", "POST"])
def board(user_id):
    user=Users.query.filter_by(user_id=user_id).first()
    b_name=user.name
    lists=Lists.query.filter_by(l_user_id=user_id).all()
    #print(lists)
    length=len(lists)
    return render_template("board.html",lists=lists,user_id=user_id,b_name=b_name,length=length)

@app.route("/<user_id>/list/<list_id>/predict", methods=["GET"])
def predict(user_id,list_id):
    list1= Lists.query.filter_by(l_user_id=user_id,list_id = list_id).first()
    list_id=list1.list_id
    list = [ list1.N,list1.P,list1.K,
            list1.temp, list1.Humdity,list1.PH, list1.RF,list1.WL]
    # print(type(N))

    
    X_test = np.array(list).reshape(1, -1)
 
    # Load the model from the file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Use the loaded model
    predictions = model.predict(X_test)
    probables = model.predict_proba(X_test)
    # print(probables[0][probables[0].argsort()[-1]])
    proba1=probables[0][probables[0].argsort()[-1]]
    proba2=probables[0][probables[0].argsort()[-2]]
    proba3=probables[0][probables[0].argsort()[-3]]
    # print(probables[0].argsort()[-1])
    target_names=np.array(["apple","banana","blackgram",
                           "chickpea","coconut","coffee","cotton",
                           "grapes","jute","kidneybeans","lentil",
                           "maize","mango","mothbeans","mungbean",
                           "muskmelon","orange","papaya","pigeonpeas",
                           "pomegranate","rice","watermelon"])
    next_best_prediction1 = target_names[probables[0].argsort()[-2]]
    next_best_prediction2 = target_names[probables[0].argsort()[-3]]
    # print(next_best_prediction1)
    # prediction2 = model.predict(X_test2)
    predictions=np.append(predictions,next_best_prediction1)
    predictions=np.append(predictions,next_best_prediction2)
    # print(predictions)
    data = {predictions[0]: proba1,
        predictions[1]: proba2,
        predictions[2]: proba3}
    labels = [predictions[0],predictions[1],predictions[2]]
    probabilities = [proba1,proba2,proba3]
    if (list1.crops == None) :
        crops=str(predictions[0])+","+str(predictions[1])+","+str(predictions[2])
        list1.crops=crops
        db.session.commit()
    

    # Plotting the bar chart
    plt.bar(labels, probabilities,color='orange')

    # Labeling the chart
    plt.xlabel('Predictions')
    plt.ylabel('Probabilities')
    plt.title('Prediction Probabilities')

    # Displaying the chart
    # plt.show()
    fig_location="proba_bar_plot_"+str(list_id)+".png"
    plt.savefig("static/"+fig_location)
    bar_path=url_for('static',filename=fig_location)
    str1="Nitrogen="+str(list1.N)+","
    str2="Phosphorous="+str(list1.P)+","
    str3="Potassium="+str(list1.K)+","
    str4="Temprature="+str(list1.temp)+","
    str5="Ph of Water="+str(list1.PH)+","
    str6="Rainfall="+str(list1.RF)+","
    str7="WaterLevel="+str(list1.WL)+","
    str8="Humidity="+str(list1.Humdity)
    
    query='''.give best cultivation 
    practices according to data and also suggest what kind of fertilizers and pestcides should be used .'''
    pages=["These are soil parameters",str1,str2,str3,str4,str5,str6,str7,str8,
           "best crop predicted is ",predictions[0],query]
    final_query=' '.join(pages)
    # print(final_query)
    # answer=llm(final_query)
    # print(list1.suggestion)
    
    if (list1.suggestion == None) :
        answer=llm(final_query)
        list1.suggestion=answer
        db.session.commit()
    else:
        answer=list1.suggestion

    
    return render_template('Display.html',prediction_text=predictions,
                           user_id=user_id,list_id=list_id,bar_path=bar_path,answer=answer)

@app.route("/<user_id>/list/create", methods=["GET", "POST"])
def create_list(user_id):
    if request.method == "GET":
        lists=Lists.query.filter_by(l_user_id=user_id).all()
        # if len(lists)<5:
        return render_template("list_form.html",user_id=user_id)
        # else:
        #     error="No More lists"
        #     return render_template("update_error.html",error=error,user_id=user_id)
    else :
        title=request.form["title"]
        N=request.form["nitrogen"]
        P=request.form["phosphorus"]
        K=request.form["pottasium"]
        temperature=request.form["temperature"]
        humidity=request.form["humidity"]
        ph=request.form["ph"]
        rainfall=request.form["rainfall"]
        water_level=request.form["Water Level"]
        l1=Lists(l_user_id=user_id,title=title,
                 N=N,P=P,K=K,temp=temperature,
                 Humdity=humidity,WL=water_level,RF=rainfall,PH=ph)
        db.session.add(l1)  
        db.session.commit()
        link="/"+str(user_id)+"/board"
        return redirect(link)


@app.route("/<user_id>/list/<list_id>/update", methods=["GET", "POST"])
def update_list(user_id,list_id):
    list1= Lists.query.filter_by(l_user_id=user_id,list_id = list_id).first()
    id_list=list_id
    if request.method == "GET":
        l_title=list1.title
        return render_template("list_update_form.html",list = list1,l_title=l_title,user_id=user_id)
    else:
        title=request.form["title"]
        N=request.form["nitrogen"]
        P=request.form["phosphorus"]
        K=request.form["pottasium"]
        temperature=request.form["temperature"]
        humidity=request.form["humidity"]
        ph=request.form["ph"]
        rainfall=request.form["rainfall"]
        water_level=request.form["Water Level"]
        list1.title=title
        list1.N=N
        list1.P=P
        list1.K=K
        list1.temp=temperature
        list1.PH=ph
        list1.Humdity=humidity
        list1.RF=rainfall
        list1.WL=water_level
        db.session.commit()
        if (list1.crops != None) :
            list1.crops=None
            db.session.commit()
        if (list1.suggestion != None) :
            list1.suggestion=None
            db.session.commit()    
        link="/"+str(user_id)+"/board"
        return redirect(link)


@app.route("/<user_id>/list/<list_id>/download", methods=["GET", "POST"])
def export_list(user_id,list_id):
    list1= Lists.query.filter_by(list_id = list_id).first()
    pdf_path="static/"+str(list_id)+"_prediction.pdf"
    pdf_file = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []   
    elements.append(Paragraph( "CROP PREDICTION AND SUGGESTIONS REPORT ", styles["Heading1"]))
    elements.append(Paragraph( list1.title, styles["Heading1"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Nitrogen: " + str(list1.N), styles["Normal"]))
    elements.append(Paragraph("Phosphorous: " + str(list1.P), styles["Normal"]))
    elements.append(Paragraph("Potassium: " + str(list1.K), styles["Normal"]))
    elements.append(Paragraph("PH: " + str(list1.PH), styles["Normal"]))
    elements.append(Paragraph("Temprature: " + str(list1.temp), styles["Normal"]))
    elements.append(Paragraph("Humidity: " + str(list1.Humdity), styles["Normal"]))
    elements.append(Paragraph("RainFall: " + str(list1.RF), styles["Normal"]))
    elements.append(Paragraph("WaterLevel: " + str(list1.WL), styles["Normal"]))
    elements.append(Paragraph("Predictions for above parameters are " + list1.crops, styles["Normal"]))
    fig_location="static/proba_bar_plot_"+str(list_id)+".png"
    img = Image(fig_location)
    aspect_ratio = img.drawWidth / float(img.drawHeight)
    img.drawHeight = 200
    img.drawWidth = 200 * aspect_ratio
    elements.append(img)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Best Practices for maximum production according to parameters ." ,styles["Normal"]))
    elements.append(Paragraph(list1.suggestion, styles["Normal"]))
    elements.append(Spacer(1, 40))
    pdf_file.build(elements)
    return send_file(pdf_path, mimetype='application/pdf')



    



@app.route("/<user_id>/list/<list_id>/delete", methods=["GET", "POST"])
def delete_list(user_id,list_id):
    if request.method == "POST":
        list2= Lists.query.filter_by(l_user_id=user_id,list_id = list_id).first()  
        #print(list2)
        db.session.delete(list2)
        db.session.commit()
        link="/"+str(user_id)+"/board"
        return redirect(link)
    elif request.method=="GET":
        #print(user_id)
        list1= Lists.query.filter_by(l_user_id=user_id,list_id = list_id).first()
        error="Warning!!!!!!!!!!!!!!!!!!!!!  Deleting form will delete all details in them . Proceed "
        return render_template("warning_delete.html",error=error,list_id=list_id,obj_name=list1.title,object="list",user_id=user_id)






