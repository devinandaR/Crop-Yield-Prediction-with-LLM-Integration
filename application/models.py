from .database import db

class Users(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.Integer , autoincrement = True , primary_key = True)
    username = db.Column(db.String , unique = True, nullable = False)
    password = db.Column(db.String , nullable = False)
    name = db.Column(db.String , nullable = False)
    relationship1 = db.relationship("Lists")

class Lists(db.Model):
    __tablename__ = 'lists'
    list_id = db.Column(db.Integer ,autoincrement=True, primary_key=True,unique=True)
    l_user_id=db.Column(db.Integer , db.ForeignKey("users.user_id"))
    title= db.Column(db.String , unique = True, nullable = False)
    N=db.Column(db.Integer)
    P=db.Column(db.Integer)
    K=db.Column(db.Integer)
    PH=db.Column(db.Integer)
    temp=db.Column(db.Integer)
    Humdity=db.Column(db.Integer)
    RF=db.Column(db.Integer)
    WL=db.Column(db.Integer)
    crops=db.Column(db.String )
    suggestion=db.Column(db.String)
    
    
