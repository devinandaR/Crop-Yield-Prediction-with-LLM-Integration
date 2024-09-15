import os
from flask import Flask
from application import config
from application.config import LocalDevelopmentConfig
from application.database import db
# from flask_restful import Resource ,Api


app = None
api=None



app = Flask(__name__, template_folder="templates")
# db.create_all()
app.config.from_object(LocalDevelopmentConfig)
db.init_app(app)

app.app_context().push()



# Import all the controllers so they are loaded
from application.controllers import *
#Add all restful controllers)



db.create_all()
if __name__ == '__main__':
  # Run the Flask app
  app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
  app.run(host="0.0.0.0",debug=True,port=5000,threaded=True)
