from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
from utils.tf_mcc import MCC

path_to_model = "model/fake_news_classifier"
model = tf.keras.models.load_model(path_to_model)

app = FastAPI()

class TestRequest(BaseModel):

	a : str
	b : int

class FakeNewsResponse(BaseModel):

	probability : float
	is_fake : bool


@app.get("/news_classifier", response_model = FakeNewsResponse)
async def root(title : str):

	#run model:
	#prob = model([title])[0]
	prob = model(np.array([title])).numpy().flatten()[0]

	prob_boolean = prob < 0.5

	return {"probability": float(prob), "is_fake" : prob_boolean}

@app.post("/test_post")
async def test_post(request : TestRequest):

	a = request.a
	b = request.b

	return {"message": f"a was: {a} and one plus b = {1+b}"}


#'{"a" : "some stuff", "b" : 23}'

#curl -X POST -H "content-type:application/json" "http://0.0.0.0:8000/test_post" -d '{"a" : "some stuff", "b" : 23}'