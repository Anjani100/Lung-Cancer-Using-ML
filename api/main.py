from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}



# To run server
# uvicorn main:app --reload
# And go to http://127.0.0.1:8000/

# https://pypi.org/project/fastapi/
# https://pypi.org/project/unsplash/