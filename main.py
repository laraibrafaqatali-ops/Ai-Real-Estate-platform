import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.recommender import PropertyRecommender
from src.ai_assistant import AIAssistant
from src.rag_engine import PropertyRAGEngine

app = FastAPI(title="AI Real Estate Intelligence Platform")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize modules
model = joblib.load('models/property_price_model.pkl')
recommender = PropertyRecommender()
ai_bot = AIAssistant()
rag_engine = PropertyRAGEngine()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="predict.html"
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    GrLivArea: float = Form(...),
    OverallQual: int = Form(...),
    YearBuilt: int = Form(...),
    TotalBsmtSF: float = Form(...),
    FullBath: int = Form(...),
    BedroomAbvGr: int = Form(...),
    GarageCars: int = Form(...),
    Neighborhood: str = Form(...)
):
    input_data = pd.DataFrame([{
        'GrLivArea': GrLivArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'TotalBsmtSF': TotalBsmtSF,
        'FullBath': FullBath,
        'BedroomAbvGr': BedroomAbvGr,
        'GarageCars': GarageCars,
        'Neighborhood': Neighborhood
    }])
    
    predicted_price = model.predict(input_data)[0]
    return templates.TemplateResponse(
        request=request, 
        name="predict.html", 
        context={"prediction": f"${predicted_price:,.2f}"}
    )

@app.get("/recommend", response_class=HTMLResponse)
async def recommend_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="recommend.html"
    )

@app.post("/recommend", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    max_budget: float = Form(...),
    neighborhood: str = Form("All"),
    min_bedrooms: int = Form(1)
):
    results = recommender.recommend(max_budget=max_budget, location=neighborhood, min_bedrooms=min_bedrooms)
    return templates.TemplateResponse(
        request=request, 
        name="recommend.html", 
        context={"recommendations": results}
    )

@app.get("/ai-chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="ai_chat.html"
    )

@app.post("/ai-chat", response_class=HTMLResponse)
async def chat_response(request: Request, query: str = Form(...)):
    answer = ai_bot.ask(query)
    return templates.TemplateResponse(
        request=request, 
        name="ai_chat.html", 
        context={"query": query, "answer": answer}
    )

@app.get("/rag-docs", response_class=HTMLResponse)
async def rag_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="rag.html"
    )

@app.post("/rag-docs", response_class=HTMLResponse)
async def rag_response(request: Request, query: str = Form(...)):
    answer = rag_engine.query_docs(query)
    return templates.TemplateResponse(
        request=request, 
        name="rag.html", 
        context={"query": query, "answer": answer}
    )