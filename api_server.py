"""
FastAPI Backend for Bitcoin Tax Optimizer SaaS
Main API server with authentication, portfolio management, and tax calculation endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional
from datetime import datetime, date
from decimal import Decimal
import jwt
import os
from enum import Enum

# Import your existing optimizer
from bitcoin_tax_optimizer import BitcoinTaxOptimizer, TaxMethod, Transaction

# Initialize FastAPI app
app = FastAPI(
    title="Bitcoin Tax Optimizer API",
    description="API for optimizing Bitcoin tax reporting",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

# Database would go here - using in-memory for demo
# In production: Use SQLAlchemy + PostgreSQL
USERS_DB = {}
PORTFOLIOS_DB = {}
TRANSACTIONS_DB = {}

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    subscription_tier: str
    created_at: datetime

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str]
    total_btc: Decimal
    total_cost_basis: Decimal
    transaction_count: int
    created_at: datetime

class TransactionCreate(BaseModel):
    tx_date: date
    tx_type: str
    amount_btc: Decimal
    price_usd: Decimal
    fee_usd: Decimal = Decimal('0')
    tx_id: Optional[str] = None
    exchange: Optional[str] = None
    notes: Optional[str] = None
    
    @validator('tx_type')
    def validate_tx_type(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('tx_type must be "buy" or "sell"')
        return v.lower()
    
    @validator('amount_btc', 'price_usd', 'fee_usd')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return v

class TransactionResponse(BaseModel):
    id: str
    portfolio_id: str
    tx_date: date
    tx_type: str
    amount_btc: Decimal
    price_usd: Decimal
    fee_usd: Decimal
    tx_id: Optional[str]
    exchange: Optional[str]
    notes: Optional[str]
    created_at: datetime

class TaxMethodEnum(str, Enum):
    FIFO = "FIFO"
    LIFO = "LIFO"
    HIFO = "HIFO"

class TaxReportRequest(BaseModel):
    portfolio_id: str
    tax_year: int
    method: TaxMethodEnum

class TaxReportSummary(BaseModel):
    total_proceeds: Decimal
    total_cost_basis: Decimal
    total_gain_loss: Decimal
    short_term_gain_loss: Decimal
    long_term_gain_loss: Decimal
    remaining_btc: Decimal
    remaining_cost_basis: Decimal

class TaxReportResponse(BaseModel):
    id: str
    portfolio_id: str
    tax_year: int
    method: str
    summary: TaxReportSummary
    generated_at: datetime

# Helper Functions
def create_token(user_id: str) -> str:
    """Create JWT token"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow().timestamp() + 86400 * 30  # 30 days
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return user_id"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def check_subscription_limits(user_id: str, transaction_count: int):
    """Check if user is within subscription limits"""
    user = USERS_DB.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    tier = user.get("subscription_tier", "free")
    
    limits = {
        "free": 25,
        "pro": 500,
        "premium": 999999,
        "enterprise": 999999
    }
    
    if transaction_count > limits.get(tier, 25):
        raise HTTPException(
            status_code=403,
            detail=f"Transaction limit exceeded for {tier} tier. Please upgrade."
        )

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Bitcoin Tax Optimizer API",
        "version": "1.0.0"
    }

@app.post("/auth/register")
async def register(user: UserCreate):
    """Register a new user"""
    if user.email in [u["email"] for u in USERS_DB.values()]:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = f"user_{len(USERS_DB) + 1}"
    USERS_DB[user_id] = {
        "id": user_id,
        "email": user.email,
        "password": user.password,  # In production: hash with bcrypt
        "name": user.name,
        "subscription_tier": "free",
        "created_at": datetime.utcnow()
    }
    
    token = create_token(user_id)
    return {
        "token": token,
        "user": UserResponse(**USERS_DB[user_id])
    }

@app.post("/auth/login")
async def login(credentials: UserLogin):
    """Login user and return JWT token"""
    user = next(
        (u for u in USERS_DB.values() if u["email"] == credentials.email),
        None
    )
    
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"])
    return {
        "token": token,
        "user": UserResponse(**user)
    }

@app.get("/auth/me")
async def get_current_user(user_id: str = Depends(verify_token)):
    """Get current user info"""
    user = USERS_DB.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user)

@app.post("/portfolios", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio: PortfolioCreate,
    user_id: str = Depends(verify_token)
):
    """Create a new portfolio"""
    portfolio_id = f"portfolio_{len(PORTFOLIOS_DB) + 1}"
    PORTFOLIOS_DB[portfolio_id] = {
        "id": portfolio_id,
        "user_id": user_id,
        "name": portfolio.name,
        "description": portfolio.description,
        "total_btc": Decimal('0'),
        "total_cost_basis": Decimal('0'),
        "transaction_count": 0,
        "created_at": datetime.utcnow()
    }
    return PortfolioResponse(**PORTFOLIOS_DB[portfolio_id])

@app.get("/portfolios", response_model=List[PortfolioResponse])
async def get_portfolios(user_id: str = Depends(verify_token)):
    """Get all portfolios for current user"""
    user_portfolios = [
        p for p in PORTFOLIOS_DB.values() 
        if p["user_id"] == user_id
    ]
    return [PortfolioResponse(**p) for p in user_portfolios]

@app.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: str,
    user_id: str = Depends(verify_token)
):
    """Get specific portfolio"""
    portfolio = PORTFOLIOS_DB.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return PortfolioResponse(**portfolio)

@app.post("/portfolios/{portfolio_id}/transactions", response_model=TransactionResponse)
async def create_transaction(
    portfolio_id: str,
    transaction: TransactionCreate,
    user_id: str = Depends(verify_token)
):
    """Add a transaction to a portfolio"""
    portfolio = PORTFOLIOS_DB.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Check subscription limits
    current_count = portfolio["transaction_count"]
    check_subscription_limits(user_id, current_count + 1)
    
    transaction_id = f"tx_{len(TRANSACTIONS_DB) + 1}"
    tx_data = {
        "id": transaction_id,
        "portfolio_id": portfolio_id,
        **transaction.dict(),
        "created_at": datetime.utcnow()
    }
    TRANSACTIONS_DB[transaction_id] = tx_data
    
    # Update portfolio stats
    portfolio["transaction_count"] += 1
    if transaction.tx_type == "buy":
        portfolio["total_btc"] += transaction.amount_btc
        portfolio["total_cost_basis"] += (transaction.amount_btc * transaction.price_usd)
    
    return TransactionResponse(**tx_data)

@app.get("/portfolios/{portfolio_id}/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    portfolio_id: str,
    user_id: str = Depends(verify_token)
):
    """Get all transactions for a portfolio"""
    portfolio = PORTFOLIOS_DB.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    portfolio_transactions = [
        tx for tx in TRANSACTIONS_DB.values()
        if tx["portfolio_id"] == portfolio_id
    ]
    return [TransactionResponse(**tx) for tx in portfolio_transactions]

@app.post("/reports/generate", response_model=TaxReportResponse)
async def generate_tax_report(
    request: TaxReportRequest,
    user_id: str = Depends(verify_token)
):
    """Generate tax report for a portfolio"""
    portfolio = PORTFOLIOS_DB.get(request.portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if portfolio["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get transactions for this portfolio
    portfolio_transactions = [
        tx for tx in TRANSACTIONS_DB.values()
        if tx["portfolio_id"] == request.portfolio_id
    ]
    
    if not portfolio_transactions:
        raise HTTPException(status_code=400, detail="No transactions found")
    
    # Use your existing optimizer
    optimizer = BitcoinTaxOptimizer()
    
    # Convert to Transaction objects
    for tx in portfolio_transactions:
        optimizer.transactions.append(Transaction(
            date=datetime.combine(tx["tx_date"], datetime.min.time()),
            tx_type=tx["tx_type"],
            amount_btc=tx["amount_btc"],
            price_usd=tx["price_usd"],
            fee_usd=tx["fee_usd"],
            tx_id=tx.get("tx_id", "")
        ))
    
    # Generate report
    method_map = {
        "FIFO": TaxMethod.FIFO,
        "LIFO": TaxMethod.LIFO,
        "HIFO": TaxMethod.HIFO
    }
    
    report = optimizer.generate_tax_report(method_map[request.method])
    
    report_id = f"report_{len(PORTFOLIOS_DB) + 1}"
    
    return TaxReportResponse(
        id=report_id,
        portfolio_id=request.portfolio_id,
        tax_year=request.tax_year,
        method=request.method,
        summary=TaxReportSummary(**report["summary"]),
        generated_at=datetime.utcnow()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
