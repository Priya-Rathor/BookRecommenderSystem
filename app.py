from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import pickle
import numpy as np
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager
import logging
import asyncio
import traceback

# Try to import optional dependencies
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Book Recommendation API",
    description="Advanced book recommendation system with AI-powered search",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration with better defaults
SECRET_KEY = os.getenv("SECRET_KEY", "9d5f4c2a8e7f6b1c3d9e2f7a0b4d6c8f9e3a2b7d1c5f0a8e4d7b2c9f1a6e3d5")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyCuK9KFbeMxI5nzr8D8RvNpSW7cunHamig")
DATABASE_URL = os.getenv("DATABASE_URL", "user_data.db")

# Configure Gemini safely
model = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini model configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        model = None
else:
    logger.warning("Gemini API key not configured or library not available")

security = HTTPBearer(auto_error=False)

# Pydantic models with better validation
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    class Config:
        str_strip_whitespace = True

class UserLogin(BaseModel):
    username: str
    password: str
    
    class Config:
        str_strip_whitespace = True

class BookRecommendationRequest(BaseModel):
    book_title: str
    
    class Config:
        str_strip_whitespace = True

class SmartSearchRequest(BaseModel):
    query: str
    use_gemini: bool = True
    
    class Config:
        str_strip_whitespace = True

class UserPreferences(BaseModel):
    favorite_genres: List[str] = []
    favorite_authors: List[str] = []

class BookInteraction(BaseModel):
    book_title: str
    author: str = ""
    interaction_type: str  # 'view', 'like', 'rate'
    rating: Optional[int] = None
    
    class Config:
        str_strip_whitespace = True

# Database setup with better error handling
def init_database():
    """Initialize SQLite database for user management"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                book_title TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT '',
                interaction_type TEXT NOT NULL,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                favorite_genres TEXT DEFAULT '',
                favorite_authors TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_connection():
    """Database connection context manager with better error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_URL, timeout=30.0)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed"
        )
    finally:
        if conn:
            conn.close()

# Global variables for recommendation data
popular_df = None
pt = None
books = None
similarity_scores = None

# Load recommendation data safely
def load_recommendation_data():
    """Load recommendation data with comprehensive error handling"""
    global popular_df, pt, books, similarity_scores
    
    pickle_files = {
        'popular_df': 'popular.pkl',
        'pt': 'pt.pkl', 
        'books': 'books.pkl',
        'similarity_scores': 'similarity_scores.pkl'
    }
    
    loaded_data = {}
    
    for var_name, filename in pickle_files.items():
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    loaded_data[var_name] = pickle.load(f)
                logger.info(f"Loaded {filename} successfully")
            else:
                logger.warning(f"File {filename} not found")
                loaded_data[var_name] = None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            loaded_data[var_name] = None
    
    # Assign to global variables
    popular_df = loaded_data['popular_df']
    pt = loaded_data['pt']
    books = loaded_data['books']
    similarity_scores = loaded_data['similarity_scores']
    
    if all(data is not None for data in loaded_data.values()):
        logger.info("All recommendation data loaded successfully!")
    else:
        logger.warning("Some recommendation data files missing - using fallback data")

# Authentication functions with better security
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    try:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password processing failed"
        )

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token creation failed"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token with better error handling"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            return dict(user)
    except Exception as e:
        logger.error(f"User lookup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User authentication failed"
        )

# Gemini integration with proper error handling
def gemini_smart_search(query: str, book_titles: List[str]) -> List[str]:
    """Use Gemini to provide intelligent book recommendations based on query"""
    if not model or not book_titles:
        return []
    
    try:
        # Limit context size to prevent API limits
        limited_books = book_titles[:50]
        book_context = "\n".join(limited_books)
        
        prompt = f"""
        You are a book recommendation expert. Based on the user's query: "{query}"
        
        Please recommend books from this available list that best match the user's interests:
        {book_context}
        
        Return only the exact book titles from the list, one per line, up to 10 recommendations.
        Consider the user's mood, preferences, and specific requirements mentioned in their query.
        """
        
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            recommended_titles = [title.strip() for title in response.text.split('\n') if title.strip()]
            # Filter to ensure recommendations are from available books
            valid_recommendations = [title for title in recommended_titles if title in book_titles]
            return valid_recommendations[:10]
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
    
    return []

def fallback_smart_search(query: str, book_titles: List[str]) -> List[str]:
    """Fallback search using TF-IDF similarity"""
    if not SKLEARN_AVAILABLE or not book_titles:
        # Simple fallback - case insensitive substring search
        query_lower = query.lower()
        matches = [title for title in book_titles if query_lower in title.lower()]
        return matches[:10]
    
    try:
        # Create corpus with query and book titles
        corpus = [query] + book_titles
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate similarity between query and all books
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get top 10 most similar books
        top_indices = similarities.argsort()[-10:][::-1]
        return [book_titles[i] for i in top_indices if similarities[i] > 0.1]
        
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        # Ultra simple fallback
        query_lower = query.lower()
        matches = [title for title in book_titles if query_lower in title.lower()]
        return matches[:10]

# API Endpoints with comprehensive error handling

@app.on_event("startup")
async def startup_event():
    """Initialize database and load data on startup"""
    try:
        init_database()
        load_recommendation_data()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Book Recommendation API is running", 
        "version": "2.0.0",
        "gemini_available": model is not None,
        "sklearn_available": SKLEARN_AVAILABLE,
        "recommendation_data_loaded": all(x is not None for x in [popular_df, pt, books, similarity_scores])
    }

@app.post("/auth/register")
async def register(user: UserRegister):
    """Register new user"""
    # Basic validation
    if len(user.username.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", 
                         (user.username.strip(), user.email))
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            
            # Hash password and create user
            hashed_password = hash_password(user.password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (user.username.strip(), user.email, hashed_password)
            )
            conn.commit()
            
            return {"message": "User registered successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login")
async def login(user: UserLogin):
    """Login user and return JWT token"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (user.username.strip(),))
            db_user = cursor.fetchone()
            
            if not db_user or not verify_password(user.password, db_user['password_hash']):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
            
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": db_user['username']}, expires_delta=access_token_expires
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "username": db_user['username']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.get("/books/popular")
async def get_popular_books():
    """Get popular books"""
    if popular_df is None:
        # Return dummy data if pickle files not available
        return {
            "books": [
                {
                    "title": "The Great Gatsby",
                    "author": "F. Scott Fitzgerald",
                    "image_url": "https://via.placeholder.com/150x200",
                    "num_ratings": 1000,
                    "avg_rating": 4.5
                },
                {
                    "title": "To Kill a Mockingbird",
                    "author": "Harper Lee",
                    "image_url": "https://via.placeholder.com/150x200",
                    "num_ratings": 850,
                    "avg_rating": 4.2
                },
                {
                    "title": "1984",
                    "author": "George Orwell",
                    "image_url": "https://via.placeholder.com/150x200",
                    "num_ratings": 900,
                    "avg_rating": 4.6
                }
            ]
        }
    
    try:
        books_data = []
        for _, row in popular_df.iterrows():
            try:
                books_data.append({
                    "title": str(row.get('Book-Title', 'Unknown Title')),
                    "author": str(row.get('Book-Author', 'Unknown Author')),
                    "image_url": str(row.get('Image-URL-M', 'https://via.placeholder.com/150x200')),
                    "num_ratings": int(row.get('num_ratings', 0)),
                    "avg_rating": float(row.get('avg_ratings', 0.0))
                })
            except Exception as e:
                logger.warning(f"Error processing book row: {e}")
                continue
        
        return {"books": books_data}
        
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch popular books"
        )

@app.post("/books/recommend")
async def recommend_books(request: BookRecommendationRequest, current_user: dict = Depends(get_current_user)):
    """Get book recommendations based on a book title"""
    if pt is None or similarity_scores is None or books is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation system not available"
        )
    
    book_title = request.book_title.strip()
    if not book_title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Book title cannot be empty"
        )
    
    try:
        # Find the index of the book
        indices = np.where(pt.index == book_title)[0]
        if len(indices) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found in our database"
            )
        
        index = indices[0]
        similar_items = sorted(
            list(enumerate(similarity_scores[index])), 
            key=lambda x: x[1], 
            reverse=True
        )[1:6]  # Get top 5 recommendations
        
        recommendations = []
        for i in similar_items:
            try:
                title = pt.index[i[0]]
                temp_df = books[books['Book-Title'] == title]
                if not temp_df.empty:
                    book_data = temp_df.drop_duplicates('Book-Title').iloc[0]
                    recommendations.append({
                        "title": str(book_data.get('Book-Title', 'Unknown Title')),
                        "author": str(book_data.get('Book-Author', 'Unknown Author')),
                        "image_url": str(book_data.get('Image-URL-M', 'https://via.placeholder.com/150x200')),
                        "similarity_score": float(i[1])
                    })
            except Exception as e:
                logger.warning(f"Error processing recommendation: {e}")
                continue
        
        # Log user interaction
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO user_interactions (user_id, book_title, author, interaction_type) VALUES (?, ?, ?, ?)",
                    (current_user['id'], book_title, "", "recommendation_request")
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")
        
        return {"recommendations": recommendations, "based_on": book_title}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )

@app.post("/books/smart-search")
async def smart_search(request: SmartSearchRequest, current_user: dict = Depends(get_current_user)):
    """Smart book search using AI"""
    if pt is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search system not available"
        )
    
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty"
        )
    
    try:
        book_titles = list(pt.index)
        
        if request.use_gemini and model:
            recommendations = gemini_smart_search(query, book_titles)
        else:
            recommendations = fallback_smart_search(query, book_titles)
        
        # Get detailed book information
        search_results = []
        for title in recommendations:
            try:
                if books is not None:
                    temp_df = books[books['Book-Title'] == title]
                    if not temp_df.empty:
                        book_data = temp_df.drop_duplicates('Book-Title').iloc[0]
                        search_results.append({
                            "title": str(book_data.get('Book-Title', 'Unknown Title')),
                            "author": str(book_data.get('Book-Author', 'Unknown Author')),
                            "image_url": str(book_data.get('Image-URL-M', 'https://via.placeholder.com/150x200'))
                        })
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        # Log search interaction
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO user_interactions (user_id, book_title, author, interaction_type) VALUES (?, ?, ?, ?)",
                    (current_user['id'], query, "", "smart_search")
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log search interaction: {e}")
        
        return {
            "results": search_results,
            "query": query,
            "method": "gemini" if (request.use_gemini and model) else "fallback"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Smart search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

@app.get("/books/search")
async def search_books(q: str = ""):
    """Basic book search for autocomplete"""
    if not q.strip() or pt is None:
        return {"results": []}
    
    try:
        query = q.strip().lower()
        matching_books = [book for book in pt.index if query in book.lower()][:10]
        return {"results": matching_books}
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"results": []}

@app.post("/user/interaction")
async def log_interaction(interaction: BookInteraction, current_user: dict = Depends(get_current_user)):
    """Log user book interaction"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO user_interactions 
                   (user_id, book_title, author, interaction_type, rating) 
                   VALUES (?, ?, ?, ?, ?)""",
                (current_user['id'], interaction.book_title.strip(), interaction.author.strip(), 
                 interaction.interaction_type, interaction.rating)
            )
            conn.commit()
        
        return {"message": "Interaction logged successfully"}
        
    except Exception as e:
        logger.error(f"Interaction logging error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log interaction"
        )

@app.get("/user/recommendations")
async def get_personalized_recommendations(current_user: dict = Depends(get_current_user)):
    """Get personalized recommendations based on user history"""
    if pt is None or similarity_scores is None or books is None:
        # Fallback to popular books
        return await get_popular_books()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get user's interaction history
            cursor.execute(
                """SELECT book_title, author, interaction_type, rating, COUNT(*) as frequency
                   FROM user_interactions 
                   WHERE user_id = ? AND interaction_type != 'smart_search'
                   GROUP BY book_title 
                   ORDER BY frequency DESC, created_at DESC
                   LIMIT 10""",
                (current_user['id'],)
            )
            
            history = cursor.fetchall()
            
            if not history:
                # Return popular books if no history
                return await get_popular_books()
            
            # Generate recommendations based on user's most interacted books
            all_recommendations = []
            
            for interaction in history[:3]:  # Use top 3 books
                book_title = interaction['book_title']
                if book_title in pt.index:
                    try:
                        index = np.where(pt.index == book_title)[0][0]
                        similar_items = sorted(
                            list(enumerate(similarity_scores[index])), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[1:6]
                        
                        for item in similar_items:
                            title = pt.index[item[0]]
                            # Don't recommend already interacted books
                            if title not in [h['book_title'] for h in history]:
                                temp_df = books[books['Book-Title'] == title]
                                if not temp_df.empty:
                                    book_data = temp_df.drop_duplicates('Book-Title').iloc[0]
                                    all_recommendations.append({
                                        "title": str(book_data.get('Book-Title', 'Unknown Title')),
                                        "author": str(book_data.get('Book-Author', 'Unknown Author')),
                                        "image_url": str(book_data.get('Image-URL-M', 'https://via.placeholder.com/150x200')),
                                        "similarity_score": float(item[1]),
                                        "based_on": book_title
                                    })
                    except Exception as e:
                        logger.warning(f"Error processing book {book_title}: {e}")
                        continue
            
            # Remove duplicates and sort by similarity score
            seen_titles = set()
            unique_recommendations = []
            for rec in sorted(all_recommendations, key=lambda x: x['similarity_score'], reverse=True):
                if rec['title'] not in seen_titles:
                    seen_titles.add(rec['title'])
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= 10:
                        break
            
            if not unique_recommendations:
                return await get_popular_books()
            
            return {
                "recommendations": unique_recommendations,
                "based_on_history": [h['book_title'] for h in history[:3]]
            }
            
    except Exception as e:
        logger.error(f"Personalized recommendation error: {e}")
        logger.error(traceback.format_exc())
        # Fallback to popular books
        return await get_popular_books()

@app.get("/user/history")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """Get user's interaction history"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT book_title, author, interaction_type, rating, created_at
                   FROM user_interactions 
                   WHERE user_id = ? 
                   ORDER BY created_at DESC
                   LIMIT 50""",
                (current_user['id'],)
            )
            
            history = [dict(row) for row in cursor.fetchall()]
            return {"history": history}
            
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user history"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)