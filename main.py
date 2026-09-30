import os
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.joblib")
INDICES_PATH = os.path.join(BASE_DIR, "indices.joblib")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.joblib")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    Normalized into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.joblib must be a dict or pandas Series-like (with .items())"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all joblib files once when the API starts."""
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    for path in (DF_PATH, INDICES_PATH, TFIDF_MATRIX_PATH, TFIDF_PATH):
        if not os.path.exists(path):
            raise RuntimeError(f"Required file not found: {path}")

    df = joblib.load(DF_PATH)
    indices_obj = joblib.load(INDICES_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    tfidf_obj = joblib.load(TFIDF_PATH)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.joblib must contain a DataFrame with a 'title' column")

    yield



app = FastAPI(title="Movie Recommender API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """Raw TMDB response for keyword search (multiple results)."""
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch a poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from the local df using cosine similarity
    on the TF-IDF matrix.
    """
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns the raw TMDB shape with a 'results' list.
    Streamlit uses it for dropdown suggestions and the results grid.
    """
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover popular movies in that genre
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    For a selected movie, returns:
      - movie details
      - TF-IDF recommendations (local) + posters
      - genre recommendations (TMDB) + posters

    It picks the BEST TMDB match for the query.
    For multiple matches, use /tmdb/search.
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
