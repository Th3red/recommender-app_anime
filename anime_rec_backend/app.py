import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from thefuzz import process  # Switched from fuzzywuzzy to thefuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hnswlib
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from flask_cors import CORS
import logging
import pickle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import signal
import sys
from dotenv import load_dotenv  # Import dotenv

from profiler import timeit_decorator, memory_profiler_decorator


app = Flask(__name__)
CORS(app)
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graceful shutdown handling
def graceful_shutdown(signum, frame):
    logger.info("Shutting down gracefully...")
    # Perform cleanup tasks if necessary
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Environment Variables
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("No SECRET_KEY set for Flask application")
app.config['SECRET_KEY'] = SECRET_KEY

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set for Flask application")

# SQLAlchemy Engine with Connection Pooling
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,      # Checks connections for liveness
        pool_size=10,            # Initial number of connections
        max_overflow=20          # Additional connections allowed beyond pool_size
    )
    connection = engine.connect()
    logger.info("Database connection established.")
except SQLAlchemyError as e:
    logger.exception("Database connection failed.")
    raise e

# Globals for ANN (user-user) approach
user_index = None          # hnswlib index object
#user_vectors = None        # scipy.sparse csr_matrix of shape (num_users, num_anime)
user_index_list = []       # list of user_ids in row order (Python ints)
anime_id_to_col = {}       # mapping from anime_id -> column index in user_vectors (Python ints)
INDEX_PATH = 'hnsw_index.bin'
ANIME_MAPPING_PATH = 'anime_id_to_col.pkl'
USER_INDEX_LIST_PATH = 'user_index_list.pkl'

# 1) Build user-anime matrix from DB using sparse matrix
@timeit_decorator
@memory_profiler_decorator
def build_user_anime_matrix_sparse():
    """
    Creates a user x anime matrix (sparse) using CSR format.
    Returns: (user_vectors_sparse, user_index_list, anime_id_to_col)
      - user_vectors_sparse: scipy.sparse.csr_matrix of shape (num_users, num_anime)
      - user_index_list: list of user_ids in row order (ints)
      - anime_id_to_col: dict mapping anime_id -> column index (ints)
    """
    logger.info("Fetching ratings from database...")
    try:
        with engine.connect() as conn:
            ratings_df = pd.read_sql("SELECT user_id, anime_id, rating FROM ratings", conn)
    except SQLAlchemyError as e:
        logger.exception("Failed to fetch ratings from database.")
        raise e

    # Replace -1 with 0 to indicate no rating
    ratings_df['rating'] = ratings_df['rating'].replace(-1, 0)

    # Ensure columns are Python int, not numpy.int64
    ratings_df['user_id'] = ratings_df['user_id'].astype(int)
    ratings_df['anime_id'] = ratings_df['anime_id'].astype(int)

    unique_users = ratings_df['user_id'].unique()
    unique_anime = ratings_df['anime_id'].unique()

    # Sort and build lists of Python ints
    user_index_list_local = sorted(unique_users.tolist()) 
    anime_id_list = sorted(unique_anime.tolist())

    # Maps: user_id -> row index, anime_id -> col index
    user_id_to_row = {uid: i for i, uid in enumerate(user_index_list_local)}
    anime_id_to_col_local = {aid: j for j, aid in enumerate(anime_id_list)}

    num_users = len(user_index_list_local)
    num_anime = len(anime_id_list)

    logger.info(f"Number of users: {num_users}, Number of anime: {num_anime}")

    # Create CSR (Compressed Sparse Row) matrix
    logger.info("Building sparse user-anime matrix...")
    user_vectors_sparse = csr_matrix(
        (
            ratings_df['rating'].values,
            (
                ratings_df['user_id'].map(user_id_to_row).values,
                ratings_df['anime_id'].map(anime_id_to_col_local).values
            )
        ),
        shape=(num_users, num_anime),
        dtype=np.float32
    )

    logger.info("Sparse user-anime matrix built successfully.")

    return user_vectors_sparse, user_index_list_local, anime_id_to_col_local


@timeit_decorator
@memory_profiler_decorator
def build_and_save_dense_matrix():
    # Build sparse matrix to manage memory
    sparse_matrix, user_list, anime_mapping = build_user_anime_matrix_sparse()
    # Convert to dense
    dense_matrix = sparse_matrix.toarray()
    # Save dense matrix to disk
    np.save('dense_matrix.npy', dense_matrix)
    # Save mappings as before
    with open('anime_mapping.pkl', 'wb') as f:
        pickle.dump(anime_mapping, f)
    with open('user_index_list.pkl', 'wb') as f:
        pickle.dump(user_list, f)
    return dense_matrix, user_list, anime_mapping

# Build the hnswlib index (on startup)
@timeit_decorator
@memory_profiler_decorator
def build_hnsw_index():
    global user_index, user_index_list, anime_id_to_col

    # Check if the index and mappings already exist
    if os.path.exists('dense_matrix.npy'):
        logger.info("Loading dense matrix from disk...")
        dense_matrix = np.load('dense_matrix.npy')
    elif os.path.exists(INDEX_PATH) and os.path.exists(ANIME_MAPPING_PATH) and os.path.exists(USER_INDEX_LIST_PATH):
        logger.info("Loading hnswlib index and mappings from disk...")
        
        # Load mappings
        try:
            with open(ANIME_MAPPING_PATH, 'rb') as f:
                anime_id_to_col = pickle.load(f)
            with open(USER_INDEX_LIST_PATH, 'rb') as f:
                user_index_list = pickle.load(f)
        except Exception as e:
            logger.exception("Failed to load mappings from disk.")
            raise e
        
        # Initialize hnswlib index
        try:
            user_index = hnswlib.Index(space='cosine', dim=len(anime_id_to_col))
            user_index.load_index(INDEX_PATH)
            user_index.set_ef(50)  # Set ef for querying
            logger.info("hnswlib index and mappings loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load hnswlib index.")
            raise e
        return

    # If index does not exist, build it
    logger.info("Building user-anime sparse matrix and hnswlib index...")
    try:
        user_vectors_sparse, user_index_list_local, anime_id_to_col_local = build_user_anime_matrix_sparse()
    except Exception as e:
        logger.exception("Failed to build user-anime matrix.")
        raise e

    logger.info("Normalizing user vectors...")
    # Convert the normalized sparse matrix to dense
    user_vectors_norm = normalize(user_vectors_sparse, norm='l2', axis=1).toarray()

    logger.info("Initializing hnswlib index...")
    index = hnswlib.Index(space='cosine', dim=user_vectors_norm.shape[1])

    index.init_index(max_elements=user_vectors_norm.shape[0] + 10000, ef_construction=100, M=16)

    logger.info("Adding user vectors to hnswlib index...")
    index.add_items(user_vectors_norm, np.arange(user_vectors_norm.shape[0]))

    index.set_ef(50)

    # Assign to globals
    user_index = index
    user_vectors = user_vectors_sparse
    user_index_list = user_index_list_local
    anime_id_to_col = anime_id_to_col_local

    logger.info("Saving hnswlib index and mappings to disk...")
    try:
        user_index.save_index(INDEX_PATH)
        with open(ANIME_MAPPING_PATH, 'wb') as f:
            pickle.dump(anime_id_to_col, f)
        with open(USER_INDEX_LIST_PATH, 'wb') as f:
            pickle.dump(user_index_list, f)
        logger.info("hnswlib index and mappings built and saved successfully.")
    except Exception as e:
        logger.exception("Failed to save hnswlib index and mappings.")
        raise e

# 3) Insert or update a new user in the index
@timeit_decorator
@memory_profiler_decorator
def insert_new_user_into_index(new_user_id, user_top_anime):
    """
    Build a vector for the new/updated user from their top anime and add it to hnsw index & DB.
    """
    global user_index, user_vectors, user_index_list, anime_id_to_col

    # Ensure new_user_id is a Python int
    new_user_id = int(new_user_id)

    if user_index is None:
        # If for some reason the index isn't built yet
        build_hnsw_index()

    if new_user_id in user_index_list:
        logger.info(f"User {new_user_id} already in index. No update performed.")
        return

    # Create a local user vector from the top anime
    # interpret each 'top anime' as rating=10
    num_anime = len(anime_id_to_col)
    new_vector = np.zeros((num_anime,), dtype=np.float32)
    for entry in user_top_anime:
        anime_name = entry.get('name')
        anime_id = get_anime_id_by_name_fuzzy(anime_name)
        if anime_id and anime_id in anime_id_to_col:
            col_idx = anime_id_to_col[anime_id]
            new_vector[col_idx] = 10.0  # or a user-specified rating?

    # Insert ratings into DB
    try:
        with engine.connect() as conn:
            for entry in user_top_anime:
                anime_name = entry.get('name')
                anime_id = get_anime_id_by_name_fuzzy(anime_name)
                if anime_id:
                    conn.execute(
                        text("""
                            INSERT INTO ratings (user_id, anime_id, rating)
                            VALUES (:user_id, :anime_id, :rating)
                            ON CONFLICT (user_id, anime_id) DO UPDATE SET rating = EXCLUDED.rating
                        """),
                        {"user_id": new_user_id, "anime_id": anime_id, "rating": 10}
                    )
    except SQLAlchemyError as e:
        logger.exception("Failed to insert/update ratings in the database.")
        raise e

    # Normalize for cosine
    norm_val = np.linalg.norm(new_vector)
    if norm_val == 0:
        norm_val = 1
    new_vector_norm = (new_vector / norm_val).reshape(1, -1)

    # Add to the index incrementally
    try:
        user_index.add_items(new_vector_norm, np.array([len(user_index_list)], dtype=np.int32))
        user_index_list.append(new_user_id)
        logger.info(f"User {new_user_id} added to hnswlib index.")
    except Exception as e:
        logger.exception("Failed to add new user to hnswlib index.")
        raise e

    # Save updated index and mappings to disk
    try:
        user_index.save_index(INDEX_PATH)
        with open(USER_INDEX_LIST_PATH, 'wb') as f:
            pickle.dump(user_index_list, f)
        logger.info(f"Updated hnswlib index and mappings saved.")
    except Exception as e:
        logger.exception("Failed to save updated hnswlib index and mappings.")
        raise e

# ------------------ SHARED UTILS ------------------

def load_anime_data():
    logger.info("Loading anime data from database...")
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT anime_id, name FROM anime", conn)
        logger.info("Anime data loaded successfully.")
        return df
    except SQLAlchemyError as e:
        logger.exception("Failed to load anime data from database.")
        raise e

anime_df = load_anime_data()

def get_anime_name_by_id(anime_id):
    # Cast to int
    anime_id = int(anime_id)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM anime WHERE anime_id = :anime_id"), {"anime_id": anime_id}).fetchone()
        return result[0] if result else 'Unknown Anime'
    except SQLAlchemyError as e:
        logger.exception(f"Failed to fetch anime name for anime_id {anime_id}.")
        return 'Unknown Anime'

def get_anime_id_by_name_fuzzy(name):
    if not name:
        return None
    choices = anime_df['name'].tolist()
    match, score = process.extractOne(name, choices)
    if score > 70:
        matched_row = anime_df[anime_df['name'] == match]
        return int(matched_row.iloc[0]['anime_id'])
    return None

def recommend_popular_anime():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT anime_id, COUNT(rating) AS rating_count, AVG(rating) AS avg_rating
                    FROM ratings
                    GROUP BY anime_id
                    ORDER BY rating_count DESC, avg_rating DESC
                    LIMIT 20
                """)
            )
            popular_anime = result.fetchall()
        return [
            {
                'anime_id': row[0],
                'anime_name': get_anime_name_by_id(row[0]),
                'count': row[1],
                'avg_rating': row[2]
            }
            for row in popular_anime
        ]
    except SQLAlchemyError as e:
        logger.exception("Failed to fetch popular anime.")
        return []

# --------------------- RECOMMENDATION LOGIC ---------------------

@app.route('/users/recommendations', methods=['POST'])
def get_recommendations():
    try:
        request_data = request.json
        user_id = request_data.get('user_id')
        algorithm = request_data.get('algorithm', 'user-user')  # Default to user-user CF

        if not user_id:
            logger.warning("No user_id provided in the request.")
            return jsonify({"error": "Missing user_id"}), 400

        if algorithm == 'user-user':
            return get_user_user_recommendations(user_id)
        elif algorithm == 'content-based':
            return get_content_based_recommendations()
        else:
            logger.warning(f"Invalid algorithm specified: {algorithm}")
            return jsonify({"error": "Invalid algorithm specified"}), 400
    except Exception as e:
        logger.exception("Error in get_recommendations endpoint.")
        return jsonify({"error": "Internal Server Error"}), 500

# --------------------- USER-USER WITH ANN ---------------------
@timeit_decorator
@memory_profiler_decorator
def get_user_user_recommendations(user_id):
    """
    Approximate user-user approach via hnswlib index.
    - If user not in index, insert them using their top anime list.
    - Then find K nearest neighbors and combine their top-rated anime.
    """
    global user_index, user_vectors, user_index_list

    # Cast user_id to int
    user_id = int(user_id)

    request_data = request.json
    user_top_anime = request_data.get('anime_list', [])

    # If user not in index, insert them
    if user_id not in user_index_list:
        logger.info(f"[User-User] New user {user_id} not in index -> inserting...")
        insert_new_user_into_index(user_id, user_top_anime)

    # If user vectors or index not built, fallback
    if user_index is None:
        logger.warning("ANN index or user_vectors not ready, returning popular anime.")
        return jsonify(recommend_popular_anime())

    # Find the user's internal index
    try:
        internal_id = user_index_list.index(user_id)
    except ValueError:
        # If we still don't have them, fallback
        logger.warning(f"User {user_id} not found after insertion, returning popular.")
        return jsonify(recommend_popular_anime())

    # k nearest neighbors
    k = 10

    # Build user's vector from DB
    user_vector = build_user_vector_from_db(user_id)
    # Replace -1 with 0
    user_vector[user_vector == -1] = 0
    # Convert to sparse if necessary
    user_vector_sparse = csr_matrix(user_vector)
    # Normalize
    user_vector_norm = normalize(user_vector_sparse, norm='l2', axis=1).toarray()

    # Query the index
    try:
        labels, distances = user_index.knn_query(user_vector_norm, k=k)
        neighbor_indices = labels[0]
    except Exception as e:
        logger.exception("Failed to query hnswlib index.")
        return jsonify({"error": "Failed to fetch recommendations."}), 500

    # Convert neighbor_indices to real user_ids
    neighbor_user_ids = []
    for idx in neighbor_indices:
        idx = int(idx)
        if idx < len(user_index_list):
            neighbor_user_ids.append(user_index_list[idx])

    # Exclude user’s own anime
    user_anime_ids = set()
    for anime in user_top_anime:
        anime_name = anime.get('name')
        aid = get_anime_id_by_name_fuzzy(anime_name)
        if aid is not None:
            user_anime_ids.add(int(aid))

    if not neighbor_user_ids:
        # No neighbors found
        return jsonify([])

    # Cast neighbor_user_ids to python int in case they aren't
    neighbor_user_ids = [int(u) for u in neighbor_user_ids]
    user_anime_ids_tuple = tuple(int(a) for a in user_anime_ids)

    # If user_anime_ids is empty, pass a dummy
    if not user_anime_ids_tuple:
        user_anime_ids_tuple = (-1,)  # or some anime_id that doesn't exist

    # Fetch neighbors' ratings for all anime not in user_anime_ids
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT anime_id, rating 
                    FROM ratings 
                    WHERE user_id IN :user_ids
                      AND anime_id NOT IN :anime_ids
                """),
                {"user_ids": tuple(neighbor_user_ids), "anime_ids": user_anime_ids_tuple}
            )
            recommended_ratings = result.fetchall()
    except SQLAlchemyError as e:
        logger.exception("Failed to fetch recommended ratings from database.")
        return jsonify({"error": "Failed to fetch recommendations."}), 500

    # Aggregate
    recommendations = {}
    for (anime_id, rating) in recommended_ratings:
        a_id = int(anime_id)
        if a_id not in recommendations:
            recommendations[a_id] = {'total_rating': 0, 'count': 0}
        recommendations[a_id]['total_rating'] += float(rating)
        recommendations[a_id]['count'] += 1

    # Sort and top 15
    sorted_recommendations = sorted(
        [
            {
                'anime_id': a_id,
                'anime_name': get_anime_name_by_id(a_id),
                'total_rating': rec['total_rating'],
                'count': rec['count']
            }
            for a_id, rec in recommendations.items()
        ],
        key=lambda x: (x['count'], x['total_rating']),
        reverse=True
    )
    # Fetch genres for these anime in one go
    top_n = 15  # or however many you want
    sliced_recs = sorted_recommendations[:top_n]
    anime_ids = [r["anime_id"] for r in sliced_recs]

    if anime_ids:
        # Get genre from anime table
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT anime_id, genre
                        FROM anime
                        WHERE anime_id IN :anime_ids
                    """),
                    {"anime_ids": tuple(anime_ids)}
                )
                rows = result.fetchall()
            # Build a dict: { anime_id: genreStr }
            genre_map = {row[0]: row[1] for row in rows}
        except SQLAlchemyError as e:
            logger.exception("Failed to fetch genres for recommended anime.")
            genre_map = {}
    else:
        genre_map = {}

    # Merge genre into the recommendation list
    for rec in sliced_recs:
        rec["genre"] = genre_map.get(rec["anime_id"], "Unknown Genre")

    for rec in sliced_recs:
        max_count = 1.5  # Adjust based on typical neighbor count
        star_scale = 5
        rec["stars"] = int(round((rec["count"] / max_count) * star_scale))
    
        # Remove unwanted fields
        del rec["count"]
        del rec["total_rating"]

    return jsonify(sliced_recs)

def build_user_vector_from_db(user_id):
    """
    Build a single user vector (dense) from the `ratings` table for user_id.
    We'll use the same anime_id_to_col mapping as in build_user_anime_matrix.
    """
    global anime_id_to_col
    user_id = int(user_id)

    num_anime = len(anime_id_to_col)
    vec = np.zeros((num_anime,), dtype=np.float32)

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT anime_id, rating FROM ratings WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            rows = result.fetchall()
    except SQLAlchemyError as e:
        logger.exception(f"Failed to fetch ratings for user_id {user_id}.")
        return vec  # Return zero vector if failed

    for (aid, r) in rows:
        aid = int(aid)
        if aid in anime_id_to_col:
            col_idx = anime_id_to_col[aid]
            # Replace -1 with 0
            rating = float(r)
            if rating == -1:
                rating = 0.0
            vec[col_idx] = rating
    return vec

# --------------------- CONTENT-BASED ---------------------
@timeit_decorator
@memory_profiler_decorator
def get_content_based_recommendations():
    request_data = request.json
    user_top_anime = request_data.get('anime_list', [])

    if not user_top_anime:
        return jsonify({"error": "No anime list provided in the request."}), 400

    # Collect anime IDs
    user_anime_ids = set()
    for anime in user_top_anime:
        anime_name = anime.get('name')
        aid = get_anime_id_by_name_fuzzy(anime_name)
        if aid is not None:
            user_anime_ids.add(int(aid))

    if not user_anime_ids:
        return jsonify({"error": "No valid anime IDs found for the submitted anime list."}), 404

    # Fetch anime data
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT anime_id, genre, name FROM anime"))
            data = result.fetchall()
        df = pd.DataFrame(data, columns=['anime_id', 'genre', 'name'])
        df['anime_id'] = df['anime_id'].astype(int)
        df['genre'] = df['genre'].astype(str).str.replace(',', ' ').fillna('unknown')
    except SQLAlchemyError as e:
        logger.exception("Failed to fetch anime data for content-based recommendations.")
        return jsonify({"error": "Failed to fetch anime data."}), 500

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = csr_matrix(tfidf.fit_transform(df['genre']))

    # Build indices for liked anime
    liked_indices = []
    for aid in user_anime_ids:
        # find index in df
        found_rows = df.index[df['anime_id'] == aid].tolist()
        if found_rows:
            liked_indices.append(found_rows[0])

    if not liked_indices:
        return jsonify({"error": "No matching anime found in database"}), 404

    similarity_scores = cosine_similarity(tfidf_matrix[liked_indices], tfidf_matrix).mean(axis=0)

    # Rank
    recommendations = []
    for i, score in enumerate(similarity_scores):
        rec_anime_id = int(df.iloc[i]['anime_id'])
        recommendations.append((rec_anime_id, df.iloc[i]['genre'], float(score)))

    recommendations.sort(key=lambda x: x[2], reverse=True)

    # Filter out user’s own anime
    filtered_recs = [rec for rec in recommendations if rec[0] not in user_anime_ids]
    top_recs = filtered_recs[:20]
    anime_ids = [rec[0] for rec in top_recs]

    if anime_ids:
        # Get genre from anime table
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT anime_id, genre
                        FROM anime
                        WHERE anime_id IN :anime_ids
                    """),
                    {"anime_ids": tuple(anime_ids)}
                )
                rows = result.fetchall()
            # Build a dict: { anime_id: genreStr }
            genre_map = {row[0]: row[1] for row in rows}
        except SQLAlchemyError as e:
            logger.exception("Failed to fetch genres for recommended anime.")
            genre_map = {}
    else:
        genre_map = {}

    # Merge genre into the recommendation list using enumerate
    for i, rec in enumerate(top_recs):
        rec = list(rec)  # Convert tuple to list for modification
        rec[1] = genre_map.get(rec[0], "Unknown Genre")  # Update genre
        top_recs[i] = tuple(rec)  # Convert back to tuple and update the list

    # Optionally remove or rename fields you don’t want to expose
    output = []
    for rec in top_recs:
        a_id = rec[0]
        genre = rec[1]
        score = rec[2]
        output.append({
            "anime_id": a_id,
            "anime_name": get_anime_name_by_id(a_id),
            "genre": genre,
            "similarity_score": score
        })

    # Handle empty recommendations
    if not output:
        return jsonify({"message": "No recommendations found."}), 200

    return jsonify(output)


# -------------------- AUTOCOMPLETE ENDPOINT --------------------
@app.route('/api/anime-suggestions', methods=['GET'])
def anime_suggestions():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT name FROM anime
                    WHERE LOWER(name) LIKE :query
                    LIMIT 10
                """),
                {"query": f"%{query}%"}
            )
            matches = [row[0] for row in result.fetchall()]
        return jsonify(matches)
    except SQLAlchemyError as e:
        logger.exception("Failed to fetch anime suggestions.")
        return jsonify({"error": "Failed to fetch suggestions."}), 500

# -------------------------- MAIN --------------------------
if __name__ == '__main__':
    build_hnsw_index()  # Build or load index once before starting
    app.run(debug=True)