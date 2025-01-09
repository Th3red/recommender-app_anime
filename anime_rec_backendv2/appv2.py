import psycopg2
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import hnswlib  # NEW
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

from profiler import timeit_decorator, memory_profiler_decorator


app = Flask(__name__)
# Load environment variables
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY")
app.config['SECRET_KEY'] = SECRET_KEY
url = urlparse(DATABASE_URL)
conn = psycopg2.connect(
    database=url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
)
cur = conn.cursor()

# ---------------------------------------------------------
#            GLOBALS FOR ANN (user-user) APPROACH
# ---------------------------------------------------------
user_index = None          # hnswlib index object
user_vectors = None        # numpy array of shape (num_users, num_anime)
user_index_list = []       # list of user_ids in row order (Python ints)
anime_id_to_col = {}       # mapping from anime_id -> column index in user_vectors (Python ints)

# 1) Build user-anime matrix from DB
@timeit_decorator
@memory_profiler_decorator
def build_user_anime_matrix():
    """
    Creates a user x anime matrix (dense, naive approach).
    Returns: (user_vectors, user_index_list, anime_id_to_col)
      - user_vectors: np.ndarray of shape (num_users, num_anime)
      - user_index_list: list of user_ids in row order (ints)
      - anime_id_to_col: dict mapping anime_id -> column index (ints)
    """
    ratings_df = pd.read_sql("SELECT user_id, anime_id, rating FROM ratings", conn)
    
    # Ensure columns are Python int, not numpy.int64
    ratings_df['user_id'] = ratings_df['user_id'].astype(int)
    ratings_df['anime_id'] = ratings_df['anime_id'].astype(int)

    unique_users = ratings_df['user_id'].unique()   # all Python ints now
    unique_anime = ratings_df['anime_id'].unique()  # all Python ints

    # Sort and build lists of Python int
    user_index_list_local = sorted(unique_users.tolist())  # e.g. [1, 2, 5, 10, ...]
    anime_id_list = sorted(unique_anime.tolist())

    # Maps: user_id -> row index, anime_id -> col index
    user_id_to_row = {uid: i for i, uid in enumerate(user_index_list_local)}
    anime_id_to_col_local = {aid: j for j, aid in enumerate(anime_id_list)}

    num_users = len(user_index_list_local)
    num_anime = len(anime_id_list)
    user_vectors_local = np.zeros((num_users, num_anime), dtype=np.float32)

    # Fill with ratings
    for _, row in ratings_df.iterrows():
        u = row['user_id']     # already a Python int
        a = row['anime_id']    # already a Python int
        r = row['rating']      # could be float or int
        user_row_idx = user_id_to_row[u]
        anime_col_idx = anime_id_to_col_local[a]
        user_vectors_local[user_row_idx, anime_col_idx] = r

    return user_vectors_local, user_index_list_local, anime_id_to_col_local


# 2) Build the hnswlib index (on startup)
@timeit_decorator
@memory_profiler_decorator
def build_hnsw_index():
    global user_index, user_vectors, user_index_list, anime_id_to_col

    print("Building user-anime matrix from DB...")
    user_vectors_local, user_index_list_local, anime_id_to_col_local = build_user_anime_matrix()
    user_vectors_dim = user_vectors_local.shape[1]

    # Store them in globals
    user_vectors = user_vectors_local
    user_index_list = user_index_list_local  # e.g. [1, 2, 5, 10, ...] (all Python ints)
    anime_id_to_col = anime_id_to_col_local  # {anime_id(int): col_index(int)}

    # Normalize all user vectors for cosine
    norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    user_vectors_norm = user_vectors / norms

    print("Initializing hnswlib index...")
    index = hnswlib.Index(space='cosine', dim=user_vectors_dim)
    # Let’s allow some overhead for new users
    index.init_index(max_elements=len(user_index_list) + 10000, ef_construction=100, M=16)

    # Add existing user vectors
    index.add_items(user_vectors_norm, np.arange(len(user_index_list)))

    # Increase ef for better search accuracy
    index.set_ef(50)

    # Assign to global
    user_index = index
    print("hnswlib user-user index built successfully.")


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
        print(f"User {new_user_id} already in index. No update performed.")
        return

    # Create a local user vector from the top anime
    # Interpret each 'top anime' as rating=10
    num_anime = len(anime_id_to_col)
    new_vector = np.zeros((num_anime,), dtype=np.float32)
    for entry in user_top_anime:
        anime_name = entry['name']
        anime_id = get_anime_id_by_name_fuzzy(anime_name)
        if anime_id and anime_id in anime_id_to_col:
            col_idx = anime_id_to_col[anime_id]
            new_vector[col_idx] = 10.0  # or a user-specified rating?

    # Insert ratings into DB, casting IDs to Python int
    for entry in user_top_anime:
        anime_id = get_anime_id_by_name_fuzzy(entry['name'])
        if anime_id:
            cur.execute(
                """
                INSERT INTO ratings (user_id, anime_id, rating)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, anime_id) DO UPDATE SET rating = EXCLUDED.rating
                """,
                (int(new_user_id), int(anime_id), 10)
            )
    conn.commit()

    # Normalize for cosine
    norm_val = np.linalg.norm(new_vector)
    if norm_val == 0:
        norm_val = 1
    new_vector_norm = (new_vector / norm_val).reshape(1, -1)

    # Add to the index
    internal_id = len(user_index_list)
    user_index.add_items(new_vector_norm, np.array([internal_id], dtype=np.int32))
    user_index_list.append(new_user_id)

    print(f"User {new_user_id} added to hnswlib with internal ID = {internal_id}.")


# ------------------ UTILS ------------------


def load_anime_data():
    df = pd.read_sql("SELECT anime_id, name FROM anime", conn)
    return df

anime_df = load_anime_data()

def get_anime_name_by_id(anime_id):
    # Cast to int
    anime_id = int(anime_id)
    cur.execute("SELECT name FROM anime WHERE anime_id = %s", (anime_id,))
    result = cur.fetchone()
    return result[0] if result else 'Unknown Anime'

def get_anime_id_by_name_fuzzy(name):
    choices = anime_df['name'].tolist()
    match, score = process.extractOne(name, choices)
    if score > 70:
        matched_row = anime_df[anime_df['name'] == match]
        return int(matched_row.iloc[0]['anime_id'])
    return None

def recommend_popular_anime():
    cur.execute(
        """
        SELECT anime_id, COUNT(rating) AS rating_count, AVG(rating) AS avg_rating
        FROM ratings
        GROUP BY anime_id
        ORDER BY rating_count DESC, avg_rating DESC
        LIMIT 20
        """
    )
    popular_anime = cur.fetchall()
    return [
        {
            'anime_id': row[0],
            'anime_name': get_anime_name_by_id(row[0]),
            'count': row[1],
            'avg_rating': row[2]
        }
        for row in popular_anime
    ]

# --------------------- RECOMMENDATION LOGIC ---------------------

@app.route('/users/recommendations', methods=['POST'])
def get_recommendations():
    request_data = request.json
    user_id = request_data.get('user_id')
    algorithm = request_data.get('algorithm', 'user-user')  # Default to user-user CF

    if algorithm == 'user-user':
        return get_user_user_recommendations(user_id)
    elif algorithm == 'content-based':
        return get_content_based_recommendations()
    else:
        return jsonify({"error": "Invalid algorithm specified"}), 400


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
        print(f"[User-User] New user {user_id} not in index -> inserting...")
        insert_new_user_into_index(user_id, user_top_anime)

    # If user vectors or index not built, fallback
    if (user_index is None) or (user_vectors is None):
        print("ANN index or user_vectors not ready, returning popular anime.")
        return jsonify(recommend_popular_anime())

    # Find the user's internal index
    try:
        internal_id = user_index_list.index(user_id)
    except ValueError:
        # If we still don't have them, fallback
        print(f"User {user_id} not found after insertion, returning popular.")
        return jsonify(recommend_popular_anime())

    # k nearest neighbors
    k = 10

    # If the user was inserted dynamically, they won't exist in user_vectors array
    # So lets ****** build their vector on the fly from user_top_anime or from DB
    user_vector = build_user_vector_from_db(user_id)
    # Normalize
    norm_val = np.linalg.norm(user_vector)
    if norm_val == 0:
        norm_val = 1
    user_vector_norm = (user_vector / norm_val).reshape(1, -1)

    labels, distances = user_index.knn_query(user_vector_norm, k=k)
    neighbor_indices = labels[0]

    # Convert neighbor_indices to real user_ids
    neighbor_user_ids = []
    for idx in neighbor_indices:
        idx = int(idx)
        if idx < len(user_index_list):
            neighbor_user_ids.append(user_index_list[idx])

    # EXCLUDE user’s own anime
    user_anime_ids = set()
    for anime in user_top_anime:
        aid = get_anime_id_by_name_fuzzy(anime['name'])
        if aid is not None:
            user_anime_ids.add(int(aid))

    if not neighbor_user_ids:
        # No neighbors found
        return jsonify([])

    # Cast neighbor_user_ids to python int in case they arent
    neighbor_user_ids = [int(u) for u in neighbor_user_ids]
    user_anime_ids_tuple = tuple(int(a) for a in user_anime_ids)

    # If user_anime_ids is empty, pass a dummy
    if not user_anime_ids_tuple:
        user_anime_ids_tuple = (-1,)  # or some anime_id that doesn't exist

    # Fetch neighbors' ratings for all anime not in user_anime_ids
    cur.execute(
        """
        SELECT anime_id, rating 
        FROM ratings 
        WHERE user_id IN %s
          AND anime_id NOT IN %s
        """,
        (tuple(neighbor_user_ids), user_anime_ids_tuple)
    )
    recommended_ratings = cur.fetchall()

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
    top_n = 15  # or however many
    sliced_recs = sorted_recommendations[:top_n]
    anime_ids = [r["anime_id"] for r in sliced_recs]

    if anime_ids:
        # Get genre from anime table
        cur.execute(
            """
            SELECT anime_id, genre
            FROM anime
            WHERE anime_id IN %s
            """,
            (tuple(anime_ids),)
        )
        rows = cur.fetchall()
        # Build a dict: { anime_id: genreStr }
        genre_map = {row[0]: row[1] for row in rows}
    else:
        genre_map = {}

    # Merge genre into the recommendation list
    for rec in sliced_recs:
        rec["genre"] = genre_map.get(rec["anime_id"], "Unknown Genre")

    for rec in sliced_recs:
        max_count = 1.5  # or some typical neighbor count
        star_scale = 5
        rec["stars"] = int(round((rec["count"] / max_count) * star_scale))

        del rec["count"]
        del rec["total_rating"]

    return jsonify(sliced_recs)

    #return jsonify(sorted_recommendations[:15])


def build_user_vector_from_db(user_id):
    """
    Build a single user vector (dense) from the `ratings` table for user_id.
    We'll use the same anime_id_to_col mapping as in build_user_anime_matrix.
    """
    global anime_id_to_col
    user_id = int(user_id)

    num_anime = len(anime_id_to_col)
    vec = np.zeros((num_anime,), dtype=np.float32)

    # Query the user's ratings
    cur.execute("SELECT anime_id, rating FROM ratings WHERE user_id = %s", (user_id,))
    rows = cur.fetchall()
    for (aid, r) in rows:
        aid = int(aid)
        if aid in anime_id_to_col:
            col_idx = anime_id_to_col[aid]
            vec[col_idx] = float(r)
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
        aid = get_anime_id_by_name_fuzzy(anime['name'])
        if aid is not None:
            user_anime_ids.add(int(aid))

    if not user_anime_ids:
        return jsonify({"error": "No valid anime IDs found for the submitted anime list."}), 404

    # Fetch anime data
    cur.execute("SELECT anime_id, genre, name FROM anime")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['anime_id', 'genre', 'name'])
    df['anime_id'] = df['anime_id'].astype(int)
    df['genre'] = df['genre'].astype(str).str.replace(',', ' ').fillna('unknown')

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

    # Filter out user’s own
    filtered_recs = [rec for rec in recommendations if rec[0] not in user_anime_ids]
    top_recs = filtered_recs[:20]
    anime_ids = [rec[0] for rec in top_recs]

    if not anime_ids:
        return jsonify({"error": "No recommendations found."}), 404

    # Fetch details for top
    cur.execute("""
        SELECT anime_id, name, genre 
        FROM anime
        WHERE anime_id IN %s
        """, (tuple(anime_ids),)
    )
    rows = cur.fetchall()
    details = {int(r[0]): {"name": r[1], "genre": r[2]} for r in rows}

    output = []
    for rec in top_recs:
        a_id = rec[0]
        score = rec[2]
        animed = details.get(a_id, {})
        output.append({
            "anime_id": a_id,
            "anime_name": animed.get("name", "Unknown Anime"),
            "genre": animed.get("genre", "Unknown Genre"),
            "similarity_score": score
        })
    return jsonify(output)


# -------------------- AUTOCOMPLETE ENDPOINT --------------------
@app.route('/api/anime-suggestions', methods=['GET'])
def anime_suggestions():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    matches = anime_df[anime_df['name'].str.contains(query, case=False)]
    return jsonify(matches['name'].head(10).tolist())

# ---------------------- LOAD ANIME DATA (fuzzy) ------------------
#anime_data = pd.read_csv('anime.csv')  # for fuzzy matching
anime_df['anime_id'] = anime_df['anime_id'].astype(int)  # ensure it's Python int

# -------------------------- MAIN --------------------------
if __name__ == '__main__':
    build_hnsw_index()  # Build index once before starting
    app.run(debug=True)
