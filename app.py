from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load data with error handling
try:
    popular_df = pickle.load(open('popular.pkl', 'rb'))
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create dummy data for testing if files don't exist
    popular_df = None
    pt = None
    books = None
    similarity_scores = None

@app.route('/')
def index():
    if popular_df is not None:
        return render_template('index.html',
                             book_name=list(popular_df['Book-Title'].values),
                             author=list(popular_df['Book-Author'].values),
                             image=list(popular_df['Image-URL-M'].values),
                             votes=list(popular_df['num_ratings'].values),
                             rating=list(popular_df['avg_ratings'].values)
                             )
    else:
        # Dummy data for demo
        return render_template('index.html',
                             book_name=["Sample Book 1", "Sample Book 2", "Sample Book 3"],
                             author=["Author 1", "Author 2", "Author 3"],
                             image=["https://via.placeholder.com/150x200", "https://via.placeholder.com/150x200", "https://via.placeholder.com/150x200"],
                             votes=[1000, 850, 750],
                             rating=[4.5, 4.2, 4.8]
                             )

@app.route('/recommend')
def recommend_ui():
    # Get list of available books for autocomplete
    book_list = []
    if pt is not None:
        book_list = list(pt.index)
    return render_template('recommend.html', book_list=book_list)

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    
    if not user_input or pt is None or similarity_scores is None:
        return render_template('recommend.html', 
                             data=[], 
                             error="Book not found or data not loaded",
                             book_list=list(pt.index) if pt is not None else [])
    
    try:
        # Find the index of the book
        indices = np.where(pt.index == user_input)[0]
        if len(indices) == 0:
            return render_template('recommend.html', 
                                 data=[], 
                                 error="Book not found in our database",
                                 book_list=list(pt.index),
                                 user_input=user_input)
        
        index = indices[0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), 
                             key=lambda x: x[1], reverse=True)[1:6]  # Get top 5 recommendations

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            if not temp_df.empty:
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                data.append(item)

        return render_template('recommend.html', 
                             data=data, 
                             user_input=user_input,
                             book_list=list(pt.index))
    
    except Exception as e:
        return render_template('recommend.html', 
                             data=[], 
                             error=f"An error occurred: {str(e)}",
                             book_list=list(pt.index) if pt is not None else [],
                             user_input=user_input)

@app.route('/search_books')
def search_books():
    """API endpoint for book search autocomplete"""
    query = request.args.get('q', '').lower()
    if pt is not None and query:
        matching_books = [book for book in pt.index if query in book.lower()][:10]
        return jsonify(matching_books)
    return jsonify([])

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)