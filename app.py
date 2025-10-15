import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Configure page
st.set_page_config(page_title="Product Recommender System", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
    }
    .explanation-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_interactions' not in st.session_state:
    st.session_state.user_interactions = defaultdict(list)
if 'current_user' not in st.session_state:
    st.session_state.current_user = 'user_001'
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Sample Product Catalog with image URLs
PRODUCTS = {
    'P001': {
        'name': 'Wireless Headphones Pro',
        'category': 'Electronics',
        'price': 79.99,
        'features': ['bluetooth', 'noise-canceling', 'wireless'],
        'rating': 4.5,
        'description': 'Premium wireless headphones with active noise cancellation',
        'image': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop'
    },
    'P002': {
        'name': 'Professional Running Shoes',
        'category': 'Sports',
        'price': 89.99,
        'features': ['athletic', 'comfortable', 'durable'],
        'rating': 4.7,
        'description': 'High-performance running shoes for athletes',
        'image': 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop'
    },
    'P003': {
        'name': 'Smart Watch Series X',
        'category': 'Electronics',
        'price': 199.99,
        'features': ['fitness-tracking', 'bluetooth', 'waterproof'],
        'rating': 4.3,
        'description': 'Advanced smartwatch with comprehensive fitness tracking',
        'image': 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop'
    },
    'P004': {
        'name': 'Premium Yoga Mat',
        'category': 'Sports',
        'price': 29.99,
        'features': ['exercise', 'comfortable', 'portable'],
        'rating': 4.6,
        'description': 'Non-slip yoga mat for all fitness levels',
        'image': 'https://images.unsplash.com/photo-1601925260368-ae2f83cf8b7f?w=400&h=300&fit=crop'
    },
    'P005': {
        'name': 'Laptop Backpack Deluxe',
        'category': 'Accessories',
        'price': 49.99,
        'features': ['durable', 'spacious', 'waterproof'],
        'rating': 4.4,
        'description': 'Spacious laptop backpack with multiple compartments',
        'image': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop'
    },
    'P006': {
        'name': 'Portable Bluetooth Speaker',
        'category': 'Electronics',
        'price': 59.99,
        'features': ['bluetooth', 'wireless', 'portable'],
        'rating': 4.5,
        'description': 'Compact speaker with powerful sound quality',
        'image': 'https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop'
    },
    'P007': {
        'name': 'Fitness Tracker Band',
        'category': 'Electronics',
        'price': 69.99,
        'features': ['fitness-tracking', 'waterproof', 'bluetooth'],
        'rating': 4.2,
        'description': 'Track your daily activity and sleep patterns',
        'image': 'https://images.unsplash.com/photo-1575311373937-040b8e1fd5b6?w=400&h=300&fit=crop'
    },
    'P008': {
        'name': 'Insulated Water Bottle',
        'category': 'Sports',
        'price': 19.99,
        'features': ['portable', 'durable', 'sports'],
        'rating': 4.8,
        'description': 'Keep drinks cold for 24 hours or hot for 12 hours',
        'image': 'https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400&h=300&fit=crop'
    },
    'P009': {
        'name': 'Fast Charge Power Bank',
        'category': 'Accessories',
        'price': 39.99,
        'features': ['portable', 'fast-charging', 'compact'],
        'rating': 4.3,
        'description': '20000mAh portable charger with fast charging',
        'image': 'https://images.unsplash.com/photo-1609091839311-d5365f9ff1c5?w=400&h=300&fit=crop'
    },
    'P010': {
        'name': 'Carbon Fiber Tennis Racket',
        'category': 'Sports',
        'price': 129.99,
        'features': ['athletic', 'professional', 'durable'],
        'rating': 4.6,
        'description': 'Professional-grade tennis racket for serious players',
        'image': 'https://images.unsplash.com/photo-1617083278081-e1e8b5d9f295?w=400&h=300&fit=crop'
    },
}

# Helper Functions
def create_product_features_matrix():
    all_features = set()
    for prod in PRODUCTS.values():
        all_features.update(prod['features'])
    
    feature_list = sorted(list(all_features))
    matrix = []
    
    for prod_id in PRODUCTS:
        prod = PRODUCTS[prod_id]
        vector = [1 if feat in prod['features'] else 0 for feat in feature_list]
        vector.extend([1 if prod['category'] == cat else 0 for cat in ['Electronics', 'Sports', 'Accessories']])
        vector.append(prod['price'] / 200)
        vector.append(prod['rating'] / 5)
        matrix.append(vector)
    
    return np.array(matrix), list(PRODUCTS.keys())

def get_recommendations(user_id, n=3):
    interactions = st.session_state.user_interactions[user_id]
    
    if not interactions:
        sorted_prods = sorted(PRODUCTS.items(), key=lambda x: x[1]['rating'], reverse=True)
        return [pid for pid, _ in sorted_prods[:n]]
    
    viewed_products = [i['product_id'] for i in interactions if i['action'] == 'view']
    purchased_products = [i['product_id'] for i in interactions if i['action'] == 'purchase']
    
    feature_matrix, product_ids = create_product_features_matrix()
    
    user_vector = np.zeros(feature_matrix.shape[1])
    for pid in viewed_products:
        if pid in product_ids:
            idx = product_ids.index(pid)
            user_vector += feature_matrix[idx] * 0.5
    
    for pid in purchased_products:
        if pid in product_ids:
            idx = product_ids.index(pid)
            user_vector += feature_matrix[idx] * 2.0
    
    if user_vector.sum() == 0:
        sorted_prods = sorted(PRODUCTS.items(), key=lambda x: x[1]['rating'], reverse=True)
        return [pid for pid, _ in sorted_prods[:n]]
    
    similarities = cosine_similarity([user_vector], feature_matrix)[0]
    
    product_scores = list(zip(product_ids, similarities))
    product_scores = [(pid, score) for pid, score in product_scores if pid not in purchased_products]
    product_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [pid for pid, _ in product_scores[:n]]

def generate_explanation(user_id, product_id, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        interactions = st.session_state.user_interactions[user_id]
        product = PRODUCTS[product_id]
        
        viewed = [PRODUCTS[i['product_id']]['name'] for i in interactions if i['action'] == 'view']
        purchased = [PRODUCTS[i['product_id']]['name'] for i in interactions if i['action'] == 'purchase']
        
        prompt = f"""You are an e-commerce recommendation system. Explain why this product is recommended to the user in a professional, data-driven manner. Keep it concise (2-3 sentences).

Product: {product['name']}
Category: {product['category']}
Price: ${product['price']}
Features: {', '.join(product['features'])}
Rating: {product['rating']}/5

User's History:
- Viewed: {', '.join(viewed) if viewed else 'No previous views'}
- Purchased: {', '.join(purchased) if purchased else 'No previous purchases'}

Provide a clear explanation based on the user's behavior and product attributes."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"This product is recommended based on your browsing patterns and preferences in {product['category']}. With a {product['rating']}/5 rating and features including {', '.join(product['features'][:2])}, it aligns well with your interests."

def record_interaction(user_id, product_id, action):
    st.session_state.user_interactions[user_id].append({
        'product_id': product_id,
        'action': action,
        'timestamp': datetime.now().isoformat()
    })

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key, help="Enter your Google Gemini API key")
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key configured")
    else:
        st.info("Get your free API key at: makersuite.google.com")
    
    st.divider()
    
    st.header("User Profile")
    st.session_state.current_user = st.selectbox(
        "Select User Account",
        ['user_001', 'user_002', 'user_003'],
        help="Switch between different user profiles"
    )
    
    st.subheader("Recent Activity")
    interactions = st.session_state.user_interactions[st.session_state.current_user]
    if interactions:
        for i in interactions[-5:]:
            action_emoji = "VIEW" if i['action'] == 'view' else "PURCHASE"
            st.text(f"[{action_emoji}] {PRODUCTS[i['product_id']]['name']}")
    else:
        st.info("No activity recorded")
    
    if st.button("Clear Activity History", type="secondary"):
        st.session_state.user_interactions[st.session_state.current_user] = []
        st.rerun()

# Main Content
st.markdown('<div class="main-header">Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered personalized product recommendations</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Recommendations", "Product Catalog", "Analytics"])

with tab1:
    st.subheader("Personalized Product Recommendations")
    
    if not st.session_state.api_key:
        st.warning("Configure your Gemini API key in the sidebar to enable AI-generated explanations")
    
    col_refresh, col_empty = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", type="primary"):
            st.rerun()
    
    st.divider()
    
    recommended_ids = get_recommendations(st.session_state.current_user, n=3)
    
    for prod_id in recommended_ids:
        product = PRODUCTS[prod_id]
        
        col_img, col_content = st.columns([1, 2])
        
        with col_img:
            st.image(product['image'], use_container_width=True)
        
        with col_content:
            st.markdown(f"### {product['name']}")
            st.markdown(f"**Category:** {product['category']} | **Price:** ${product['price']} | **Rating:** {product['rating']}/5.0")
            st.markdown(f"{product['description']}")
            st.markdown(f"**Key Features:** {', '.join(product['features'])}")
            
            if st.session_state.api_key:
                with st.expander("Why is this recommended?"):
                    with st.spinner("Generating explanation..."):
                        explanation = generate_explanation(
                            st.session_state.current_user,
                            prod_id,
                            st.session_state.api_key
                        )
                        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Details", key=f"view_{prod_id}"):
                    record_interaction(st.session_state.current_user, prod_id, 'view')
                    st.success("Added to view history")
                    st.rerun()
            with col2:
                if st.button("Add to Cart", key=f"buy_{prod_id}", type="primary"):
                    record_interaction(st.session_state.current_user, prod_id, 'purchase')
                    st.success("Added to cart")
                    st.rerun()
        
        st.divider()

with tab2:
    st.subheader("Complete Product Catalog")
    
    category_filter = st.multiselect(
        "Filter by Category",
        options=['Electronics', 'Sports', 'Accessories'],
        default=['Electronics', 'Sports', 'Accessories']
    )
    
    filtered_products = {pid: p for pid, p in PRODUCTS.items() if p['category'] in category_filter}
    
    cols = st.columns(3)
    for idx, (pid, product) in enumerate(filtered_products.items()):
        with cols[idx % 3]:
            st.image(product['image'], use_container_width=True)
            st.markdown(f"**{product['name']}**")
            st.markdown(f"${product['price']} | {product['rating']}/5")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View", key=f"cat_view_{pid}"):
                    record_interaction(st.session_state.current_user, pid, 'view')
                    st.rerun()
            with col2:
                if st.button("Buy", key=f"cat_buy_{pid}"):
                    record_interaction(st.session_state.current_user, pid, 'purchase')
                    st.rerun()

with tab3:
    st.subheader("User Analytics Dashboard")
    
    user_interactions = st.session_state.user_interactions[st.session_state.current_user]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        views = len([i for i in user_interactions if i['action'] == 'view'])
        st.metric("Total Views", views)
    
    with col2:
        purchases = len([i for i in user_interactions if i['action'] == 'purchase'])
        st.metric("Total Purchases", purchases)
    
    with col3:
        conversion_rate = (purchases / views * 100) if views > 0 else 0
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    st.divider()
    
    if user_interactions:
        st.subheader("Interaction History")
        history_df = pd.DataFrame([
            {
                'Product Name': PRODUCTS[i['product_id']]['name'],
                'Category': PRODUCTS[i['product_id']]['category'],
                'Action': i['action'].upper(),
                'Timestamp': i['timestamp'].split('T')[0] + ' ' + i['timestamp'].split('T')[1].split('.')[0]
            }
            for i in user_interactions
        ])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("Category Distribution")
        category_counts = {}
        for i in user_interactions:
            cat = PRODUCTS[i['product_id']]['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            chart_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Interactions'])
            st.bar_chart(chart_df.set_index('Category'))
    else:
        st.info("No interaction data available. Start browsing products to see analytics.")

st.divider()
st.caption("Recommendation Engine: Collaborative filtering with content-based features | AI Explanations: Google Gemini")