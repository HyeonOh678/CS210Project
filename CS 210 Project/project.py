# import libs
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
app = Flask(__name__)

# Load data
df = pd.read_csv(r"C:\Users\Bryan Oh\Desktop\CS 210 Project\amazon.csv")

#remove duplicates
df = df.drop_duplicates()

#standardize col names
df.columns = [
    'product_id', 'product_name', 'category', 'discounted_price', 'actual_price',
    'discount_percentage', 'rating', 'rating_count', 'about_product', 'user_id',
    'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link'
]

# Since the user_id row has a bunch of user_ids separated by commas, we will separate them
#then we will make new rows for each user
df['user_id'] = df['user_id'].str.split(',')
df = df.explode('user_id')

#We get rid of whitespace characters
df['user_id'] = df['user_id'].str.strip()

#we can use strip to get rid of the Indian rupee sign and also the % sign.
df["discounted_price"] = df["discounted_price"].str.lstrip("₹")
df["actual_price"] = df["actual_price"].str.lstrip("₹")
df["discount_percentage"] = df["discount_percentage"].str.rstrip("%")

#also replace commas in discounted price and rating_count
df["discounted_price"] = df["discounted_price"].str.replace('[,]','', regex=True)
df["actual_price"] = df["actual_price"].str.replace('[,]','', regex=True)
df["rating_count"] = df["rating_count"].str.replace('[,]', '', regex=True)

#then we can also convert these to their proper numeric form
df["discounted_price"] = df["discounted_price"].astype(float)
df["actual_price"] = df["actual_price"].astype(float)
df["discount_percentage"] = df["discount_percentage"].astype(float)

#We need rating to be numeric as we might need to use this for our recommendation system perhaps later
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

#replace null/missing values with 0
df["rating_count"].fillna('0', inplace=True)

df["category"] = df["category"].str.split("|").str[0] #this is to simplify categories, since there are so many categories

# Create user matrix and calculate similarity
userMatrix = df.pivot_table(index='user_id', columns='product_id', values='rating')
userSimilarity = cosine_similarity(userMatrix.fillna(0))
userDF = pd.DataFrame(userSimilarity, index=userMatrix.index, columns=userMatrix.index)

# We have a user id example here
user_id = 'AG3D6O4STAQKAY2UVGEUV46KN35Q'
similar_users = userDF[user_id].sort_values(ascending=False) #this is to ensure that other user_ids with highest similarity are brought to the top when displaying

# Drop user
similar_users.drop(index = user_id, inplace=True)

def get_recommendations(user_id):

  similar_users = userDF[user_id].sort_values(ascending=False)
  # This recommends products to the selected user_id based on similar users
  user_products = df[df['user_id'] == user_id]['product_id'].unique()
  recommended_products = list(user_products) #this is to make sure that the recommended products include what the user already bought. Relevant items will be our true positives for the score calculation.

  #this function will add whatever items that similar users bought to the recommended_products list.
  for similar_user, similarity in similar_users.items():
    if similarity == 1.0:  #Only compare users with the highest 100% similarity
      products = df[df['user_id'] == similar_user]['product_id']
      for product in products:
          if product not in user_products:
              recommended_products.append(product)
      if len(recommended_products) >= 20: #we don't want to show too many products or only show the top 7-10 maybe.
          break

  return recommended_products




@app.route('/')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        recommendations = get_recommendations(user_id)
        return render_template('recommendations.html', recommendations=recommendations)
    return render_template('index.html')


if __name__ == '__main__':

    app.run()

