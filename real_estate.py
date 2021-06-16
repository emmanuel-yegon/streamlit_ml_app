import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt

# loading in the model to predict on the data
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
my_data = 'data.csv'
df = pd.read_csv('data.csv')

data = 'Bengaluru_House_Data.csv'
df2 = pd.read_csv('Bengaluru_House_Data.csv')

def prediction(location,sqft, bath, bhk,):
	loc_index = np.where(df.columns==location)[0][0]
		
	x = np.zeros(len(df.columns))
	x[0] = sqft
	x[1] = bath
	x[2] = bhk
	if loc_index >= 0:
		x[loc_index] = 1
	return classifier.predict([x])[0]

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Bengaluru House Price Prediction App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)

	st.title("Bengaluru Price App Prediction")
    
	st.sidebar.header('Specify input parameters')

	location = st.sidebar.selectbox("Select Location",(df.columns[3:]))
	sqft = st.sidebar.number_input("Area(sqft",300,30000)
	bath = st.sidebar.number_input("No of bathrooms",1,4)
	bhk = st.sidebar.number_input("No of bedrooms",1,4)
	result =""
			
	if st.button("Predict"):
		result = prediction(location,sqft, bath, bhk )
	st.success('Predicted Price is ::{}'.format(result))

    # data
	if st.checkbox("Show all datasets"):
		st.subheader("Data")
		st.write(df2.head())
	# data information
	if st.checkbox("Summary  of Data"):
		st.subheader("Data summary")
		st.write(df.describe())

	#visualizing
	st.subheader('Distribution of the no of bathrooms')
	fig = plt.figure()
	plt.hist(df.bath, rwidth=0.8)
	# plt.title('Distribution of the no of bathrooms')
	plt.xlabel("Number of bathrooms")
	plt.ylabel("Count")
	st.pyplot(fig)

	st.subheader('Distribution of the no of bedrooms')
	fig = plt.figure()
	plt.hist(df.bhk, rwidth=0.8)
	# plt.title('Distribution of the no of bathrooms')
	plt.xlabel("Number of bedrooms")
	plt.ylabel("Count")
	st.pyplot(fig)


	
	
		
if __name__=='__main__':
	main()
