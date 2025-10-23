import streamlit as st
import numpy as np
import pandas as pd

# Title of the app
st.title('My First Streamlit App')

# Displaying text
st.write("Hello, Streamlit!")

# Creating a slider
x = st.slider("Select a number", 0, 100, 25)

# Displaying the square of the slider value
st.write(f"Square of {x} is {x ** 2}")

# Creating a dataframe
data = pd.DataFrame({
    'Column 1': np.random.randn(10),
    'Column 2': np.random.randn(10),
})

# Displaying the dataframe
st.write(data)
