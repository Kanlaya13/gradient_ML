import streamlit as st
import numpy as np
import pylab as py
import pandas as pd
import matplotlib.pyplot as plt


st.title("Machine Learning -Gradient Descent")

# Random Data
st.markdown("### Random Data")
st.markdown('ข้อมูลที่นำมา Random เป็นข้อมูลที่ได้สร้างขึ้นมา ซึ่งสร้างข้อมูลทั้งหมดเป็น 0 - 100 โดยมีการสร้างฟังก์ชันเชิงเส้นจะเท่ากับ 2เท่า เราสามารถใช้ข้อมูลนี้ในการศึกษาการสร้างแบบจำลองเชิงเส้นหรือในการพล็อตกราฟเพื่อสังเกตความสัมพันธ์ จะเพิ่มค่าที่ไม่พึงประสงค์ออกไปเพื่อให้มีความสมดุลของข้อมูล')

# Create data
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# Noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# Plot scatter
fig, ax = plt.subplots()
ax.scatter(x, y)

# Display the plot in Streamlit
#st.pyplot(fig)

# Create a button to display the plot
if st.button('Random Data'):
    # Display the plot in Streamlit
    st.pyplot(fig)


# Linear Regression
st.markdown("### Linear Regression")
st.markdown('ก่อนการทำงานของ Gradient Descent ต้องเรียนรู้พื้นฐานของ Linear Regression เพราะว่า Gradient Descent จะทำงานโดยค่าความชัน และค่าคงที่ของสมการเส้นตรง ใน Linear Regression นั่นเองLinear Regression คือ การทำนายโดยการสร้างเส้นตรงบนจุดของข้อมูล ซึ่งเส้นตรงดังกล่าว เกิดมาจากสมการเส้นตรง')
st.latex(r"y = \beta_0 + \beta_1x")

# Create data
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# Noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# Calculate absolute differences between the predicted and actual values
differences = np.abs(y - 2 * x)

# Create a DataFrame with x, y, and differences
data = pd.DataFrame({'X': x, 'Y': y, 'Difference': differences})

# Sort the DataFrame by the difference column in ascending order
sorted_data = data.sort_values('Difference')

# Select the top 10 rows
top_10 = sorted_data.head(10)

# ปรับแต่งกราฟ Linear
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data')
ax.plot(x, 2 * x, color='red', label='Linear Regression')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# แสดงกราฟใน Streamlit Linear
#st.pyplot(fig)

# Display the top 10 values in a table
#st.write('Top 10 Best Values:')
#st.dataframe(top_10)

# Create a button to display the plot
if st.button('Linear Regression'):
    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write('Top 10 Best Values:')
    st.dataframe(top_10)

# Mean Squared Error
st.markdown("### Mean Squared Error (MSE)")
st.markdown('ค่าของความคลาดเคลื่อนในการทำนายหรือประเมินค่าโมเดลทางสถิติ โดยใช้วิธีการเลือกจากการหาผลต่างระหว่างค่าจริง')
st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")

# Create data
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# Noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# สร้าง DataFrame จากข้อมูล
data = {'X': x, 'Y': y}
df = pd.DataFrame(data)

# สร้างคอลัมน์ 'Predicted Y' โดยใช้สมการเชิงเส้น
df['Predicted Y'] = 2 * x

# คำนวณค่า MSE
ydat = pd.DataFrame()
ydat['actual_value'] = df['Y']
ydat['prediction'] = df['Predicted Y']
ydat['square_error'] = (df['Y'] - df['Predicted Y'])**2

# แสดงตารางใน Streamlit
st.write('Mean Squared Error Value:')
st.write(ydat)



# Loss function
st.markdown("### Loss function")
st.markdown('การประเมินความคลาดเคลื่อนระหว่างค่าจริง (observed values) และค่าทำนาย (predicted values) ของโมเดลทางสถิติหรือโมเดลการเรียนรู้ เป้าหมายของฟังก์ชันการสูญเสียคือการวัดความแม่นยำของโมเดลโดยการประเมินผลต่างระหว่างค่าจริงและค่าทำนาย')
st.latex(r''' w = w - \alpha \frac{d}{dw} \frac{1}{n} \sum_{i=1}^{n} (h_{i} - y_{i})^{2} ''')

# Create data
N = 100
D = 100

x = np.random.rand(N) * D
y = 2 * x

# Noise data
na = 50
fakenoise = (np.random.rand(len(y)) - 0.5) * na
y = y + fakenoise

# ฟังก์ชันคำนวณค่า MSE
def mse_cal(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ฟังก์ชัน Loss function
def Loss_func(w):
    h = w * x
    return mse_cal(y, h)

# สร้างหน้า Streamlit
#st.title('Loss Function')

# สร้างตัวแปรสำหรับเก็บค่า Loss function
loss_value = None

# สร้างส่วนของการกรอกค่า Loss function
with st.form(key='loss_form'):
    w_input = st.number_input('Enter the value of w', min_value=0.0, max_value=100.0, step=0.1, value=1.0)
    submit_button = st.form_submit_button(label='Loss Function')
    
    # คำนวณค่า Loss function เมื่อกดปุ่ม Submit
    if submit_button:
        loss_value = Loss_func(w_input)

# แสดงค่า Loss function
if loss_value is not None:
    st.write(f'Loss function for w = {w_input}: {loss_value:.2f}')




# Gradient Descent
st.markdown("### Train Gradient Descent")
st.markdown('Optimization algorithm ตัวหนึ่ง สำหรับใช้ในการหาค่า weight ที่ทำให้ Machine Learning models มีค่า error ต่ำที่สุด โดย Machine Learning ที่นิยมนำ Gradient Descent มาใช้ก็จะมี Linear Regression, Logistic Regression และ Neural Networks เป็นต้น')

# กำหนดข้อมูลเริ่มต้น
X = np.random.rand(100) * 100
y = 3*X + 5 + np.random.randn(100) * 10  # เพิ่มเส้นผ่านจุดบนกราฟ

# กำหนดค่าเริ่มต้นสำหรับพารามิเตอร์ของเส้นตรง
learning_rate = 0.0001
initial_slope = 0
initial_intercept = 0
num_iterations = 1000

# ฟังก์ชันสำหรับการ Gradient Descent
def gradient_descent(X, y, learning_rate, initial_slope, initial_intercept, num_iterations):
    n = len(X)
    slope = initial_slope
    intercept = initial_intercept

    for _ in range(num_iterations):
        y_pred = slope*X + intercept
        slope_gradient = (-2/n) * np.sum(X * (y - y_pred))
        intercept_gradient = (-2/n) * np.sum(y - y_pred)
        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient

    return slope, intercept

# ฝึกข้อมูลด้วย Gradient Descent
final_slope, final_intercept = gradient_descent(X, y, learning_rate, initial_slope, initial_intercept, num_iterations)



# Create a button to display the graph and table
if st.button('Train Gradient Deascent'):
    # Create the graph with the scatter plot and the regression line
    y_pred = final_slope*X + final_intercept
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Data')
    ax.plot(X, y_pred, color='red', label='Gradient Descent')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()

    # Show the graph
    st.pyplot(fig)

    # Create the table with the data
    data = pd.DataFrame({'X': X, 'y': y, 'y_pred': y_pred})

    # Sort the data by the predicted values
    sorted_data = data.sort_values(by='y_pred', ascending=False)

    # Show the top 10 best predicted values in a table
    st.write('Top 10 Best Values:')
    st.write(sorted_data.head(10))