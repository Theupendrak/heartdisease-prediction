from datetime import datetime
from flask import Flask, render_template, request, send_file, session
import pickle
import numpy as np
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load the Random Forest Classifier model
filename = 'heart_disease_model.pkl'
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)
app.secret_key = "your_secret_key"  
# Required for session storage

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/trends')
def trends():
    return render_template('trends.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/prediction')
def prediction():
    return render_template('main.html')

@app.route('/article')
def article():
    return render_template('article.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        user_data = {key: request.form[key] for key in request.form}
        
        # Convert input data for model prediction
        data = np.array([[int(user_data['age']), int(user_data['sex']), int(user_data['cp']), int(user_data['trestbps']), 
                          int(user_data['chol']), int(user_data['fbs']), int(user_data['restecg']), int(user_data['thalach']), 
                          int(user_data['exang']), float(user_data['oldpeak']), int(user_data['slope']), int(user_data['ca']), 
                          int(user_data['thal'])]])

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Predict class and probability
        my_prediction = model.predict(data_scaled)[0]  # 0 (low risk) or 1 (high risk)
        probability = model.predict_proba(data_scaled)[0][1]  # Probability of class 1 (high risk)
        
        
        # Convert probability to percentage
        risk_percentage = round(probability * 100, 2)
        prediction_text = "High Risk" if my_prediction == 1 else "Low Risk"
        
        # Store results in session for PDF download
        session['user_data'] = user_data
        session['prediction'] = prediction_text
        session['risk_percentage'] = risk_percentage
        
        return render_template('result.html', prediction=prediction_text, risk_percentage=risk_percentage)


@app.route('/download_report')
def download_report():
    user_data = session.get('user_data', {})
    prediction = session.get('prediction', "Unknown")
    risk_percentage = session.get('risk_percentage', 0)
    
    pdf_buffer = generate_pdf(user_data, prediction, risk_percentage)
    return send_file(pdf_buffer, as_attachment=True, download_name="Heart_Disease_Report.pdf", mimetype='application/pdf')

def generate_pdf(user_data, prediction, risk_percentage):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Heart Disease Prediction Report")  
    
    # Set Title

    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<b><font size=18 color=red>Heart Disease Prediction Report</font></b><br/><hr/>", styles['Title'])
    # title_table = Table(title)
    # title_table.setStyle(TableStyle([
    #     ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
    #     ('FONTSIZE', (0, 0), (-1, -1), 16),
    #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.red),
    #     ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    # ]))

    elements.append(title)
    
    # Prediction and Risk Percentage

    summary_data = [["Prediction","Value"],["Risk Level", prediction], ["Risk Percentage", f"{risk_percentage}%"]]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    
    elements.append(Table([[' ']]))  
    
    # Spacer
    
    # User Input Data
    user_data_list = [["Features","Values"]]+ [[key, value] for key, value in user_data.items()]
    user_table = Table(user_data_list, colWidths=[200, 200])
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkcyan),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(user_table)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = Paragraph(f"<font size=10 color=grey>Report Generated on: {timestamp}</font>", styles['Normal'])
    elements.append(Spacer(1, 20))  # Spacer before footer
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == '__main__':
    app.run(debug=True)