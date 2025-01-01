from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import requests
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class DataAnalyzer:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.current_df = None
        
    def generate_analysis(self, prompt, df_info):
        """Generate analysis using Ollama"""
        try:
            context = f"""
            Dataset Information:
            Columns: {', '.join(df_info['columns'])}
            Shape: {df_info['shape']}
            Data Types: {df_info['dtypes']}
            
            Basic Statistics:
            {df_info['description']}
            
            Based on this dataset, {prompt}
            
            Provide your response only in the following JSON format:
            {{
                "analysis": "detailed text analysis",
                "visualization": {{
                    "type": "recommended plot type (line, bar, scatter, histogram, box, etc.)",
                    "x": "column name for x-axis",
                    "y": "column name for y-axis",
                    "title": "suggested plot title",
                    "additional_params": {{}}
                }},
                "insights": ["key insight 1", "key insight 2", "key insight 3"]
            }}
            """

            
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": context,
                    "stream": False
                }
            )
            response.raise_for_status()
            print(response.json()['response'])
            # Parse the response to ensure it's valid JSON
            try:
                return json.loads(response.json()['response'])
            except json.JSONDecodeError:
                return self._generate_fallback_response()
                
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            return self._generate_fallback_response()
    
    def _generate_fallback_response(self):
        """Generate a fallback response when analysis fails"""
        return {
            "analysis": "Unable to generate detailed analysis. Please try a different prompt.",
            "visualization": {
                "type": "bar",
                "x": self.current_df.columns[0],
                "y": self.current_df.columns[1] if len(self.current_df.columns) > 1 else self.current_df.columns[0],
                "title": "Data Overview",
                "additional_params": {}
            },
            "insights": ["No specific insights generated"]
        }
    
    def create_visualization(self, viz_config, df):
        """Create visualization based on configuration"""
        try:
            self.current_df = df
            plot_type = viz_config['type'].lower()
            
            if plot_type == 'line':
                fig = px.line(df, x=viz_config['x'], y=viz_config['y'], title=viz_config['title'])
            elif plot_type == 'bar':
                fig = px.bar(df, x=viz_config['x'], y=viz_config['y'], title=viz_config['title'])
            elif plot_type == 'scatter':
                fig = px.scatter(df, x=viz_config['x'], y=viz_config['y'], title=viz_config['title'])
            elif plot_type == 'histogram':
                fig = px.histogram(df, x=viz_config['x'], title=viz_config['title'])
            elif plot_type == 'box':
                fig = px.box(df, x=viz_config['x'], y=viz_config['y'], title=viz_config['title'])
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                            title="Default Visualization")
            
            return json.loads(fig.to_json())
        
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            # Return a simple fallback visualization
            fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                        title="Error occurred - Showing default visualization")
            return json.loads(fig.to_json())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

analyzer = DataAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read the file into a pandas DataFrame
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Generate dataset info
            df_info = {
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'dtypes': df.dtypes.astype(str).to_dict(),
                'description': df.describe().to_json()
            }
            
            # Save DataFrame to a temporary file
            temp_file = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            df.to_csv(temp_file, index=False)
            
            return jsonify({
                'success': True,
                'df_info': df_info,
                'message': 'File uploaded successfully'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.json
        prompt = data.get('prompt')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename'))
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        
        df_info = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'description': df.describe().to_json()
        }
        
        # Generate analysis
        analysis_result = analyzer.generate_analysis(prompt, df_info)
        
        # Create visualization
        visualization = analyzer.create_visualization(analysis_result['visualization'], df)
        
        return jsonify({
            'analysis': analysis_result['analysis'],
            'visualization': visualization,
            'insights': analysis_result['insights']
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
