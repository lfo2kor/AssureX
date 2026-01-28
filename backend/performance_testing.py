from flask import Flask, render_template_string, request, jsonify, Response
import phoenix as px
from opentelemetry import trace
from phoenix.otel import register
import requests
import time
import uuid
import pandas as pd
import json
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import io
import csv
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

phoenix_session = None
tracer = None

# Globals to store last test runs
last_accuracy_results = None
last_accuracy_stats = None
last_load_results = None
last_load_stats = None

def init_phoenix():
    global phoenix_session, tracer
    if phoenix_session is None:
        try:
            print("🔧 Initializing Phoenix...")
            phoenix_session = px.launch_app()
            print(f"✅ Phoenix launched at: {phoenix_session.url}")
            tracer_provider = register()
            tracer = trace.get_tracer(__name__)
            print("✅ Tracer registered successfully")
        except Exception as e:
            print(f"❌ Phoenix initialization failed: {e}")
            return None
    return phoenix_session.url if phoenix_session else None

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()
    text1_clean = re.sub(r'\s+', ' ', text1_clean)
    text2_clean = re.sub(r'\s+', ' ', text2_clean)
    similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio() * 100
    return similarity

def clean_response_text(text):
    if not text:
        return ""
    
    text = re.sub(r'\[chunk_[^\]]+\]', '', text)
    text = re.sub(r'\[source_[^\]]+\]', '', text)
    text = re.sub(r'\[doc_[^\]]+\]', '', text)
    text = re.sub(r'\[ref_[^\]]+\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = text.replace('\\n', ' ').replace('\\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def phoenix_llm_evaluate(question, expected_answer, actual_answer, api_key, model_name="gpt-4o", 
                        azure_endpoint=None, azure_deployment=None, azure_api_version=None):
    """
    Use Phoenix with direct OpenAI API calls (more reliable than llm_classify with Azure)
    """
    try:
        prompt = f"""You are evaluating an AI assistant's answer to a question.

[Question]
{question}

[Expected Answer]
{expected_answer}

[AI's Actual Answer]
{actual_answer}

Compare the AI's actual answer with the expected answer. Evaluate based on:
1. Factual accuracy
2. Completeness
3. Relevance

Respond with ONLY ONE WORD followed by a brief explanation:
- "correct" if the answer is accurate and complete (90-100% accurate)
- "partial" if the answer is somewhat correct but incomplete (40-89% accurate)
- "incorrect" if the answer is wrong or missing key information (0-39% accurate)

Format: [WORD] - [Brief explanation]

Example: correct - The answer accurately covers all key points about RFID technology."""

        # Use direct API calls with Phoenix tracing
        if azure_endpoint and azure_deployment:
            print(f"🔵 Using Phoenix-tracked Azure OpenAI: {azure_deployment}")
            
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            url = f"{azure_endpoint}/openai/deployments/{azure_deployment}/chat/completions?api-version={azure_api_version or '2024-08-01-preview'}"
            
            data = {
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            # This call will be traced by Phoenix if tracer is active
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse response
                label = "partial"
                explanation = content
                
                content_lower = content.lower()
                if content_lower.startswith("correct"):
                    label = "correct"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                elif content_lower.startswith("incorrect"):
                    label = "incorrect"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                elif content_lower.startswith("partial"):
                    label = "partial"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                
                score_map = {"correct": 95, "partial": 55, "incorrect": 15}
                score = score_map.get(label, 50)
                
                print(f"✅ Phoenix Eval: {label} ({score}) - {explanation[:50]}...")
                return score, f"{label}: {explanation}"
            else:
                print(f"❌ Azure API error: {response.status_code}")
                return calculate_similarity(expected_answer, actual_answer), "API error, using similarity"
                
        else:
            # Regular OpenAI
            print(f"🟢 Using Phoenix-tracked OpenAI: {model_name}")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                    headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                label = "partial"
                explanation = content
                
                content_lower = content.lower()
                if content_lower.startswith("correct"):
                    label = "correct"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                elif content_lower.startswith("incorrect"):
                    label = "incorrect"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                elif content_lower.startswith("partial"):
                    label = "partial"
                    explanation = content.split("-", 1)[1].strip() if "-" in content else content
                
                score_map = {"correct": 95, "partial": 55, "incorrect": 15}
                score = score_map.get(label, 50)
                
                print(f"✅ Phoenix Eval: {label} ({score}) - {explanation[:50]}...")
                return score, f"{label}: {explanation}"
            else:
                print(f"❌ OpenAI API error: {response.status_code}")
                return calculate_similarity(expected_answer, actual_answer), "API error, using similarity"
            
    except Exception as e:
        print(f"❌ Phoenix evaluation error: {e}")
        return calculate_similarity(expected_answer, actual_answer), f"Error: {str(e)[:50]}"

def custom_phoenix_evaluator(question, expected_answer, actual_answer, api_key, model_name="gpt-4o",
                             azure_endpoint=None, azure_deployment=None, azure_api_version=None):
    """
    Custom Phoenix evaluator with granular 0-100 scoring using direct API calls
    """
    try:
        prompt = f"""You are an expert evaluator assessing AI-generated answers.

Question: {question}

Expected Answer: {expected_answer}

Actual Answer: {actual_answer}

Score the Actual Answer on a scale of 0-100 based on:
1. Factual accuracy compared to Expected Answer
2. Completeness of information
3. Relevance to the question
4. Clarity and coherence
5.Ignore images and links if it is not there in expected answer

Respond with ONLY a number between 0-100 followed by a hyphen and brief reasoning.
Format: [SCORE] - [reasoning]

Example: 85 - The answer correctly addresses the main points but lacks some detail about X."""

        if azure_endpoint and azure_deployment:
            print(f"🔵 Phoenix Custom with Azure: {azure_deployment}")
            
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            url = f"{azure_endpoint}/openai/deployments/{azure_deployment}/chat/completions?api-version={azure_api_version or '2024-08-01-preview'}"
            
            data = {
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Respond with score and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse score from response
                match = re.search(r'(\d+)\s*-\s*(.+)', content)
                if match:
                    score = float(match.group(1))
                    reasoning = match.group(2).strip()
                    print(f"✅ Phoenix Custom: {score} - {reasoning[:50]}...")
                    return score, reasoning
                else:
                    # Fallback: extract first number
                    numbers = re.findall(r'\b(\d+)\b', content)
                    if numbers:
                        score = float(numbers[0])
                        reasoning = content
                        print(f"✅ Phoenix Custom: {score} (parsed)")
                        return score, reasoning
                    else:
                        return calculate_similarity(expected_answer, actual_answer), "Failed to parse score"
            else:
                print(f"❌ Azure API error: {response.status_code}")
                return calculate_similarity(expected_answer, actual_answer), "API error"
                
        else:
            # OpenAI
            print(f"🟢 Phoenix Custom with OpenAI: {model_name}")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Respond with score and reasoning."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                    headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                match = re.search(r'(\d+)\s*-\s*(.+)', content)
                if match:
                    score = float(match.group(1))
                    reasoning = match.group(2).strip()
                    print(f"✅ Phoenix Custom: {score} - {reasoning[:50]}...")
                    return score, reasoning
                else:
                    numbers = re.findall(r'\b(\d+)\b', content)
                    if numbers:
                        score = float(numbers[0])
                        reasoning = content
                        return score, reasoning
                    else:
                        return calculate_similarity(expected_answer, actual_answer), "Failed to parse score"
            else:
                print(f"❌ OpenAI API error: {response.status_code}")
                return calculate_similarity(expected_answer, actual_answer), "API error"
            
    except Exception as e:
        print(f"❌ Custom Phoenix eval error: {e}")
        return calculate_similarity(expected_answer, actual_answer), f"Error: {str(e)[:50]}"

# HTML report generators (unchanged)
def generate_accuracy_html(results, stats):
    template = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Accuracy Report</title>
      <style>
        body{font-family:Segoe UI,Arial; padding:24px; background:#f4f6fb;}
        .card{background:white;border-radius:10px;padding:16px;margin-bottom:12px;box-shadow:0 6px 18px rgba(0,0,0,0.06)}
        h1{color:#333}
        .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
        .stat{padding:12px;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;min-width:120px;text-align:center}
        table{width:100%;border-collapse:collapse}
        th{background:#667eea;color:white;padding:8px;text-align:left}
        td{padding:8px;border-bottom:1px solid #e8e8e8;vertical-align:top}
        .badge{padding:6px 10px;border-radius:999px;font-weight:600}
        .high{background:#d4edda;color:#155724}
        .mid{background:#fff3cd;color:#856404}
        .low{background:#f8d7da;color:#721c24}
        pre {white-space:pre-wrap;word-wrap:break-word;margin:0}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>🧪 Accuracy Test Report (Phoenix LLM-as-Judge)</h1>
        <p>Generated: {{ now }}</p>
      </div>

      <div class="card">
        <div class="stats">
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.total_tests }}</div>
            <div>Tests</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.avg_accuracy }}%</div>
            <div>Avg Score</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.avg_time }}ms</div>
            <div>Avg Time</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.total_time }}s</div>
            <div>Total Time</div>
          </div>
        </div>

        <table>
          <thead>
            <tr><th>#</th><th>Question</th><th>Expected Response</th><th>Generated Response</th><th>Score</th><th>Reasoning</th><th>Time</th></tr>
          </thead>
          <tbody>
            {% for r in results %}
            <tr>
              <td>{{ loop.index }}</td>
              <td><pre>{{ r.question }}</pre></td>
              <td><pre>{{ r.expected_answer }}</pre></td>
              <td><pre>{{ r.actual_response }}</pre></td>
              <td>
                {% if r.accuracy >= 70 %}
                  <span class="badge high">{{ '%.1f'|format(r.accuracy) }}%</span>
                {% elif r.accuracy >= 40 %}
                  <span class="badge mid">{{ '%.1f'|format(r.accuracy) }}%</span>
                {% else %}
                  <span class="badge low">{{ '%.1f'|format(r.accuracy) }}%</span>
                {% endif %}
              </td>
              <td><pre>{{ r.reasoning or '-' }}</pre></td>
              <td>{{ '%.0f'|format(r.time_ms) }}ms</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    return render_template_string(template, results=results, stats=stats, now=time.strftime("%Y-%m-%d %H:%M:%S"))

def generate_load_html(results, stats):
    template = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Load Test Report</title>
      <style>
        body{font-family:Segoe UI,Arial; padding:24px; background:#f4f6fb;}
        .card{background:white;border-radius:10px;padding:16px;margin-bottom:12px;box-shadow:0 6px 18px rgba(0,0,0,0.06)}
        h1{color:#333}
        .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
        .stat{padding:12px;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;min-width:120px;text-align:center}
        table{width:100%;border-collapse:collapse}
        th{background:#667eea;color:white;padding:8px;text-align:left}
        td{padding:8px;border-bottom:1px solid #e8e8e8}
        pre {white-space:pre-wrap;word-wrap:break-word;margin:0}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>👥 Load Test Report</h1>
        <p>Generated: {{ now }}</p>
      </div>

      <div class="card">
        <div class="stats">
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.total_requests }}</div>
            <div>Requests</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.success_rate }}%</div>
            <div>Success Rate</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.avg_latency }}ms</div>
            <div>Avg Latency</div>
          </div>
          <div class="stat">
            <div style="font-size:20px;font-weight:700">{{ stats.total_time }}s</div>
            <div>Total Time</div>
          </div>
        </div>

        <table>
          <thead>
            <tr><th>User</th><th>Query</th><th>Status</th><th>Latency</th></tr>
          </thead>
          <tbody>
            {% for r in results %}
            <tr>
              <td>{{ r.user_id }}</td>
              <td><pre>{{ r.query }}</pre></td>
              <td>{{ r.status }}</td>
              <td>{{ '%.0f'|format(r.latency_ms) }}ms</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    return render_template_string(template, results=results, stats=stats, now=time.strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/')
def index():
    phoenix_url = init_phoenix()
    if not phoenix_url:
        phoenix_url = "Phoenix failed to start"
    
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Phoenix Testing Suite with LLM-as-Judge</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .phoenix-link {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            margin-top: 10px;
            text-decoration: none;
            color: white;
            font-weight: 600;
        }
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            color: #666;
            border-bottom: 3px solid transparent;
        }
        .tab.active {
            color: #667eea;
            background: white;
            border-bottom-color: #667eea;
        }
        .tab-content {
            display: none;
            padding: 30px;
        }
        .tab-content.active {
            display: block;
        }
        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], textarea, select, input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        textarea {
            resize: vertical;
            min-height: 80px;
            font-family: inherit;
        }
        .file-upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
        }
        .file-info {
            margin-top: 15px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 8px;
            display: none;
        }
        .file-info.show { display: block; }
        .btn-execute {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
        }
        .btn-execute:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loading.show { display: block; }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .results.show { display: block; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-card.success { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }
        .stat-card.warning { background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    font-size: 14px;
    border: 2px solid #667eea;
}
.result-table th {
    background: #667eea;
    color: white;
    padding: 12px 8px;
    text-align: left;
    font-weight: 600;
    border: 1px solid #5a6ecc;
}
.result-table td {
    padding: 10px 8px;
    border: 1px solid #e0e0e0;
}
.result-table tr:nth-child(even) {
    background-color: #f8f9ff;
}
        .accuracy-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            display: inline-block;
        }
        .accuracy-high { background: #d4edda; color: #155724; }
        .accuracy-medium { background: #fff3cd; color: #856404; }
        .accuracy-low { background: #f8d7da; color: #721c24; }
        .status-success { color: #28a745; font-weight: 600; }
        .status-error { color: #dc3545; font-weight: 600; }
        .test-mode {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        .mode-card {
            padding: 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            cursor: pointer;
            text-align: center;
        }
        .mode-card.selected {
            border-color: #667eea;
            background: #f0f4ff;
        }
        .eval-options {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .eval-option {
            display: flex;
            align-items: center;
            cursor: pointer;
        }
        .eval-option input[type="radio"] {
            width: auto;
            margin-right: 8px;
        }
        .btn-download {
    background: linear-gradient(135deg, #20c997 0%, #28a745 100%);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.btn-download:hover {
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}
.btn-download::before {
    content: "📥";
    margin-right: 8px;
    font-size: 16px;
}
        .llm-config {
            display: block;
            background: #f0f4ff;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #667eea;
            margin-bottom: 20px;
        }
        #multiUserOptions { display: none; }

        #fullModal {
            display: none;
            position: fixed;
            top: 6%;
            left: 50%;
            transform: translateX(-50%);
            width: 84%;
            max-width: 1200px;
            max-height: 82%;
            overflow: auto;
            background: #fff;
            border: 2px solid #667eea;
            padding: 18px;
            z-index: 9999;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        #fullModal pre { white-space: pre-wrap; word-wrap: break-word; }
        #fullModalClose { float: right; cursor: pointer; background:#eee; border:0; padding:6px 10px; border-radius:6px; }
        
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Phoenix Testing Suite</h1>
            <p>Comprehensive testing with Phoenix LLM-as-Judge</p>
            <a href=\"""" + phoenix_url + """\" target="_blank" class="phoenix-link">📊 Open Phoenix Dashboard</a>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="accuracy">🎯 Accuracy Testing</div>
            <div class="tab" data-tab="load">👥 Load Testing</div>
        </div>
        
        <!-- ACCURACY TESTING TAB -->
        <div class="tab-content active" id="accuracyTab">
            <div class="info-box">
                <strong>🤖 Phoenix LLM-as-Judge Enabled</strong><br>
                This test uses Phoenix's built-in evaluation framework for accurate, consistent scoring.
                All evaluations are tracked in the Phoenix dashboard.
            </div>
            
            <form id="accuracyForm">
                <div class="form-group">
                    <label>Base URL</label>
                    <input type="text" id="baseUrl1" value="http://pa-backend-dev.westeurope.cloudapp.azure.com">
                </div>
                
                <div class="form-group">
                    <label>User ID</label>
                    <input type="text" id="userId1" value="test_user">
                </div>
                
                <div class="form-group">
    <label>Evaluation Method</label>
    <div class="info-box">
        <strong>🎯 Phoenix Custom (0-100 scale)</strong><br>
        Using continuous numerical scoring for precise evaluation on a 0-100 scale.
    </div>
    <!-- Add a hidden input to maintain compatibility -->
    <input type="hidden" name="evalMethod" value="phoenix_custom" id="evalMethodFixed">
</div>
                
                <div class="llm-config" id="llmConfig">
                    <h4 style="color: #667eea; margin-bottom: 15px;">🤖 LLM Configuration</h4>
                    
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="llmProvider">
                            <option value="azure">Azure OpenAI</option>
                            <option value="openai">OpenAI</option>
                        </select>
                    </div>
                    
                    <div id="azureConfig">
                        <div class="form-group">
                            <label>Azure Deployment Name</label>
                            <input type="text" id="azureDeployment" placeholder="gpt-4o">
                        </div>
                        <div class="form-group">
                            <label>Azure Endpoint</label>
                            <input type="text" id="azureEndpoint" placeholder="https://your-resource.openai.azure.com">
                        </div>
                        <div class="form-group">
                            <label>API Version</label>
                            <input type="text" id="azureApiVersion" value="2024-08-01-preview">
                        </div>
                        <div class="form-group">
                            <label>Azure API Key</label>
                            <input type="password" id="azureApiKey" placeholder="Your Azure OpenAI key">
                        </div>
                    </div>
                    
                    <div id="openaiConfig" style="display: none;">
                        <div class="form-group">
                            <label>Model</label>
                            <select id="judgeModel">
                                <option value="gpt-4o">GPT-4o</option>
                                <option value="gpt-4o-mini">GPT-4o-mini</option>
                                <option value="gpt-4-turbo">GPT-4 Turbo</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>OpenAI API Key</label>
                            <input type="password" id="judgeApiKey" placeholder="sk-proj-...">
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Upload Test Cases (CSV/Excel)</label>
                    <div class="file-upload-area" id="fileUploadArea">
                        <div style="font-size: 48px;">📁</div>
                        <p>Click to upload or drag & drop</p>
                        <p style="font-size: 12px; color: #666;">CSV or Excel with: question, expected_answer</p>
                        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
                    </div>
                    <div class="file-info" id="fileInfo">
                        <strong>File:</strong> <span id="fileName"></span><br>
                        <strong>Test cases:</strong> <span id="fileRows"></span>
                    </div>
                </div>
                
                <button type="button" class="btn-execute" id="accuracyBtn" disabled>🚀 Run Accuracy Test</button>
            </form>
            
            <div class="loading" id="loading1">
                <div class="spinner"></div>
                <p>Testing with Phoenix LLM-as-Judge...</p>
            </div>
            
            <div class="results" id="results1">
                <div class="stats-grid" id="statsGrid1"></div>
                <table class="result-table" border="1" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr>
                            <th>Sl.no</th>
                            <th>Question</th>
                            <th>Expected Response</th>
                            <th>Generated Response</th>
                            <th>Score</th>
                            <th>Reasoning</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody id="resultBody1"></tbody>
                </table>
            </div>

            <button class="btn-download" id="downloadReportBtnAccuracy" style="display:none; margin-top:12px;">Download Accuracy Report</button>
        </div>
        
        <!-- LOAD TESTING TAB -->
        <div class="tab-content" id="loadTab">
            <form id="loadForm">
                <div class="form-group">
                    <label>Base URL</label>
                    <input type="text" id="baseUrl2" value="http://pa-backend-dev.westeurope.cloudapp.azure.com">
                </div>
                <div class="form-group">
                    <label>Test Mode</label>
                    <div class="test-mode">
                        <div class="mode-card selected" data-mode="single">
                            <h3>👤 Single User</h3>
                        </div>
                        <div class="mode-card" data-mode="multi">
                            <h3>👥 Multiple Users</h3>
                        </div>
                    </div>
                </div>
                <div class="form-group" id="singleUserOptions">
                    <label>Query</label>
                    <textarea id="singleQuery">What is RFID scanning?</textarea>
                </div>
                <div id="multiUserOptions">
    <div class="form-group">
        <label>Upload Questions File (CSV/Excel)</label>
        <p class="info-box" style="font-size: 14px;">
            Each row represents one user and their question. Number of users equals number of rows.
        </p>
        <div class="file-upload-area" id="loadFileUploadArea">
            <div style="font-size: 48px;">📁</div>
            <p>Click to upload or drag & drop</p>
            <p style="font-size: 12px; color: #666;">CSV or Excel with a 'question' column</p>
            <p style="font-size: 11px; color: #999;">Each row = 1 user, Users count = Row count</p>
            <input type="file" id="loadFileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div class="file-info" id="loadFileInfo">
            <strong>File:</strong> <span id="loadFileName"></span><br>
            <strong>Questions:</strong> <span id="loadFileRows"></span>
        </div>
    </div>
</div>
            
                    
                    
                <button type="button" class="btn-execute" id="loadBtn">🚀 Run Load Test</button>
            </form>
            <div class="loading" id="loading2">
                <div class="spinner"></div>
                <p>Running...</p>
            </div>

            <button class="btn-download" id="downloadReportBtnLoad" style="display:none; margin-top:12px;">Download Load Report</button>

            <div class="results" id="results2">
                <div class="stats-grid" id="statsGrid2"></div>
                <table class="result-table">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Query</th>
                            <th>Status</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody id="resultBody2"></tbody>
                </table>
            </div>
        </div>
    </div>

    <div id="fullModal" aria-hidden="true">
      <button id="fullModalClose" aria-label="Close">Close</button>
      <h3>Full Response</h3>
      <pre id="fullModalContent"></pre>
    </div>

    <script>
        function escapeHtml(str) {
            if (!str) return '';
            return String(str).replace(/[&<>"']/g, function(m) {
                return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m];
            });
        }

        var accuracyHasResults = false;
        var loadHasResults = false;

        document.querySelectorAll('.tab').forEach(function(tab) {
            tab.addEventListener('click', function() {
                var tabName = this.getAttribute('data-tab');
                document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
                document.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
                this.classList.add('active');
                document.getElementById(tabName === 'accuracy' ? 'accuracyTab' : 'loadTab').classList.add('active');

                if (tabName === 'accuracy') {
                    document.getElementById('downloadReportBtnAccuracy').style.display = accuracyHasResults ? 'inline-block' : 'none';
                    document.getElementById('downloadReportBtnLoad').style.display = 'none';
                } else {
                    document.getElementById('downloadReportBtnLoad').style.display = loadHasResults ? 'inline-block' : 'none';
                    document.getElementById('downloadReportBtnAccuracy').style.display = 'none';
                }
            });
        });
        
        document.getElementById('llmProvider').addEventListener('change', function() {
            var azureConfig = document.getElementById('azureConfig');
            var openaiConfig = document.getElementById('openaiConfig');
            if (this.value === 'azure') {
                azureConfig.style.display = 'block';
                openaiConfig.style.display = 'none';
            } else {
                azureConfig.style.display = 'none';
                openaiConfig.style.display = 'block';
            }
        });

        // Show/hide LLM config based on eval method
        document.querySelectorAll('input[name="evalMethod"]').forEach(function(radio) {
            radio.addEventListener('change', function() {
                var llmConfig = document.getElementById('llmConfig');
                if (this.value === 'similarity') {
                    llmConfig.style.display = 'none';
                } else {
                    llmConfig.style.display = 'block';
                }
            });
        });

        // Load testing: toggle between manual and file query
       

        // File upload for accuracy testing
        var uploadedFile = null;
        var fileUploadArea = document.getElementById('fileUploadArea');
        var fileInput = document.getElementById('fileInput');
        
        fileUploadArea.addEventListener('click', function() {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                uploadedFile = e.target.files[0];
                var formData = new FormData();
                formData.append('file', uploadedFile);
                
                fetch('/validate_file', {
                    method: 'POST',
                    body: formData
                })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    document.getElementById('fileName').textContent = uploadedFile.name;
                    document.getElementById('fileRows').textContent = data.row_count;
                    document.getElementById('fileInfo').classList.add('show');
                    document.getElementById('accuracyBtn').disabled = false;
                });
            }
        });

        // File upload for load testing (questions)
        var uploadedLoadFile = null;
        var loadFileUploadArea = document.getElementById('loadFileUploadArea');
        var loadFileInput = document.getElementById('loadFileInput');
        
        loadFileUploadArea.addEventListener('click', function() {
            loadFileInput.click();
        });
        
        loadFileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                uploadedLoadFile = e.target.files[0];
                var formData = new FormData();
                formData.append('file', uploadedLoadFile);
                
                fetch('/validate_load_file', {
                    method: 'POST',
                    body: formData
                })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        uploadedLoadFile = null;
                        return;
                    }
                    document.getElementById('loadFileName').textContent = uploadedLoadFile.name;
                    document.getElementById('loadFileRows').textContent = data.row_count + ' questions';
                    document.getElementById('loadFileInfo').classList.add('show');
                });
            }
        });

        document.getElementById('downloadReportBtnAccuracy').addEventListener('click', function() {
            window.location.href = '/download_report?type=accuracy&format=html';
        });
        document.getElementById('downloadReportBtnLoad').addEventListener('click', function() {
            window.location.href = '/download_report?type=load&format=html';
        });

     document.getElementById('accuracyBtn').addEventListener('click', function() {
    // Use fixed phoenix_custom instead of reading radio button
    var evalMethod = 'phoenix_custom'; 

    var formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('base_url', document.getElementById('baseUrl1').value);
    formData.append('user_id', document.getElementById('userId1').value);
    formData.append('eval_method', evalMethod);
    
    // Always need LLM config for phoenix_custom
    var provider = document.getElementById('llmProvider').value;
    if (provider === 'azure') {
        formData.append('azure_deployment', document.getElementById('azureDeployment').value.trim());
        formData.append('azure_endpoint', document.getElementById('azureEndpoint').value.trim());
        formData.append('azure_api_version', document.getElementById('azureApiVersion').value.trim());
        formData.append('judge_api_key', document.getElementById('azureApiKey').value.trim());
        formData.append('judge_model', 'azure');
    } else {
        formData.append('judge_api_key', document.getElementById('judgeApiKey').value.trim());
        formData.append('judge_model', document.getElementById('judgeModel').value);
    }
    
    // Rest of your existing code remains the same
    var btn = this;
    btn.disabled = true;
    document.getElementById('loading1').classList.add('show');
    
    // Existing fetch call...
            
            fetch('/run_accuracy_test', { method: 'POST', body: formData })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                var html = '';
                html += '<div class="stat-card"><div class="stat-value">' + data.stats.total_tests + '</div><div class="stat-label">Tests</div></div>';
                html += '<div class="stat-card ' + (data.stats.avg_accuracy >= 70 ? 'success' : 'warning') + '"><div class="stat-value">' + data.stats.avg_accuracy + '%</div><div class="stat-label">Avg Score</div></div>';
                html += '<div class="stat-card"><div class="stat-value">' + data.stats.avg_time + 'ms</div><div class="stat-label">Avg Time</div></div>';
                document.getElementById('statsGrid1').innerHTML = html;
                
                var tbody = document.getElementById('resultBody1');
                tbody.innerHTML = '';
                data.results.forEach(function(r, i) {
                    var row = tbody.insertRow();
                    row.insertCell(0).textContent = i + 1;
                    row.insertCell(1).textContent = r.question.substring(0, 50) + '...';
                    row.insertCell(2).textContent = r.expected_answer.substring(0, 50) + '...';

                    var previewCell = row.insertCell(3);
                    var previewText = (r.actual_response || '').substring(0, 200);
                    var truncated = (r.actual_response && r.actual_response.length > 200) ? '...' : '';
                    previewCell.innerHTML = '<span class="preview-text">' + escapeHtml(previewText) + truncated + '</span>' +
                                            ' <button class="view-full-btn" data-full="' + encodeURIComponent(r.actual_response || '') + '" style="margin-left:8px;padding:4px 8px;">View</button>';

                    var scoreCell = row.insertCell(4);
                    var scoreClass = r.accuracy >= 70 ? 'accuracy-high' : r.accuracy >= 40 ? 'accuracy-medium' : 'accuracy-low';
                    scoreCell.innerHTML = '<span class="accuracy-badge ' + scoreClass + '">' + r.accuracy.toFixed(1) + '%</span>';
                    row.insertCell(5).textContent = r.reasoning || '-';
                    row.insertCell(6).textContent = r.time_ms.toFixed(0) + 'ms';
                });
                
                document.getElementById('results1').classList.add('show');
                accuracyHasResults = true;
                document.getElementById('downloadReportBtnAccuracy').style.display = 'inline-block';
            })
            .finally(function() {
                btn.disabled = false;
                document.getElementById('loading1').classList.remove('show');
            });
        });
        
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList && e.target.classList.contains('view-full-btn')) {
                var encoded = e.target.getAttribute('data-full') || '';
                try {
                    var text = decodeURIComponent(encoded);
                } catch (err) {
                    var text = encoded;
                }
                document.getElementById('fullModalContent').textContent = text;
                document.getElementById('fullModal').style.display = 'block';
                document.getElementById('fullModal').setAttribute('aria-hidden', 'false');
            }
        });
        document.getElementById('fullModalClose').addEventListener('click', function() {
            document.getElementById('fullModal').style.display = 'none';
            document.getElementById('fullModal').setAttribute('aria-hidden', 'true');
            document.getElementById('fullModalContent').textContent = '';
        });

        var selectedMode = 'single';
        document.querySelectorAll('.mode-card').forEach(function(card) {
            card.addEventListener('click', function() {
                document.querySelectorAll('.mode-card').forEach(function(c) { c.classList.remove('selected'); });
                this.classList.add('selected');
                selectedMode = this.getAttribute('data-mode');
                if (selectedMode === 'single') {
                    document.getElementById('singleUserOptions').style.display = 'block';
                    document.getElementById('multiUserOptions').style.display = 'none';
                } else {
                    document.getElementById('singleUserOptions').style.display = 'none';
                    document.getElementById('multiUserOptions').style.display = 'block';
                }
            });
        });
        
        // Replace the loadBtn click handler with this simplified version
document.getElementById('loadBtn').addEventListener('click', function() {
    var selectedMode = document.querySelector('.mode-card.selected').getAttribute('data-mode');
    
    if (selectedMode === 'single') {
        // Single user mode - simple query
        var formData = {
            base_url: document.getElementById('baseUrl2').value,
            mode: 'single',
            query: document.getElementById('singleQuery').value
        };
        
        runLoadTest(formData);
    } else {
        // Multi-user mode - ONLY file upload option
        if (!uploadedLoadFile) {
            alert('Please upload a questions file first!');
            return;
        }
        
        var formDataObj = new FormData();
        formDataObj.append('file', uploadedLoadFile);
        formDataObj.append('base_url', document.getElementById('baseUrl2').value);
        formDataObj.append('mode', 'multi');
        
        var btn = document.getElementById('loadBtn');
        btn.disabled = true;
        document.getElementById('loading2').classList.add('show');
        
        fetch('/run_load_test_with_file', {
            method: 'POST',
            body: formDataObj
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            displayLoadResults(data);
        })
        .catch(function(err) {
            alert('Error: ' + err);
        })
        .finally(function() {
            btn.disabled = false;
            document.getElementById('loading2').classList.remove('show');
        });
    }
});
        
        function runLoadTest(formData) {
            var btn = document.getElementById('loadBtn');
            btn.disabled = true;
            document.getElementById('loading2').classList.add('show');
            
            fetch('/run_load_test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                displayLoadResults(data);
            })
            .catch(function(err) {
                alert('Error: ' + err);
            })
            .finally(function() {
                btn.disabled = false;
                document.getElementById('loading2').classList.remove('show');
            });
        }
        
        function displayLoadResults(data) {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            var html = '';
            html += '<div class="stat-card"><div class="stat-value">' + data.stats.total_requests + '</div><div class="stat-label">Requests</div></div>';
            html += '<div class="stat-card ' + (data.stats.success_rate >= 90 ? 'success' : 'warning') + '"><div class="stat-value">' + data.stats.success_rate + '%</div><div class="stat-label">Success</div></div>';
            html += '<div class="stat-card"><div class="stat-value">' + data.stats.avg_latency + 'ms</div><div class="stat-label">Avg Latency</div></div>';
            document.getElementById('statsGrid2').innerHTML = html;
            
            var tbody = document.getElementById('resultBody2');
            tbody.innerHTML = '';
            data.results.forEach(function(r) {
                var row = tbody.insertRow();
                row.insertCell(0).textContent = r.user_id;
                row.insertCell(1).textContent = r.query.substring(0, 60) + (r.query.length > 60 ? '...' : '');
                var statusCell = row.insertCell(2);
                statusCell.innerHTML = '<span class="status-' + (r.success ? 'success' : 'error') + '">' + (r.success ? '✓ ' + r.status : '✗ ' + r.status) + '</span>';
                row.insertCell(3).textContent = r.latency_ms.toFixed(0) + 'ms';
            });
            
            document.getElementById('results2').classList.add('show');
            loadHasResults = true;
            document.getElementById('downloadReportBtnLoad').style.display = 'inline-block';
        }
    </script>
</body>
</html>"""

@app.route('/validate_file', methods=['POST'])
def validate_file():
    try:
        file = request.files.get('file')
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.lower()
        return jsonify({"row_count": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/validate_load_file', methods=['POST'])
def validate_load_file():
    """Validate load testing questions file"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
            
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        df.columns = df.columns.str.strip().str.lower()
        
        # Check for question column
        question_cols = [c for c in df.columns if 'question' in c or 'query' in c]
        if not question_cols:
            return jsonify({"error": "File must have a 'question' or 'query' column"}), 400
        
        return jsonify({
            "row_count": len(df),
            "question_column": question_cols[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/download_report', methods=['GET'])
def download_report():
    report_type = request.args.get('type', 'accuracy')
    fmt = request.args.get('format', 'csv').lower()

    if report_type == 'accuracy':
        results = last_accuracy_results
        stats = last_accuracy_stats
        if not results:
            return jsonify({"error": "No accuracy results to download"}), 400

        if fmt == 'html':
            html = generate_accuracy_html(results, stats)
            return Response(html, mimetype="text/html",
                            headers={"Content-Disposition": "attachment; filename=accuracy_report.html"})
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Question", "Expected", "Actual", "Score", "Reasoning", "Time_ms"])
            for r in results:
                writer.writerow([
                    r.get("question", ""),
                    r.get("expected_answer", ""),
                    r.get("actual_response", ""),
                    r.get("accuracy", ""),
                    r.get("reasoning", ""),
                    r.get("time_ms", "")
                ])
            output.seek(0)
            return Response(output.getvalue(), mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=accuracy_report.csv"})

    elif report_type == 'load':
        results = last_load_results
        stats = last_load_stats
        if not results:
            return jsonify({"error": "No load results to download"}), 400

        if fmt == 'html':
            html = generate_load_html(results, stats)
            return Response(html, mimetype="text/html",
                            headers={"Content-Disposition": "attachment; filename=load_report.html"})
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["User", "Query", "Status", "Latency_ms", "Success"])
            for r in results:
                writer.writerow([
                    r.get("user_id", ""),
                    r.get("query", ""),
                    r.get("status", ""),
                    r.get("latency_ms", ""),
                    r.get("success", "")
                ])
            output.seek(0)
            return Response(output.getvalue(), mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=load_report.csv"})
    else:
        return jsonify({"error": "Unknown report type"}), 400

@app.route('/run_accuracy_test', methods=['POST'])
def run_accuracy_test():
    global tracer
    global last_accuracy_results, last_accuracy_stats
    
    try:
        file = request.files.get('file')
        base_url = request.form.get('base_url')
        user_id = request.form.get('user_id')
        eval_method = request.form.get('eval_method', 'phoenix')
        judge_api_key = request.form.get('judge_api_key', '')
        judge_model = request.form.get('judge_model', 'gpt-4o')
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        df.columns = df.columns.str.strip().str.lower()
        question_col = [c for c in df.columns if 'question' in c or 'query' in c][0]
        answer_col = [c for c in df.columns if 'expected' in c or 'answer' in c][0]
        
        endpoint = f"{base_url}/api/v1/query"
        results = []
        total_start = time.time()
        
        if tracer:
            span_ctx = tracer.start_as_current_span("accuracy_test_phoenix")
        else:
            span_ctx = None

        if span_ctx:
            span_ctx.__enter__()

        try:
            for idx, row in df.iterrows():
                question = str(row[question_col]).strip()
                expected = str(row[answer_col]).strip()
                
                form_data = {
                    "query": question,
                    "user_id": user_id,
                    "session_id": str(uuid.uuid4()),
                    "include_sources": False,
                    "include_images": True
                }
                
                start_time = time.time()
                try:
                    response = requests.post(endpoint, data=form_data, timeout=60)
                    duration = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        actual_response = response.text
                        try:
                            response_json = json.loads(actual_response)
                            if isinstance(response_json, dict):
                                actual_response = response_json.get('answer', response_json.get('response', actual_response))
                        except:
                            pass
                        actual_response = clean_response_text(actual_response)
                    else:
                        actual_response = f"Error: HTTP {response.status_code}"
                    
                    reasoning = ""
                    
                    # Use Phoenix evaluation methods
                    if eval_method == 'phoenix' and judge_api_key:
                        azure_deployment = request.form.get('azure_deployment')
                        azure_endpoint = request.form.get('azure_endpoint')
                        azure_api_version = request.form.get('azure_api_version', '2024-08-01-preview')
                        
                        print(f"🤖 Phoenix LLM Judge for Q{idx+1}...")
                        accuracy, reasoning = phoenix_llm_evaluate(
                            question, expected, actual_response,
                            judge_api_key, judge_model,
                            azure_endpoint, azure_deployment, azure_api_version
                        )
                        print(f"✅ Phoenix Q{idx+1}: {accuracy:.1f}%")
                        
                    elif eval_method == 'phoenix_custom' and judge_api_key:
                        azure_deployment = request.form.get('azure_deployment')
                        azure_endpoint = request.form.get('azure_endpoint')
                        azure_api_version = request.form.get('azure_api_version', '2024-08-01-preview')
                        
                        print(f"🎯 Phoenix Custom Judge for Q{idx+1}...")
                        accuracy, reasoning = custom_phoenix_evaluator(
                            question, expected, actual_response,
                            judge_api_key, judge_model,
                            azure_endpoint, azure_deployment, azure_api_version
                        )
                        print(f"✅ Phoenix Custom Q{idx+1}: {accuracy:.1f}%")
                        
                    else:
                        accuracy = calculate_similarity(expected, actual_response)
                        reasoning = "Similarity-based evaluation"
                        print(f"📊 Similarity Q{idx+1}: {accuracy:.1f}%")
                    
                    results.append({
                        "question": question,
                        "expected_answer": expected,
                        "actual_response": actual_response,
                        "accuracy": accuracy,
                        "reasoning": reasoning,
                        "time_ms": duration
                    })
                    
                except Exception as e:
                    results.append({
                        "question": question,
                        "expected_answer": expected,
                        "actual_response": f"Error: {str(e)}",
                        "accuracy": 0.0,
                        "reasoning": "",
                        "time_ms": 0
                    })
        finally:
            if span_ctx:
                span_ctx.__exit__(None, None, None)
        
        total_time = time.time() - total_start
        accuracies = [r['accuracy'] for r in results]
        times = [r['time_ms'] for r in results]
        
        stats = {
            "total_tests": len(results),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 1) if accuracies else 0,
            "avg_time": round(sum(times) / len(times), 0) if times else 0,
            "total_time": round(total_time, 2)
        }
        
        last_accuracy_results = results
        last_accuracy_stats = stats

        return jsonify({"results": results, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_load_test', methods=['POST'])
def run_load_test():
    global tracer
    global last_load_results, last_load_stats
    try:
        data = request.json
        base_url = data.get('base_url')
        mode = data.get('mode')
        endpoint = f"{base_url}/api/v1/query"
        
        results = []
        lock = threading.Lock()
        
        def simulate_user(user_info):
            form_data = {
                "query": user_info["query"],
                "user_id": user_info["user_id"],
                "session_id": str(uuid.uuid4()),
                "include_sources": False,
                "include_images": False
            }
            
            try:
                api_start = time.time()
                response = requests.post(endpoint, data=form_data, timeout=60)
                api_duration = (time.time() - api_start) * 1000
                
                result = {
                    "user_id": user_info["user_id"],
                    "query": user_info["query"],
                    "status": response.status_code,
                    "latency_ms": api_duration,
                    "success": response.status_code == 200
                }
                
                with lock:
                    results.append(result)
                    print(f"✅ {user_info['user_id']}: {api_duration:.0f}ms - {user_info['query'][:40]}")
            except Exception as e:
                with lock:
                    results.append({
                        "user_id": user_info["user_id"],
                        "query": user_info["query"],
                        "status": "Error",
                        "latency_ms": 0,
                        "success": False
                    })
                    print(f"❌ {user_info['user_id']}: Error - {str(e)}")
        
        test_start = time.time()
        
        if mode == 'single':
            simulate_user({"user_id": "user_001", "query": data.get('query')})
        else:
            # Multi-user with manual query (all same question)
            num_users = data.get('num_users', 5)
            query = data.get('query', 'What is RFID?')
            users = [{"user_id": f"user_{i+1:03d}", "query": query} for i in range(num_users)]
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user, u) for u in users]
                for f in as_completed(futures):
                    f.result()
        
        total_time = time.time() - test_start
        successful = [r for r in results if r.get('success')]
        latencies = [r.get('latency_ms', 0) for r in successful] if successful else [0]
        
        stats = {
            "total_requests": len(results),
            "success_rate": round((len(successful) / len(results) * 100), 1) if results else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 0) if latencies else 0,
            "total_time": round(total_time, 2)
        }
        
        last_load_results = results
        last_load_stats = stats

        return jsonify({"results": results, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_load_test_with_file', methods=['POST'])
def run_load_test_with_file():
    """Run load test with questions from uploaded file"""
    global tracer
    global last_load_results, last_load_stats
    
    try:
        file = request.files.get('file')
        base_url = request.form.get('base_url')
        
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Parse file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        df.columns = df.columns.str.strip().str.lower()
        
        # Find question column
        question_cols = [c for c in df.columns if 'question' in c or 'query' in c]
        if not question_cols:
            return jsonify({"error": "File must have a 'question' or 'query' column"}), 400
        
        question_col = question_cols[0]
        endpoint = f"{base_url}/api/v1/query"
        
        results = []
        lock = threading.Lock()
        
        def simulate_user(user_info):
            form_data = {
                "query": user_info["query"],
                "user_id": user_info["user_id"],
                "session_id": str(uuid.uuid4()),
                "include_sources": False,
                "include_images": False
            }
            
            try:
                api_start = time.time()
                response = requests.post(endpoint, data=form_data, timeout=60)
                api_duration = (time.time() - api_start) * 1000
                
                result = {
                    "user_id": user_info["user_id"],
                    "query": user_info["query"],
                    "status": response.status_code,
                    "latency_ms": api_duration,
                    "success": response.status_code == 200
                }
                
                with lock:
                    results.append(result)
                    print(f"✅ {user_info['user_id']}: {api_duration:.0f}ms - {user_info['query'][:40]}")
            except Exception as e:
                with lock:
                    results.append({
                        "user_id": user_info["user_id"],
                        "query": user_info["query"],
                        "status": "Error",
                        "latency_ms": 0,
                        "success": False
                    })
                    print(f"❌ {user_info['user_id']}: Error - {str(e)}")
        
        # Create user list from file (each row = 1 user with different question)
        users = []
        for idx, row in df.iterrows():
            question = str(row[question_col]).strip()
            if question and question.lower() != 'nan':
                users.append({
                    "user_id": f"user_{idx+1:03d}",
                    "query": question
                })
        
        if not users:
            return jsonify({"error": "No valid questions found in file"}), 400
        
        print(f"\n🚀 Starting load test with {len(users)} users from file...")
        test_start = time.time()
        
        # Execute concurrent requests
        max_workers = min(len(users), 20)  # Limit concurrent workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(simulate_user, u) for u in users]
            for f in as_completed(futures):
                f.result()
        
        total_time = time.time() - test_start
        successful = [r for r in results if r.get('success')]
        latencies = [r.get('latency_ms', 0) for r in successful] if successful else [0]
        
        stats = {
            "total_requests": len(results),
            "success_rate": round((len(successful) / len(results) * 100), 1) if results else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 0) if latencies else 0,
            "total_time": round(total_time, 2)
        }
        
        last_load_results = results
        last_load_stats = stats
        
        print(f"✅ Load test completed: {len(successful)}/{len(results)} successful")

        return jsonify({"results": results, "stats": stats})
    except Exception as e:
        print(f"❌ Load test error: {e}")
        return jsonify({"error": str(e)}), 500
        mode = data.get('mode')
        endpoint = f"{base_url}/api/v1/query"
        
        results = []
        lock = threading.Lock()
        
        def simulate_user(user_info):
            form_data = {
                "query": user_info["query"],
                "user_id": user_info["user_id"],
                "session_id": str(uuid.uuid4()),
                "include_sources": False,
                "include_images": False
            }
            
            try:
                api_start = time.time()
                response = requests.post(endpoint, data=form_data, timeout=60)
                api_duration = (time.time() - api_start) * 1000
                
                result = {
                    "user_id": user_info["user_id"],
                    "query": user_info["query"],
                    "status": response.status_code,
                    "latency_ms": api_duration,
                    "success": response.status_code == 200
                }
                
                with lock:
                    results.append(result)
                    print(f"✅ {user_info['user_id']}: {api_duration:.0f}ms")
            except Exception as e:
                with lock:
                    results.append({
                        "user_id": user_info["user_id"],
                        "query": user_info["query"],
                        "latency_ms": 0,
                        "success": False
                    })
        
        test_start = time.time()
        
        if mode == 'single':
            simulate_user({"user_id": "user", "query": data.get('query')})
        else:
            num_users = data.get('num_users', 5)
            query = data.get('query', 'What is RFID?')
            users = [{"user_id": f"user_{i+1:03d}", "query": query} for i in range(num_users)]
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user, u) for u in users]
                for f in as_completed(futures):
                    f.result()
        
        total_time = time.time() - test_start
        successful = [r for r in results if r.get('success')]
        latencies = [r.get('latency_ms', 0) for r in successful] if successful else [0]
        
        stats = {
            "total_requests": len(results),
            "success_rate": round((len(successful) / len(results) * 100), 1) if results else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 0) if latencies else 0,
            "total_time": round(total_time, 2)
        }
        
        last_load_results = results
        last_load_stats = stats

        return jsonify({"results": results, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        print("\n" + "="*50)
        print("🚀 Starting Phoenix Testing Suite with LLM-as-Judge")
        print("="*50)
        phoenix_url = init_phoenix()
        print(f"\n📊 Phoenix Dashboard: {phoenix_url}")
        print(f"🌐 Web Interface: http://localhost:5000")
        print(f"\n💡 Load Testing Features:")
        print("   - Single User: Test with one query")
        print("   - Multiple Users (Manual): All users ask same question")
        print("   - Multiple Users (File): Each user asks different question from CSV/Excel")
        print("\n" + "="*50 + "\n")
        
        app.run(debug=True, port=5000, use_reloader=False)
    finally:
        if phoenix_session:
            try:
                phoenix_session.close()
            except Exception:
                pass