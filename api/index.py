from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
        <head><title>BlackBerry Chat Test</title></head>
        <body style="background:#000;color:#0f0;font-family:monospace;padding:20px;">
            <h1>✓ Flask is Working!</h1>
            <p>Server is running on Vercel</p>
            <p><a href="/health" style="color:#0ff;">Test Health Endpoint</a></p>
        </body>
    </html>
    '''

@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "All systems operational"})

if __name__ == '__main__':
    app.run(debug=True)
